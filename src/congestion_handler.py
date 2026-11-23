"""
congestion_handler.py
=====================
Selective QUBO for trapped-ion shuttling: decides, per blocked path,
whether the greedy SHAW walk is good enough or a local QUBO should plan
the moves of every ion in the congested region.

  Phase 3  Severity metrics after a blockage is detected
             kappa  local contention: occupied fraction of the path
             rho    routing regret:   (C_est - C_LB) / C_LB, C_est = C_LB + 2|B|
             d      recursive blockage depth
  Phase 4  Trigger: (kappa > kappa_th AND rho > rho_th) OR d > d_th,
           then a cheap filter: skip if the greedy walk already meets
           the lower bound (optionally a learned slack rule as well).
  Phase 5  Window W: the path, the goal trap's slots, every trap slot
           the path touches, and the neighbours of blocked positions.
           Paths whose window exceeds window_max_nodes are cut at trap
           slots into chunks that fit, planned one after another on a
           scratch copy of the board.
  Phase 6  Formulate + solve + decode (qubo_formulator / qubo_solver /
           solution_decoder). The QUBO plans the moves of EVERY ion in W,
           warm-started from the greedy walk. A plan is accepted only if
           it costs less than the greedy walk on the same board (by
           default in operations: move = swap = 1, vacate = 2) and leaves
           ions no further from their next gate partners; otherwise the
           greedy walk is used.

Called by ShawRoutingPass as::

    resolution = handler.resolve_path_congestion(path, placement, pg)
    # resolution.plan -> execute the QUBO plan, else walk resolution.path
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from qubo_formulator import InfeasibleWindow, QUBOFormulator, QUBOProblem, WindowInfo
from qubo_solver import QUBOSolver
from solution_decoder import MovePlan, SolutionDecoder
from walk_planner import (
    UNIT_COSTS, BoardView, ops_cost, plan_walk, trap_of, walk_cost, walk_ops,
)

logger = logging.getLogger("shaw_router.congestion")


@dataclass
class Resolution:
    path: List[Any]                   # path for the greedy walk
    plan: Optional[MovePlan] = None   # accepted QUBO plan (overrides path)
    outcome: str = "clear"            # clear | rerouted | walk | qubo_accepted


class CongestionHandler:
    """
    Parameters
    ----------
    kappa_threshold, rho_threshold, depth_threshold : QUBO trigger thresholds.
        The defaults (0, 0) consider every blocked path: measured on the
        200-circuit suite, stricter thresholds only gave up savings while
        saving little time, because the QUBO is already skipped whenever
        the greedy walk is provably optimal.
    qubo_enabled     : False gives the pure greedy SHAW baseline
    window_max_nodes : largest window solved at once; longer paths are
                       planned in chunks
    min_saving       : accept a QUBO plan only if it saves at least this
                       many operations versus the greedy walk
    lookahead_weight : QUBO weight on ending near each ion's next partner
    time_slack       : QUBO time steps beyond the mover's shortest route
    cost_model       : what plans are compared on (walk_planner.UNIT_COSTS:
                       operation count, the default; TIME_COSTS_US: time)
    learned_trigger  : skip windows where the greedy walk has little slack
                       over the lower bound (< 2 ops with <= 3 ions, < 4 ops
                       otherwise); rule learned by experiments/learn_trigger.py
    max_reroute_candidates : compatibility switch: 0 disables rerouting;
                            positive values enable one free-position BFS
    planning_mode   : "qubo" for full search, "incumbent" for a deterministic
                      goal-feasible greedy prefix without building a BQM
    staged_effort    : optional short tabu stage (0 disables it, the default)
    max_bqm_variables : skip solving larger models, including auxiliary bits;
                        0 disables this post-build cap
    max_time_horizon : pre-build horizon cap; 0 disables it
    solver_options   : extra QUBOSolver arguments
    seed             : base seed; each solve gets its own derived seed, so
                       runs are reproducible
    """

    def __init__(
        self,
        kappa_threshold: float = 0.0,
        rho_threshold: float = 0.0,
        depth_threshold: int = 2,
        qubo_enabled: bool = True,
        window_max_nodes: int = 16,
        min_saving: int = 1,
        lookahead_weight: float = 0.5,
        time_slack: int = 2,
        cost_model: Optional[Dict[str, float]] = None,
        learned_trigger: bool = False,
        max_reroute_candidates: int = 8,
        solver_options: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        staged_effort: int = 0,
        max_bqm_variables: int = 3000,
        max_time_horizon: int = 24,
        planning_mode: str = "qubo",
    ) -> None:
        if planning_mode not in ("qubo", "incumbent"):
            raise ValueError("planning_mode must be qubo or incumbent")
        self.planning_mode = planning_mode
        if max_bqm_variables < 0 or max_time_horizon < 0 or staged_effort < 0:
            raise ValueError("Model budgets and staged effort must be nonnegative")
        if window_max_nodes < 2 or time_slack < 0 or min_saving < 0:
            raise ValueError("Invalid window size, time slack, or minimum saving")
        if cost_model is not None and (set(UNIT_COSTS) - set(cost_model)
                                     or any(value <= 0 for value in cost_model.values())):
            raise ValueError("Cost model must provide positive primitive costs")
        self.kappa_threshold = kappa_threshold
        self.rho_threshold = rho_threshold
        self.depth_threshold = depth_threshold
        self.qubo_enabled = qubo_enabled
        self.window_max_nodes = window_max_nodes
        self.min_saving = min_saving
        self.lookahead_weight = lookahead_weight
        self.time_slack = time_slack
        self.costs = cost_model or UNIT_COSTS
        self.learned_trigger = learned_trigger
        self.max_reroute_candidates = max_reroute_candidates
        self.solver_options = solver_options or {}
        self.seed = seed
        self.staged_effort = staged_effort
        self.max_bqm_variables = max_bqm_variables
        self.max_time_horizon = max_time_horizon
        self._solve_counter = 0
        self.stats: Dict[str, Any] = {
            "total_calls": 0,
            "clear_paths": 0,
            "blocked_events": 0,
            "rerouted": 0,
            "qubo_triggers": 0,
            "qubo_solved": 0,      # triggers that reached the solver
            "qubo_accepted": 0,    # plans cheaper than the greedy walk
            "qubo_executed": 0,    # plans actually executed by the router
            "ops_saved": 0,        # greedy cost - plan cost, summed
            "qubo_time_s": 0.0,
            "build_time_s": 0.0,
            "solve_time_s": 0.0,
            "decode_time_s": 0.0,
            "early_optimal": 0,
            "incumbent_plans": 0,
            "window_records": [],
            "chunked_plans": 0,    # accepted plans solved in several windows
            "qubo_skips": Counter(),  # why a considered path did not get a QUBO plan
            "kappa_values": [],
            "rho_values": [],
            "depth_values": [],
        }

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def resolve_path_congestion(
        self,
        path: List[Any],
        placement: Any,
        pg: Optional[Any] = None,
        next_partner_pos: Optional[Dict[Any, Any]] = None,
        next_partner_ions: Optional[Dict[Any, Any]] = None,
    ) -> Resolution:
        """
        next_partner_pos: {ion: position of its next gate partner} from
        the router's lookahead; lets the QUBO prefer plans that leave
        ions well placed for upcoming gates.
        """
        self.stats["total_calls"] += 1
        blocked = self._detect_blockages(path, placement)
        if not blocked:
            self.stats["clear_paths"] += 1
            return Resolution(path)

        self.stats["blocked_events"] += 1
        kappa = self._compute_kappa(path, placement)
        rho = self._compute_rho(blocked, len(path) - 1)
        depth = self._compute_blockage_depth(blocked, placement, pg)
        self.stats["kappa_values"].append(kappa)
        self.stats["rho_values"].append(rho)
        self.stats["depth_values"].append(depth)

        greedy = Resolution(path, outcome="walk")
        if pg is not None:
            alt = self._find_unblocked_path(path[0], path[-1], placement, pg)
            if alt is not None and alt != path:
                self.stats["rerouted"] += 1
                greedy = Resolution(alt, outcome="rerouted")

        if pg is not None and self.qubo_enabled and self.should_trigger_qubo(kappa, rho, depth):
            self.stats["qubo_triggers"] += 1
            started = time.perf_counter()
            plan = self._try_qubo(path, blocked, placement, pg, greedy.path,
                                  next_partner_pos or {}, next_partner_ions or {})
            self.stats["qubo_time_s"] += time.perf_counter() - started
            if plan is not None:
                self.stats["qubo_accepted"] += 1
                return Resolution(greedy.path, plan=plan, outcome="qubo_accepted")
        return greedy

    # -----------------------------------------------------------------------
    # Selective QUBO
    # -----------------------------------------------------------------------

    def _try_qubo(
        self,
        path: List[Any],
        blocked: List[Any],
        placement: Any,
        pg: Any,
        greedy_path: List[Any],
        next_partner_pos: Dict[Any, Any],
        next_partner_ions: Optional[Dict[Any, Any]] = None,
    ) -> Optional[MovePlan]:
        """
        A plan that is cheaper than the greedy walk by at least
        `min_saving` operations and leaves ions no further from their next
        gate partners than the greedy walk would.

        Long paths are planned in chunks: the path is cut at trap slots
        into the longest prefixes whose windows fit `window_max_nodes`,
        each chunk is solved on a scratch copy of the board (falling back
        to the greedy walk for that chunk if the QUBO cannot beat it), and
        the combined plan is compared with the greedy walk as a whole.
        """
        skips = self.stats["qubo_skips"]
        mover = placement.occupant_of(path[0])
        if mover is None:
            return None
        goal = goal_slots(pg, path[-1])
        greedy_cost = walk_cost(pg, placement, greedy_path, mover, self.costs)
        lower_bound = self._lower_bound(pg, path[0], goal)
        if greedy_cost - lower_bound < self.min_saving:
            skips["greedy_optimal"] += 1
            return None

        board = BoardView(pg, placement)
        ops: List[tuple] = []
        chunks = 0
        while board.position_of(mover) not in goal:
            remaining = pg.shortest_path(board.position_of(mover), path[-1])
            chunk = self._next_chunk(remaining, board, pg)
            if chunk is None or chunks > len(path):
                skips["window_too_large"] += 1
                return None
            chunks += 1
            hints = dict(next_partner_pos)
            if chunk[-1] != path[-1]:
                hints[mover] = path[-1]  # intermediate chunk: keep heading for the target
            chunk_ops = self._solve_window(chunk, board, pg, hints)
            board.apply(chunk_ops)
            ops += chunk_ops

        plan_cost = ops_cost(pg, placement, ops, self.costs)
        if greedy_cost - plan_cost < self.min_saving - 1e-9:
            skips["no_cheaper_plan"] += 1
            return None
        greedy_board = BoardView(pg, placement)
        greedy_board.apply(walk_ops(pg, placement, greedy_path, mover))
        if (layout_cost(pg, board, next_partner_pos, next_partner_ions)
                > layout_cost(pg, greedy_board, next_partner_pos, next_partner_ions)):
            skips["worse_layout"] += 1
            return None
        self.stats["ops_saved"] += greedy_cost - plan_cost
        self.stats["chunked_plans"] += chunks > 1
        moved = {op[1] for op in ops} | {op[2] for op in ops if op[0] == "swap"}
        return MovePlan(ops=ops, trajectories={
            ion: [placement.position_of(ion), board.position_of(ion)] for ion in moved
        })

    def _next_chunk(self, remaining: List[Any], board: BoardView, pg: Any) -> Optional[List[Any]]:
        """Longest prefix of `remaining` that ends on a trap slot in another
        trap than the mover's (or at the target) and whose window fits."""
        start_trap = trap_of(pg, remaining[0])
        for end in range(len(remaining) - 1, 0, -1):
            last = remaining[end]
            is_target = end == len(remaining) - 1
            trap = trap_of(pg, last)
            if not is_target and (trap is None or trap == start_trap):
                continue
            chunk = remaining[:end + 1]
            if len(self._window_nodes(chunk, board, pg)) <= self.window_max_nodes:
                return chunk
        return None

    def _solve_window(
        self,
        chunk: List[Any],
        board: BoardView,
        pg: Any,
        hints: Dict[Any, Any],
    ) -> List[tuple]:
        """Operations moving the mover along `chunk` into its end trap:
        the QUBO plan if it beats the greedy walk, else the greedy walk."""
        mover = board.occupant_of(chunk[0])
        greedy = walk_ops(pg, board, chunk, mover)
        blocked = self._detect_blockages(chunk, board)
        if not blocked:
            return greedy
        window = self.extract_window(chunk, blocked, board, pg)
        greedy_cost = ops_cost(pg, board, greedy, self.costs)
        if self.planning_mode == "incumbent":
            # A deterministic control and low-latency option: stop at the
            # first settled, gate-feasible prefix. No model or sampler call.
            scratch = BoardView(pg, board)
            prefix = []
            for op in greedy:
                scratch.apply([op])
                prefix.append(op)
                if (scratch.position_of(mover) in window.goal_nodes
                        and all(trap_of(pg, scratch.position_of(ion)) is not None
                                and (start not in window.goal_nodes
                                     or scratch.position_of(ion) in window.goal_nodes)
                                for ion, start in window.active_ions.items())):
                    break
            if ops_cost(pg, board, prefix, self.costs) < greedy_cost:
                self.stats["incumbent_plans"] += 1
                return prefix
            return greedy

        slack = greedy_cost - self._lower_bound(pg, chunk[0], window.goal_nodes)
        if slack <= 1e-9:
            return greedy
        if self.learned_trigger and self.costs == UNIT_COSTS and slack < (2 if len(window.active_ions) <= 3 else 4):
            self.stats["qubo_skips"]["learned_trigger"] += 1
            return greedy
        window.next_partner_pos = {ion: pos for ion, pos in hints.items()
                                   if ion in window.active_ions}

        greedy_traj = walk_trajectories(pg, board, chunk, window)
        if any(op[1] not in window.active_ions or
               (op[0] == "move" and op[2] not in window.window_nodes)
               for op in greedy):
            self.stats["qubo_skips"]["incumbent_outside_window"] += 1
            return greedy
        horizon = max(len(greedy_traj[mover]) - 1,
                      min(nx.shortest_path_length(pg.graph, chunk[0], g) for g in window.goal_nodes) + self.time_slack, 1)
        if self.max_time_horizon and horizon > self.max_time_horizon:
            self.stats["qubo_skips"]["horizon_budget"] += 1
            return greedy
        build_started = time.perf_counter()
        try:
            problem = QUBOFormulator(
                slack=self.time_slack, lookahead_weight=self.lookahead_weight, costs=self.costs
            ).build(
                window, pg, min_horizon=len(greedy_traj[mover]) - 1
            )
        except InfeasibleWindow as exc:
            logger.debug("QUBO skipped: %s", exc)
            self.stats["qubo_skips"]["infeasible_window"] += 1
            return greedy

        build_elapsed = time.perf_counter() - build_started
        self.stats["build_time_s"] += build_elapsed
        variables = len(problem.bqm.variables)
        if self.max_bqm_variables and variables > self.max_bqm_variables:
            self.stats["qubo_skips"]["variable_budget"] += 1
            return greedy
        self.stats["qubo_solved"] += 1
        self._solve_counter += 1
        solve_seed = 1000 + self._solve_counter + 1_000_000 * self.seed
        options = {"seed": solve_seed, **self.solver_options}
        # Flexible goal slots allow stopping before the exact endpoint of the
        # SHAW path. Preserve this valid prefix as a search incumbent.
        for step in range(len(greedy_traj[mover])):
            if (greedy_traj[mover][step] in window.goal_nodes
                    and all(trap_of(pg, positions[step]) is not None
                            and (start not in window.goal_nodes or positions[step] in window.goal_nodes)
                            for ion, start in window.active_ions.items()
                            for positions in [greedy_traj[ion]])):
                greedy_traj = {ion: positions[:step + 1] for ion, positions in greedy_traj.items()}
                break
        initial = seed_from(problem, greedy_traj)
        incumbent = greedy
        incumbent_cost = greedy_cost
        lower_bound = self._lower_bound(pg, chunk[0], window.goal_nodes)
        stages = [options]
        if (self.staged_effort > 0 and options.get("method", "tabu") == "tabu"
                and options.get("tabu_effort", 4000) > self.staged_effort):
            stages.insert(0, {**options, "tabu_effort": self.staged_effort, "num_reads": 1})
        record = {"variables": variables, "couplings": len(problem.bqm.quadratic),
                  "horizon": problem.time_horizon, "build_s": build_elapsed,
                  "solve_s": 0.0, "decode_s": 0.0, "stages": 0,
                  "greedy_cost": greedy_cost, "lower_bound": lower_bound}
        for stage in stages:
            solution = QUBOSolver(**stage).solve(problem, initial_state=initial)
            record["stages"] += 1
            record["solve_s"] += solution.solve_time_s
            decode_started = time.perf_counter()
            plans = SolutionDecoder().valid_plans(solution)
            for candidate in plans:
                try:
                    price = ops_cost(pg, board, candidate.ops, self.costs)
                except ValueError:
                    continue
                if price < incumbent_cost - 1e-9:
                    incumbent, incumbent_cost = candidate.ops, price
            record["decode_s"] += time.perf_counter() - decode_started
            if incumbent_cost <= lower_bound + 1e-9:
                self.stats["early_optimal"] += 1
                break
        record["chosen_cost"] = incumbent_cost
        self.stats["solve_time_s"] += record["solve_s"]
        self.stats["decode_time_s"] += record["decode_s"]
        self.stats["window_records"].append(record)
        return incumbent

    def _lower_bound(self, pg: Any, source: Any, goal) -> float:
        """Cheapest possible price for the mover alone to reach `goal`."""
        weight = lambda u, v, data: min(self.costs[data["label"]], self.costs["ion_swap"])
        # A swap can carry the mover across an in-trap edge for ion_swap.
        # The optimistic minimum remains a bound for arbitrary cost models.
        return nx.multi_source_dijkstra_path_length(pg.graph, goal, weight=weight)[source]

    def _window_nodes(self, path: List[Any], placement: Any, pg: Any) -> Set[Any]:
        nodes: Set[Any] = set(path) | set(goal_slots(pg, path[-1]))
        for pos in path:
            trap = trap_of(pg, pos)
            if trap is not None:
                nodes.update(pg.slots_of(trap))
        for pos in self._detect_blockages(path, placement):
            nodes.update(pg.graph.neighbors(pos))
        return nodes

    def extract_window(
        self,
        path: List[Any],
        blocked: List[Any],
        placement: Any,
        pg: Any,
    ) -> Optional[WindowInfo]:
        """Window W around the blocked path, or None if it is too large."""
        mover = placement.occupant_of(path[0])
        nodes = self._window_nodes(path, placement, pg)
        if mover is None or len(nodes) > self.window_max_nodes:
            return None
        # Iterate in graph order (not set order) so runs are reproducible.
        ions = {placement.occupant_of(v): v for v in pg.graph
                if v in nodes and placement.is_occupied(v)}
        return WindowInfo(
            window_nodes=frozenset(nodes),
            active_ions=ions,
            mover=mover,
            goal_nodes=frozenset(goal_slots(pg, path[-1])),
            source=path[0],
            target=path[-1],
            blocked_path=list(blocked),
        )

    # -----------------------------------------------------------------------
    # Severity metrics
    # -----------------------------------------------------------------------

    @staticmethod
    def _detect_blockages(path: List[Any], placement: Any) -> List[Any]:
        """Occupied intermediate positions (endpoints excluded)."""
        return [pos for pos in path[1:-1] if placement.is_occupied(pos)]

    @staticmethod
    def _compute_kappa(path: List[Any], placement: Any) -> float:
        if not path:
            return 0.0
        return sum(1 for pos in path if placement.is_occupied(pos)) / len(path)

    @staticmethod
    def _compute_rho(blocked: List[Any], C_LB: int) -> float:
        if C_LB <= 0:
            return 0.0
        return 2 * len(blocked) / C_LB

    def _compute_blockage_depth(
        self,
        blocked: List[Any],
        placement: Any,
        pg: Optional[Any],
        _depth: int = 0,
        _visited: Optional[Set[Any]] = None,
    ) -> int:
        """Levels of occupied neighbours behind each blocker (capped at d_th + 1)."""
        if pg is None or _depth >= self.depth_threshold + 1:
            return _depth
        visited = _visited if _visited is not None else set()
        deepest = _depth
        for pos in blocked:
            if pos in visited:
                continue
            visited.add(pos)
            secondary = [n for n in pg.graph.neighbors(pos)
                         if n not in visited and placement.is_occupied(n)]
            if secondary:
                deepest = max(deepest, self._compute_blockage_depth(
                    secondary, placement, pg, _depth + 1, visited.copy()
                ))
        return deepest

    def should_trigger_qubo(self, kappa: float, rho: float, depth: int) -> bool:
        return (
            (kappa > self.kappa_threshold and rho > self.rho_threshold)
            or depth > self.depth_threshold
        )

    # -----------------------------------------------------------------------
    # Greedy fallback
    # -----------------------------------------------------------------------

    def _find_unblocked_path(
        self,
        source: Any,
        target: Any,
        placement: Any,
        pg: Any,
    ) -> Optional[List[Any]]:
        """First of the k shortest simple paths with no occupied intermediates."""
        if self.max_reroute_candidates <= 0:
            return None
        # One BFS on the free-position graph replaces k-shortest-path
        # enumeration, whose repeated spur searches dominate congested grids.
        allowed = {v for v in pg.graph if v in (source, target) or not placement.is_occupied(v)}
        try:
            return nx.shortest_path(pg.graph.subgraph(allowed), source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def summary(self) -> str:
        s = self.stats
        return (
            f"calls={s['total_calls']} blocked={s['blocked_events']} "
            f"rerouted={s['rerouted']} qubo_triggers={s['qubo_triggers']} "
            f"qubo_solved={s['qubo_solved']} qubo_accepted={s['qubo_accepted']} "
            f"ops_saved={s['ops_saved']} qubo_time={s['qubo_time_s']:.2f}s"
        )


def goal_slots(pg: Any, target: Any) -> List[Any]:
    """Positions that count as arriving: every slot of the target's trap."""
    trap = trap_of(pg, target)
    return pg.slots_of(trap) if trap is not None else [target]


def layout_cost(pg: Any, board: Any, next_partner_pos: Dict[Any, Any],
                next_partner_ions: Optional[Dict[Any, Any]] = None) -> int:
    """Aggregate partner distance, resolving identities on the candidate board."""
    return sum(nx.shortest_path_length(pg.graph, board.position_of(ion),
                   board.position_of(next_partner_ions[ion]) if next_partner_ions and ion in next_partner_ions else pos)
               for ion, pos in next_partner_pos.items())


def walk_trajectories(
    pg: Any, placement: Any, path: List[Any], window: WindowInfo
) -> Dict[Any, List[Any]]:
    """
    The greedy walk along `path` as one action per time step, for every
    ion in the window. Used to warm-start annealing from a known-valid
    plan, so the QUBO search starts at the greedy cost and only has to
    improve on it.
    """
    pos = dict(window.active_ions)
    traj = {ion: [p] for ion, p in pos.items()}
    mover = window.mover
    scratch = BoardView(pg, placement)
    for action in plan_walk(pg, placement, path, mover):
        if action[0] == "vacate":
            _, ion, hop, free_slot = action
            operations = [("move", ion, free_slot), ("move", mover, hop)]
        elif action[0] == "displace":
            operations = [("move", action[1], action[2])]
        elif action[0] == "swap":
            operations = [("swap", mover, action[1])]
        else:
            operations = [("move", mover, action[1])]
        scratch.apply(operations)
        # A vacate and following entry share a model step. The decoder
        # serializes them in vacancy order; no artificial extra horizon.
        for ion in traj:
            traj[ion].append(scratch.position_of(ion))
    return traj


def seed_from(problem: QUBOProblem, traj: Dict[Any, List[Any]]) -> Dict[str, int]:
    """BQM assignment for the given trajectories (padded with stays)."""
    seed: Dict[str, int] = {}
    for ion, positions in traj.items():
        for t in range(problem.time_horizon + 1):
            label = problem.var_map.get((ion, positions[min(t, len(positions) - 1)], t))
            if label is not None:
                seed[label] = 1
    return problem.complete(seed)
