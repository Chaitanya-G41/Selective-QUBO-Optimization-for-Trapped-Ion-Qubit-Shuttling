"""
congestion_handler.py

Implements Phases 3 – 6 of the Selective-QUBO methodology for
trapped-ion qubit shuttling:

  Phase 3  Congestion Severity Analysis
    Compute three local metrics after SHAW detects a blockage:
      κ   (kappa)  – local contention: fraction of region occupied
      ρ   (rho)    – routing regret: extra cost vs. uncongested lower bound
      d            – recursive blockage depth: how many levels of secondary
                     blocking must be cleared before the path is free

  Phase 4  Selective QUBO Trigger
    QUBO is invoked only when congestion is genuinely severe:
      (κ > κ_th  AND  ρ > ρ_th)   OR   d > d_th
    Ordinary blockage → SHAW-based greedy clearing (this file handles it).
    Severe congestion → window extracted and QUBO slot activated
                         (QUBO solve deferred to next sprint; greedy
                          fallback still applied so routing never stalls).

  Phase 5  Local Congestion Window Extraction
    A bounded subgraph W is extracted around the problematic region by
    BFS up to `window_radius` hops from each blocked position.
    Ions are classified as:
      active_ions    – inside W, must be explicitly re-routed
      obstacle_ions  – outside W, treated as fixed boundary conditions

  Phase 6  QUBO slot (wired, solve deferred)
    When the trigger fires, `extract_window` returns a fully populated
    `WindowInfo` namedtuple ready for the QUBO formulator.  The current
    implementation falls back to SHAW-style greedy clearing so that
    routing continues uninterrupted.

Reference papers
----------------
[SHAW]   Bach, Safro, Younis – "Efficient Compilation for Shuttling
         Trapped-Ion Machines via the Position Graph Architectural
         Abstraction", arXiv 2501.12470, ACM TQC 2026.
         §3.3: recursive blockage clearing is the SHAW congestion baseline.

[QUBO]   Glover, Kochenberger, Du – "A Tutorial on Formulating and
         Using QUBO Models", arXiv 1811.11538, 2019.
         Penalty construction: λ = C_max + 1 ensures constraint
         satisfaction dominates the objective.

Interface
---------
The only method called by shaw_routing_pass.py is::

    CongestionHandler.resolve_path_congestion(path, placement, pg=None)
      path      : list[str]  – position IDs from PositionGraph.shortest_path
      placement : Placement  – live ion↔position mapping
      pg        : PositionGraph | None – hardware graph (enables window
                                         extraction and rerouting; pass None
                                         to get pure blockage logging only)
      returns   : list[str]  – a cleared (or best-effort) path

All other methods are public helpers available for unit tests and the
upcoming QUBO formulation sprint.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from position_graph import Placement, PositionGraph

logger = logging.getLogger("shaw_router.congestion")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
# NOTE: an optional Laya System-One advisory gate lived here (lazy router
# cache + fixed trigger question + fail-open ask helper). Removed before
# commit to keep this R&D out of the shared tree; full code saved at
# /tmp/laya_backup/ for one-step re-adding later.

@dataclass
class CongestionMetrics:
    """Severity metrics computed for a single congestion event."""
    kappa: float          # local contention κ ∈ [0, 1]
    rho: float            # routing regret ρ ≥ 0
    depth: int            # recursive blockage depth d ≥ 0
    blocked: List[Any]    # list of blocked intermediate position IDs
    C_LB: int             # unconstrained shortest-path cost (hops)
    C_est: int            # estimated clearing cost (hops)
    qubo_triggered: bool  # whether the QUBO trigger condition fired

    def __str__(self) -> str:
        trigger = "QUBO" if self.qubo_triggered else "heuristic"
        return (
            f"CongestionMetrics(κ={self.kappa:.3f}, ρ={self.rho:.3f}, "
            f"d={self.depth}, blocked={len(self.blocked)}, "
            f"C_LB={self.C_LB}, C_est={self.C_est}, action={trigger})"
        )


@dataclass
class WindowInfo:
    """
    The bounded local window W extracted around the congested region.

    This is the primary input for the QUBO formulation (Phase 6).

    Attributes
    ----------
    window_nodes   : set of position IDs inside W
    active_ions    : {qudit_id: current_position} for ions inside W
                     (these are the decision variables in the QUBO)
    obstacle_ions  : {qudit_id: current_position} for ions outside W
                     (treated as fixed boundary conditions in the QUBO)
    source         : start position of the original blocked path
    target         : destination position of the original blocked path
    blocked_path   : the intermediate positions that were occupied
    center_nodes   : positions used as BFS seeds for window expansion
    radius         : BFS radius used
    size_capped    : True if |W| was clamped to window_max_size
    """
    window_nodes: FrozenSet[Any]
    active_ions: Dict[Any, Any]
    obstacle_ions: Dict[Any, Any]
    source: Any
    target: Any
    blocked_path: List[Any]
    center_nodes: List[Any]
    radius: int
    size_capped: bool


@dataclass
class CongestionEvent:
    """Record of one congestion resolution event (for statistics)."""
    path_length: int
    metrics: CongestionMetrics
    window: Optional[WindowInfo]
    outcome: str            # "clear", "rerouted", "greedy_cleared",
                            # "qubo_accepted", "unresolved"
    alt_path_length: int    # length of returned path (same as path_length if no reroute)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CongestionHandler:
    """
    Congestion detection, severity analysis, window extraction, and
    SHAW-style greedy clearing for the selective-QUBO routing framework.

    Parameters
    ----------
    kappa_threshold  : float  κ_th – contention threshold (default 0.5)
    rho_threshold    : float  ρ_th – regret threshold (default 0.3)
    depth_threshold  : int    d_th – recursive blockage depth threshold (2)
    window_radius    : int    BFS radius for window extraction (default 1)
    window_max_size  : int    |W| upper bound (default 20)
    max_reroute_candidates : int  how many alternative paths to try (default 8)
    required_margin  : int    accept a decoded path only if
        C_heuristic - C_QUBO >= required_margin (default 2, measured:
        margin-1 accepts routinely tie or lose after walk-execution
        turbulence eats their ~1-hop on-paper saving, while margin-2
        keeps the decisive wins and drops the poisoned ones).
        Higher margins filter marginal accepts whose walk-execution
        turbulence (vacates, re-routing from new positions) eats the
        on-paper saving.
    (No Laya parameters: the advisory gate was removed before commit;
    full code saved at /tmp/laya_backup/ for one-step re-adding later.)
    """

    def __init__(
        self,
        kappa_threshold: float = 0.5,
        rho_threshold: float = 0.3,
        depth_threshold: int = 2,
        window_radius: int = 1,
        window_max_size: int = 20,
        max_reroute_candidates: int = 8,
        required_margin: int = 2,
    ) -> None:
        self.kappa_threshold = kappa_threshold
        self.rho_threshold = rho_threshold
        self.depth_threshold = depth_threshold
        self.window_radius = window_radius
        self.window_max_size = window_max_size
        self.max_reroute_candidates = max_reroute_candidates
        self.required_margin = required_margin
        # SA draw counter: each triggered solve gets a distinct but
        # deterministic seed, so routes are reproducible run to run
        # (same circuit + same handler settings = same routing).
        # Without this, verdict benchmarking is noise-chasing: SA luck
        # swings individual circuits ±10% between identical runs.
        self._solve_counter = 0

        # Running statistics – collected for experimental evaluation (§11)
        self.stats: Dict[str, Any] = {
            "total_calls": 0,
            "clear_paths": 0,        # no blockage at all
            "blocked_events": 0,     # at least one intermediate blocked
            "qubo_triggers": 0,      # severe enough to escalate
            "qubo_accepted": 0,      # decoded path returned for walking
            "rerouted": 0,           # found a fully clear alternate path
            "greedy_cleared": 0,     # greedy blocker shuffle succeeded
            "unresolved": 0,         # returned original path unchanged
            "kappa_values": [],      # for histogram / calibration
            "rho_values": [],
            "depth_values": [],
            "accept_margins": [],    # C_heur - C_QUBO per accepted path
            "events": [],            # list[CongestionEvent]
        }

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def resolve_path_congestion(
        self,
        path: List[Any],
        placement: "Placement",
        pg: Optional["PositionGraph"] = None,
    ) -> List[Any]:
        """
        Main entry point called by ShawRoutingPass.

        Algorithm
        ---------
        1. Detect B(P) – blocked intermediate positions.
        2. If |B(P)| = 0, return path unchanged (fast path).
        3. Compute severity metrics κ, ρ, d.
        4. Check QUBO trigger condition.
        5. If QUBO triggered → extract window W, log event.
           (QUBO solve deferred; falls through to greedy clearing.)
        6. Attempt greedy clearing:
           a. Try to find a fully unblocked alternate path (reroute).
           b. If no alternate path, shuffle each blocker to nearest free
              position (SHAW-style recursive clearing).
        7. Return the best available path.

        Parameters
        ----------
        path      : list of position IDs (output of PositionGraph.shortest_path)
        placement : live ion↔position state
        pg        : position graph (enables rerouting + window extraction)

        Returns
        -------
        list of position IDs (cleared or best-effort)
        """
        self.stats["total_calls"] += 1

        # ---- Step 1: detect blockages ------------------------------------
        blocked = self._detect_blockages(path, placement)

        if not blocked:
            self.stats["clear_paths"] += 1
            logger.debug("Congestion check: path clear (%d hops)", len(path) - 1)
            return path

        # ---- Step 2: severity analysis -----------------------------------
        self.stats["blocked_events"] += 1
        C_LB = len(path) - 1

        kappa = self._compute_kappa(path, placement)
        rho = self._compute_rho(blocked, C_LB)
        depth = self._compute_blockage_depth(blocked, placement, pg)

        self.stats["kappa_values"].append(kappa)
        self.stats["rho_values"].append(rho)
        self.stats["depth_values"].append(depth)

        triggered = self.should_trigger_qubo(kappa, rho, depth)

        C_est = C_LB + len(blocked) * 2 + depth * 1
        metrics = CongestionMetrics(
            kappa=kappa,
            rho=rho,
            depth=depth,
            blocked=blocked,
            C_LB=C_LB,
            C_est=C_est,
            qubo_triggered=triggered,
        )

        logger.info(
            "Congestion detected: path len=%d, blocked=%d, %s",
            len(path) - 1, len(blocked), metrics,
        )

        # ---- Step 3: window extraction (always done when pg available) ---
        window: Optional[WindowInfo] = None
        if pg is not None:
            window = self.extract_window(path, blocked, placement, pg)
            logger.info(
                "Window W: |W|=%d positions, %d active ions, %d obstacles%s",
                len(window.window_nodes),
                len(window.active_ions),
                len(window.obstacle_ions),
                " [size-capped]" if window.size_capped else "",
            )

        # ---- Step 4: QUBO trigger ----------------------------------------
        if triggered:
            self.stats["qubo_triggers"] += 1
            logger.info(
                "QUBO TRIGGER fired (κ=%.3f > %.3f, ρ=%.3f > %.3f, d=%d > %d). "
                "Window ready; attempting selective QUBO solve.",
                kappa, self.kappa_threshold,
                rho, self.rho_threshold,
                depth, self.depth_threshold,
            )
        # ---- Step 4.5: selective QUBO solve ------------------------------
        # On a trigger, formulate the window, solve it, and decode/
        # validate the result. An accepted mover path is returned for
        # ShawRoutingPass to walk (mover-only execution; blockers are
        # handled live by the walk) -- anything else falls through to
        # greedy clearing below. (Imports are function-level: the qubo
        # modules import WindowInfo from THIS module, so top-level
        # imports would be circular.)
        resolved_path = path  # default: return original, let _walk_path handle it
        outcome = "unresolved"

        if triggered and window is not None and pg is not None:
            qubo_path = self._try_qubo_solve(window, placement, pg, path, blocked)
            if qubo_path is not None:
                resolved_path = qubo_path
                outcome = "qubo_accepted"
                self.stats["qubo_accepted"] += 1
                logger.info(
                    "QUBO accepted: returning decoded %d-hop path.",
                    len(qubo_path) - 1,
                )

        # ---- Step 5: greedy clearing -------------------------------------
        if outcome == "unresolved" and pg is not None:
            # 5a. Try rerouting – a fully clear alternate path
            alt = self._find_unblocked_path(path[0], path[-1], placement, pg)
            if alt is not None and alt != path:
                resolved_path = alt
                outcome = "rerouted"
                self.stats["rerouted"] += 1
                logger.info(
                    "Rerouted: original %d hops → alternate %d hops",
                    len(path) - 1, len(alt) - 1,
                )
            else:
                # 5b. SHAW greedy shuffle: move each blocker to a free adjacent slot
                shuffled = self._shaw_greedy_clear(path, blocked, placement, pg)
                if shuffled is not None:
                    resolved_path = shuffled
                    outcome = "greedy_cleared"
                    self.stats["greedy_cleared"] += 1
                    logger.info(
                        "Greedy cleared: returned path of length %d",
                        len(shuffled) - 1,
                    )
                else:
                    self.stats["unresolved"] += 1
                    logger.info(
                        "Could not clear path; returning original "
                        "(shaw_routing_pass will handle swap/deadlock)."
                    )
        elif pg is None:
            # No pg – can only log (a "qubo_accepted" outcome skips
            # both branches above and lands here, so re-check pg).
            self.stats["unresolved"] += 1
            logger.info("No position graph available; path returned unchanged.")

        # ---- Record event ------------------------------------------------
        event = CongestionEvent(
            path_length=len(path) - 1,
            metrics=metrics,
            window=window,
            outcome=outcome,
            alt_path_length=len(resolved_path) - 1,
        )
        self.stats["events"].append(event)

        return resolved_path

    # -----------------------------------------------------------------------
    # Phase 4.5 – selective QUBO solve (formulate → solve → decode → apply)
    # -----------------------------------------------------------------------

    def _greedy_seed_sample(
        self,
        problem: "QUBOProblem",
        pg: "PositionGraph",
    ) -> Optional[Dict[str, int]]:
        """
        Warm-start seed for SA: the mover walks the static shortest
        source→target path (then stays); anyone the walk would step on
        is shoved aside the same step into a free neighbour -- except
        into the slot the mover just vacated, which would be a swap
        (crossing) and is skipped. Unshovable collisions stay
        overlapping. One-hot valid by construction modulo walk steps
        outside the modelled window (keys without variables). May
        violate capacity/crossing where shoves fail -- fine for a
        starting point; SA repairs from a good basin instead of
        searching blindly. Returns None if no static path exists
        (caller solves cold). Covers every BQM variable (primaries +
        Rosenberg aux): neal rejects initial_states that don't match
        bqm.variables exactly. Aux default 0; SA repairs.

        NOTE (measured): shoving into the mover's vacated slot was
        tried first and reverted -- that IS a crossing move, so the
        seed tripped collision_avoidance and started worse than the
        plain walk-through. Vacancy must never come from the mover's
        own trail in the same step.
        """
        window = problem.window
        mover = next(
            (ion for ion, pos in window.active_ions.items()
             if pos == window.source),
            None,
        )
        if mover is None:
            return None
        G = getattr(pg, "graph", None)
        try:
            walk = (nx.shortest_path(G, window.source, window.target)
                    if G is not None else [window.source, window.target])
        except Exception:
            return None

        T = problem.time_horizon
        # traj[ion][t]: forward-built per-ion trajectories. The mover
        # owns its walk; anyone the walk would step on is shoved aside
        # THIS step into a free neighbour that is (a) not the slot the
        # mover just vacated (that would be a swap = crossing, which
        # the decoder rightly rejects) and (b) not anywhere on the
        # mover's future walk (parking there just schedules the next
        # collision). Unshovable collisions stay overlapping, exactly
        # like the old walk-through seed. One append per ion per step
        # keeps every trajectory exactly T+1 long by construction, so
        # one-hot validity can only break where the walk itself leaves
        # the modelled window (same as before -- those keys simply
        # have no variables).
        mover_traj = [walk[min(t, len(walk) - 1)] for t in range(T + 1)]
        walk_set = set(walk)
        traj: Dict[Any, List[Any]] = {
            ion: [start_pos] for ion, start_pos in window.active_ions.items()
        }
        for t in range(1, T + 1):
            mpos, mprev = mover_traj[t], mover_traj[t - 1]
            traj[mover].append(mpos)
            taken = {mpos}
            for ion in window.active_ions:
                if ion == mover:
                    continue
                prev = traj[ion][-1]
                if prev != mpos:
                    traj[ion].append(prev)
                    taken.add(prev)
                    continue
                shoved = None
                if G is not None and prev in G:
                    for nbr in G.neighbors(prev):
                        if nbr == mprev or nbr in walk_set:
                            continue  # swap-back or future collision
                        if nbr in taken or nbr not in window.window_nodes:
                            continue
                        if not (G.has_edge(prev, nbr) or G.has_edge(nbr, prev)):
                            continue
                        shoved = nbr
                        break
                traj[ion].append(shoved if shoved is not None else prev)
                taken.add(traj[ion][-1])

        seed: Dict[str, int] = {label: 0 for label in problem.bqm.variables}
        for ion, positions in traj.items():
            for t, pos in enumerate(positions):
                key = (ion, pos, t)
                if key in problem.var_map:
                    seed[problem.var_map[key]] = 1
        return seed

    def _greedy_outcome(
        self,
        path: List[Any],
        blocked: List[Any],
        placement: "Placement",
        pg: "PositionGraph",
    ) -> Tuple[List[Any], str]:
        """
        What step 5 (greedy clearing) would return for this path,
        WITHOUT recording stats or logging: (resolved_path, outcome)
        where outcome is one of "rerouted" / "greedy_cleared" /
        "unresolved". Pure except for reads -- safe to call
        speculatively. Step 5 keeps its own inline copy so its
        stats/logging behavior is untouched.
        """
        alt = self._find_unblocked_path(path[0], path[-1], placement, pg)
        if alt is not None and alt != path:
            return alt, "rerouted"
        shuffled = self._shaw_greedy_clear(path, blocked, placement, pg)
        if shuffled is not None:
            return shuffled, "greedy_cleared"
        return path, "unresolved"

    def _try_qubo_solve(
        self,
        window: WindowInfo,
        placement: "Placement",
        pg: "PositionGraph",
        path: List[Any],
        blocked: List[Any],
    ) -> Optional[List[Any]]:
        """
        Best-effort QUBO resolution for one triggered window. Formulate
        the window, solve it (auto exact/SA), decode + validate the
        sample, pre-clear blockers per the accepted trajectories, and
        return the accepted mover path for ShawRoutingPass to walk with
        its normal machinery (plus movement markers).

        Execution is split this way on purpose: replaying every ion's
        choreography AND then walking the mover moves it twice (once
        to the target via apply, once backward through the path via
        the walk -- measured +19% shuttle regressions), while walking
        alone re-vacates blockers the plan had already parked (2
        markers per contested hop). Pre-clearing blockers and walking
        only the mover costs each move once. The joint trajectories
        are still fully decoded + validated (that is what makes the
        path trustworthy). Anything failing -- apply included -- falls
        back to greedy clearing, so a QUBO miss never breaks routing,
        it just costs the solve time.
        """
        # Function-level imports: qubo_solver → qubo_formulator →
        # congestion_handler (this module) at top level, so importing
        # them up top would be circular.
        from qubo_formulator import QUBOFormulator
        from qubo_solver import QUBOSolver
        from solution_decoder import SolutionDecoder

        try:
            problem = QUBOFormulator().build(window, pg)
            seed = self._greedy_seed_sample(problem, pg)
            # Two-tier SA budget (measured): windows with at most 150
            # PRIMARY variables keep the full 200-read budget -- every
            # observed accept lives in this regime, and primaries (not
            # aux-inflated totals) measure the problem's true size, so
            # small-window behavior is byte-identical across budget
            # experiments. Above that, reads scale as 20000/total
            # floored at 10: monster windows get a bounded-effort
            # attempt (they have never produced an accept, only minutes
            # of anneal time) and fall back to greedy on a miss.
            # Pavan's solver defaults are untouched -- this override
            # lives only here.
            total_vars = problem.num_variables + problem.num_aux_variables
            if problem.num_variables <= 150:
                reads = 200
            else:
                reads = max(10, 20000 // max(total_vars, 1))
            self._solve_counter += 1
            solution = QUBOSolver(
                sa_num_reads=reads, sa_seed=1000 + self._solve_counter
            ).solve(problem, initial_states=seed)
            decoded = SolutionDecoder().decode_and_validate(solution, placement, pg)
        except Exception as exc:
            logger.warning(
                "QUBO formulate/solve/decode failed (%s); falling back to greedy.",
                exc,
            )
            return None

        if not decoded.accepted or not decoded.decoded_path:
            return None

        # Margin gate: the decoder accepts any strict improvement, but
        # walk-execution turbulence (vacates, re-routing from the new
        # board) routinely eats ~1 hop of on-paper savings. Demanding
        # margin >= 2 filters accepts that would tie or lose in
        # practice (measured across chain/long-range/random circuits).
        margin = decoded.C_heuristic - decoded.C_QUBO
        if margin < self.required_margin:
            logger.info(
                "QUBO margin too thin (%d < %d); falling back to greedy.",
                margin, self.required_margin,
            )
            return None
        self.stats["accept_margins"].append(margin)

        # Execution-cost gate: simulate walking the decoded path vs what
        # greedy clearing would walk, on the CURRENT board (read-only
        # dry runs via ShawRoutingPass's own planner, so estimates
        # match what the walk would really emit). Accept only if the
        # QUBO walk is strictly cheaper to execute. Rationale
        # (measured): on small boards the C_heur estimate goes so loose
        # (margins 6-8 on 008-class circuits) that fat-margin accepts
        # still lose double digits in walk-vacate execution -- hop-count
        # margin alone cannot see that. (Imports are function-level:
        # shaw_routing_pass imports this module.)
        from shaw_routing_pass import ShawRoutingPass
        mover = placement.occupant_of(decoded.decoded_path[0])
        if mover is not None:
            greedy_path, _ = self._greedy_outcome(path, blocked, placement, pg)
            qubo_cost = ShawRoutingPass._estimate_walk_markers(
                pg, placement, decoded.decoded_path, mover
            )
            greedy_cost = ShawRoutingPass._estimate_walk_markers(
                pg, placement, greedy_path, mover
            )
            if qubo_cost >= greedy_cost:
                logger.info(
                    "QUBO walk cost %d >= greedy walk cost %d; "
                    "falling back to greedy.", qubo_cost, greedy_cost,
                )
                return None

        # NOTE (measured): a blockers-only pre-clear variant was tried
        # here and reverted -- displacing blockers speculatively
        # perturbs later gates more than the walk-vacates it saves
        # (020's only win vanished, losses steepened). Mover-only
        # execution it is: the walk handles blockers live.
        return decoded.decoded_path

    # -----------------------------------------------------------------------
    # Phase 3 – Severity metrics
    # -----------------------------------------------------------------------

    def _detect_blockages(
        self,
        path: List[Any],
        placement: "Placement",
    ) -> List[Any]:
        """
        B(P) = {v ∈ P | occ(v) = 1}, excluding source and destination.

        Endpoints are intentionally excluded: the source is where the
        moving ion currently sits (of course occupied by it) and the
        destination may be occupied by the ion we are routing toward
        (and SHAW's _walk_path already handles that case via shuffling).
        """
        if len(path) <= 2:
            return []
        return [pos for pos in path[1:-1] if placement.is_occupied(pos)]

    def _compute_kappa(
        self,
        path: List[Any],
        placement: "Placement",
    ) -> float:
        """
        κ = (occupied positions in local region) / (total positions in local region)

        The "local region" is the full path P (including endpoints), giving
        a tight, path-specific density estimate rather than a global one.

        Returns a value in [0, 1].
        """
        if not path:
            return 0.0
        occupied = sum(1 for pos in path if placement.is_occupied(pos))
        return occupied / len(path)

    def _compute_rho(
        self,
        blocked: List[Any],
        C_LB: int,
        eps: float = 1e-6,
    ) -> float:
        """
        ρ = (C_est − C_LB) / max(C_LB, ε)

        C_LB  = unconstrained shortest-path hop count (len(path) − 1).
        C_est = estimated actual clearing cost.

        Each blocked intermediate position requires the blocker to be
        moved aside (+1 hop) and potentially back (+1 hop), so we model
        C_est = C_LB + 2 × |B(P)|.

        A regret of 0 means no extra cost vs. the ideal; a regret of 1
        means the clearing doubles the path cost.
        """
        if C_LB == 0:
            return 0.0
        C_est = C_LB + 2 * len(blocked)
        return (C_est - C_LB) / max(C_LB, eps)

    def _compute_blockage_depth(
        self,
        blocked: List[Any],
        placement: "Placement",
        pg: Optional["PositionGraph"],
        _current_depth: int = 0,
        _visited: Optional[Set[Any]] = None,
    ) -> int:
        """
        d = recursive blockage depth.

        Counts how many levels of secondary blocking must be resolved:
          d = 0 → all blockers can be moved to an immediately free slot.
          d = 1 → moving a blocker reveals a secondary blocker, etc.

        Uses BFS over the position graph's adjacency to detect whether
        the immediate neighbours of each blocked position are themselves
        occupied (forcing another round of clearing).

        Capped at `depth_threshold + 1` to avoid expensive traversals.
        """
        MAX_SCAN = self.depth_threshold + 1  # cap to avoid deep recursion

        if _visited is None:
            _visited = set()

        if _current_depth >= MAX_SCAN or pg is None:
            return _current_depth

        G = getattr(pg, "graph", None)
        if G is None:
            return _current_depth

        max_depth = _current_depth
        for blocker_pos in blocked:
            if blocker_pos in _visited:
                continue
            _visited.add(blocker_pos)

            # Look at immediate neighbours of this blocked position
            secondary: List[Any] = []
            for nbr in G.neighbors(blocker_pos):
                if nbr in _visited:
                    continue
                if placement.is_occupied(nbr):
                    secondary.append(nbr)

            if secondary:
                sub = self._compute_blockage_depth(
                    secondary,
                    placement,
                    pg,
                    _current_depth + 1,
                    _visited | set(secondary),
                )
                max_depth = max(max_depth, sub)

        return max_depth

    # -----------------------------------------------------------------------
    # Phase 4 – QUBO trigger condition
    # -----------------------------------------------------------------------

    def should_trigger_qubo(
        self,
        kappa: float,
        rho: float,
        depth: int,
    ) -> bool:
        """
        QUBO trigger:  (κ > κ_th  AND  ρ > ρ_th)  OR  d > d_th

        Both conditions in the conjunction must hold to avoid triggering
        on mild, easily-cleared congestion.  The depth safety trigger
        catches pathological recursive cases even if κ and ρ individually
        look modest.

        Thresholds are intentionally exposed as constructor parameters so
        they can be calibrated experimentally (methodology §11) rather
        than fixed here.
        """
        severity_trigger = (kappa > self.kappa_threshold) and (rho > self.rho_threshold)
        depth_trigger = depth > self.depth_threshold
        return severity_trigger or depth_trigger

    # -----------------------------------------------------------------------
    # Phase 5 – Local window extraction
    # -----------------------------------------------------------------------

    def extract_window(
        self,
        path: List[Any],
        blocked: List[Any],
        placement: "Placement",
        pg: "PositionGraph",
        radius: Optional[int] = None,
    ) -> WindowInfo:
        """
        Extract a bounded local window W around the blocked region.

        W contains:
          • every node on the original path,
          • every blocked intermediate position,
          • BFS expansion up to `radius` hops from each blocked position,
          • neighbouring positions of each blocking ion (candidate clear spots).

        Size is capped at `window_max_size` to keep the QUBO formulation
        manageable.  When capping, we preserve the path nodes and the
        immediate neighbours of blocked positions (most relevant for
        routing decisions) and trim outlying BFS nodes.

        Ions inside W are `active_ions` (decision variables for QUBO).
        Ions outside W are `obstacle_ions` (boundary conditions for QUBO).

        Parameters
        ----------
        path     : full original path (source → target)
        blocked  : subset of path – the occupied intermediate positions
        placement: live ion placement state
        pg       : position graph (must not be None when calling this)
        radius   : BFS radius override (defaults to self.window_radius)

        Returns
        -------
        WindowInfo namedtuple
        """
        r = radius if radius is not None else self.window_radius
        G = getattr(pg, "graph", None)

        if G is None:
            # Graceful fallback: window = just the path nodes
            window_nodes: Set[Any] = set(path)
            active: Dict[Any, Any] = {}
            obstacle: Dict[Any, Any] = {}
            for pos, occupant in self._all_occupied(G, placement):
                (active if pos in window_nodes else obstacle)[occupant] = pos
            return WindowInfo(
                window_nodes=frozenset(window_nodes),
                active_ions=active,
                obstacle_ions=obstacle,
                source=path[0],
                target=path[-1],
                blocked_path=list(blocked),
                center_nodes=list(blocked),
                radius=r,
                size_capped=False,
            )

        # ---- BFS expansion from blocked positions ------------------------
        center_nodes = list(blocked) if blocked else [path[0], path[-1]]
        bfs_nodes: Set[Any] = set()
        for cn in center_nodes:
            if cn not in G:
                continue
            reachable = nx.single_source_shortest_path_length(
                G.to_undirected() if G.is_directed() else G,
                cn,
                cutoff=r,
            )
            bfs_nodes.update(reachable.keys())

        # Always include the full path (source + all intermediates + target)
        core_nodes: Set[Any] = set(path) | set(blocked)
        # Also include immediate neighbours of blocked positions as candidate
        # "parking spots" for blocking ions
        parking_spots: Set[Any] = set()
        for bp in blocked:
            if bp in G:
                parking_spots.update(G.neighbors(bp))

        window_nodes = core_nodes | parking_spots | bfs_nodes
        size_capped = False

        if len(window_nodes) > self.window_max_size:
            # Trim: prioritize core path nodes, then parking spots, then outer BFS nodes
            size_capped = True
            trimmed: Set[Any] = set()
            priority_order = list(core_nodes) + [n for n in parking_spots if n not in core_nodes] + [n for n in bfs_nodes if n not in core_nodes and n not in parking_spots]
            for node in priority_order:
                if len(trimmed) >= self.window_max_size:
                    break
                trimmed.add(node)
            window_nodes = trimmed

        # ---- Classify ions -----------------------------------------------
        active_ions: Dict[Any, Any] = {}
        obstacle_ions: Dict[Any, Any] = {}

        all_nodes = list(G.nodes()) if G is not None else list(window_nodes)
        for pos in all_nodes:
            occupant = placement.occupant_of(pos)
            if occupant is None:
                continue
            if pos in window_nodes:
                active_ions[occupant] = pos
            else:
                obstacle_ions[occupant] = pos

        return WindowInfo(
            window_nodes=frozenset(window_nodes),
            active_ions=active_ions,
            obstacle_ions=obstacle_ions,
            source=path[0],
            target=path[-1],
            blocked_path=list(blocked),
            center_nodes=center_nodes,
            radius=r,
            size_capped=size_capped,
        )

    # -----------------------------------------------------------------------
    # Greedy clearing helpers (SHAW-style fallback)
    # -----------------------------------------------------------------------

    def _find_unblocked_path(
        self,
        source: Any,
        target: Any,
        placement: "Placement",
        pg: "PositionGraph",
    ) -> Optional[List[Any]]:
        """
        Try to find a shortest path from source to target that has no
        occupied intermediate nodes.

        Iterates over the `max_reroute_candidates` shortest simple paths
        in the position graph and returns the first fully clear one.
        Returns None if no such path exists within the search budget.

        This corresponds to SHAW's "find an alternate shuttling path"
        step before resorting to recursive blocker clearing.
        """
        G = getattr(pg, "graph", None)
        if G is None:
            return None
        if source not in G or target not in G:
            return None
        if source == target:
            return [source]

        try:
            undirected = G.to_undirected() if G.is_directed() else G
            for candidate in itertools.islice(
                nx.shortest_simple_paths(undirected, source, target),
                self.max_reroute_candidates,
            ):
                intermediates = candidate[1:-1]
                if not any(placement.is_occupied(p) for p in intermediates):
                    return candidate
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
            pass

        return None

    def _shaw_greedy_clear(
        self,
        path: List[Any],
        blocked: List[Any],
        placement: "Placement",
        pg: "PositionGraph",
    ) -> Optional[List[Any]]:
        """
        SHAW-style greedy recursive clearing.

        For each blocked intermediate position, try to find a free
        adjacent slot where the blocking ion can be temporarily parked.
        Returns the original path if all blockers can be parked (the
        caller – _walk_path – will then see free slots), or None if
        at least one blocker cannot be cleared.

        This mirrors the recursive "resolve_blockage" procedure in the
        SHAW paper (§3.3): find a free neighbour, move the blocking ion
        there, then retry the original path.

        IMPORTANT: this method only *identifies* the parking positions
        and logs the plan; it does NOT mutate `placement` or
        `routed_circuit`.  Actual movement is handled by
        ShawRoutingPass._walk_path.  The return value being the original
        `path` (with blockers now having known parking spots) signals to
        the caller that the path is reroutable.
        """
        G = getattr(pg, "graph", None)
        if G is None:
            return None

        parking_plan: Dict[Any, Any] = {}   # blocker_pos → free_neighbor

        for blocker_pos in blocked:
            occupant = placement.occupant_of(blocker_pos)
            if occupant is None:
                continue  # already vacated by a previous step

            # 1. Prefer a free slot in the same trap (in-trap shuffle)
            trap_id = G.nodes[blocker_pos].get("trap_id") if blocker_pos in G else None
            parked = False

            if trap_id is not None and hasattr(pg, "slots_of"):
                for slot in pg.slots_of(trap_id):
                    if slot == blocker_pos:
                        continue
                    if slot not in parking_plan.values() and not placement.is_occupied(slot):
                        parking_plan[blocker_pos] = slot
                        parked = True
                        logger.debug(
                            "Greedy clear: ion %s at %s → park in same trap at %s",
                            occupant, blocker_pos, slot,
                        )
                        break

            if parked:
                continue

            # 2. Try immediate neighbours outside the path
            path_set = set(path)
            for nbr in G.neighbors(blocker_pos):
                if nbr in path_set:
                    continue  # don't park onto the path we're trying to clear
                if nbr in parking_plan.values():
                    continue  # already reserved for another blocker
                if not placement.is_occupied(nbr):
                    parking_plan[blocker_pos] = nbr
                    parked = True
                    logger.debug(
                        "Greedy clear: ion %s at %s → park at neighbour %s",
                        occupant, blocker_pos, nbr,
                    )
                    break

            if not parked:
                logger.debug(
                    "Greedy clear: no parking spot found for ion %s at %s",
                    occupant, blocker_pos,
                )
                return None  # cannot clear this blocker; give up

        if parking_plan:
            logger.info(
                "Greedy clear plan: %d blocker(s) parked – %s",
                len(parking_plan),
                {str(k): str(v) for k, v in parking_plan.items()},
            )

        # All blockers have parking spots → the original path will be clear
        # once _walk_path executes the shuffle moves.
        return path

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _all_occupied(
        self,
        G: Optional[Any],
        placement: "Placement",
    ):
        """Yield (position, occupant_id) for every occupied node."""
        nodes = list(G.nodes()) if G is not None else []
        for pos in nodes:
            occ = placement.occupant_of(pos)
            if occ is not None:
                yield pos, occ

    def summary(self) -> str:
        """Return a human-readable summary of accumulated statistics."""
        s = self.stats
        n = s["total_calls"]
        if n == 0:
            return "CongestionHandler: no calls recorded yet."

        lines = [
            "=== CongestionHandler Statistics ===",
            f"  Total calls          : {n}",
            f"  Clear paths          : {s['clear_paths']} ({100*s['clear_paths']/n:.1f}%)",
            f"  Blocked events       : {s['blocked_events']} ({100*s['blocked_events']/n:.1f}%)",
            f"  QUBO triggers        : {s['qubo_triggers']}",
            f"  Rerouted (alt path)  : {s['rerouted']}",
            f"  Greedy cleared       : {s['greedy_cleared']}",
            f"  Unresolved           : {s['unresolved']}",
        ]
        if s["kappa_values"]:
            import statistics
            lines += [
                f"  κ  mean/max          : {statistics.mean(s['kappa_values']):.3f} / {max(s['kappa_values']):.3f}",
                f"  ρ  mean/max          : {statistics.mean(s['rho_values']):.3f} / {max(s['rho_values']):.3f}",
                f"  d  mean/max          : {statistics.mean(s['depth_values']):.2f} / {max(s['depth_values'])}",
            ]
        return "\n".join(lines)