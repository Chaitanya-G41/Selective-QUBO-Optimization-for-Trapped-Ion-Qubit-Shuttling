"""
solution_decoder.py
===================
Turns QUBO samples into an executable move plan.

Every candidate sample is decoded into per-ion trajectories and checked
against the physical rules directly (never inferred from the energy):

  one_hot              exactly one position per ion per time step
  initial_position     every ion starts where it really is
  movement_legality    each step is a stay or a hop along a graph edge
  capacity             at most one ion per position per time step
  collision_avoidance  no two ions exchange over a transport edge
                       (exchanges over in-trap "swap" edges are physical
                       ion swaps and are allowed)
  gate_feasibility     the mover ends in the goal trap
  scheduling           each step's moves can be ordered without a
                       rotation cycle

A valid sample becomes a MovePlan: an ordered list of operations
  ("move", ion, dest)   one hop of one ion          (cost 1)
  ("swap", ion_a, ion_b) in-trap exchange of two ions (cost 1)
using the same cost units as the greedy walk (walk_planner.py), so the
two can be compared fairly. The cheapest valid sample wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from qubo_formulator import QUBOProblem
from qubo_solver import QUBOSolution


@dataclass
class MovePlan:
    ops: List[tuple]
    trajectories: Dict[Any, List[Any]] = field(default_factory=dict)

    @property
    def cost(self) -> int:
        return len(self.ops)

    @property
    def displaced(self) -> int:
        """Ions that end somewhere other than where they started."""
        return sum(path[0] != path[-1] for path in self.trajectories.values())


class InvalidSample(Exception):
    def __init__(self, constraint: str, detail: str = "") -> None:
        super().__init__(f"{constraint}: {detail}")
        self.constraint = constraint


class SolutionDecoder:

    def valid_plans(self, solution: QUBOSolution) -> List[MovePlan]:
        """Every candidate sample that decodes to a valid plan."""
        plans = []
        for sample in solution.samples:
            try:
                plans.append(self.to_plan(sample, solution.problem))
            except InvalidSample:
                pass
        return plans

    def best_plan(self, solution: QUBOSolution) -> Tuple[Optional[MovePlan], List[str]]:
        """
        Cheapest valid plan (ties: fewest displaced ions) among the
        solver's candidates, plus the violated constraint of every
        rejected sample.
        """
        best: Optional[MovePlan] = None
        violations: List[str] = []
        for sample in solution.samples:
            try:
                plan = self.to_plan(sample, solution.problem)
            except InvalidSample as exc:
                violations.append(exc.constraint)
                continue
            if best is None or (plan.cost, plan.displaced) < (best.cost, best.displaced):
                best = plan
        return best, violations

    def to_plan(self, sample: Dict[str, int], problem: QUBOProblem) -> MovePlan:
        traj = self.decode(sample, problem)
        self.validate(traj, problem)
        return MovePlan(ops=self.schedule(traj, problem), trajectories=traj)

    # -----------------------------------------------------------------------

    @staticmethod
    def decode(sample: Dict[str, int], problem: QUBOProblem) -> Dict[Any, List[Any]]:
        """trajectory[ion][t] = position; requires exactly one bit per (ion, t)."""
        at: Dict[Tuple[Any, int], List[Any]] = {}
        for label, bit in sample.items():
            if bit and label in problem.inv_var_map:
                ion, pos, t = problem.inv_var_map[label]
                at.setdefault((ion, t), []).append(pos)

        traj: Dict[Any, List[Any]] = {}
        for ion in problem.window.active_ions:
            traj[ion] = []
            for t in range(problem.time_horizon + 1):
                positions = at.get((ion, t), [])
                if len(positions) != 1:
                    raise InvalidSample(
                        "one_hot", f"ion {ion} has {len(positions)} positions at t={t}"
                    )
                traj[ion].append(positions[0])
        return traj

    @staticmethod
    def validate(traj: Dict[Any, List[Any]], problem: QUBOProblem) -> None:
        G, window, T = problem.graph, problem.window, problem.time_horizon

        for ion, start in window.active_ions.items():
            if traj[ion][0] != start:
                raise InvalidSample("initial_position", f"ion {ion}")
            for t in range(T):
                u, v = traj[ion][t], traj[ion][t + 1]
                if u != v and not G.has_edge(u, v):
                    raise InvalidSample("movement_legality", f"ion {ion}: {u}->{v} at t={t}")

        for t in range(T + 1):
            occupied = [traj[ion][t] for ion in traj]
            if len(set(occupied)) != len(occupied):
                raise InvalidSample("capacity", f"two ions share a position at t={t}")

        if traj[window.mover][-1] not in window.goal_nodes:
            raise InvalidSample("gate_feasibility", f"mover ends at {traj[window.mover][-1]}")

    @staticmethod
    def schedule(traj: Dict[Any, List[Any]], problem: QUBOProblem) -> List[tuple]:
        """Order each step's simultaneous moves into sequential operations."""
        G = problem.graph
        ops: List[tuple] = []
        for t in range(problem.time_horizon):
            moves = {ion: (p[t], p[t + 1]) for ion, p in traj.items() if p[t] != p[t + 1]}
            occupied = {p[t]: ion for ion, p in traj.items()}

            # Two ions exchanging places in one step.
            for a, (u, v) in list(moves.items()):
                b = occupied.get(v)
                if b is not None and b in moves and moves[b] == (v, u):
                    if G.edges[u, v].get("label") != "swap":
                        raise InvalidSample("collision_avoidance", f"exchange over {u}-{v}")
                    ops.append(("swap", a, b))
                    del moves[a], moves[b]

            # Remaining moves: execute whichever has a free destination.
            occupied = {pos: ion for ion, pos in
                        ((i, traj[i][t + 1] if i not in moves else traj[i][t]) for i in traj)}
            while moves:
                ready = [ion for ion, (_, v) in moves.items() if v not in occupied]
                if not ready:
                    raise InvalidSample("scheduling", f"rotation cycle at t={t}")
                for ion in ready:
                    u, v = moves.pop(ion)
                    del occupied[u]
                    occupied[v] = ion
                    ops.append(("move", ion, v))
        return ops
