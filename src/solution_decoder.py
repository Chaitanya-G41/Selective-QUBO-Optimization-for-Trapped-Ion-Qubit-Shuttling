"""
solution_decoder.py
===================
Phases 8 and 9 of the Selective-QUBO methodology:
  8. Decode QUBO binary solution → physical shuttling operations.
  9. Validate + compare cost → accept QUBO route or fall back to heuristic.


======================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, TYPE_CHECKING

from congestion_handler import WindowInfo
from qubo_solver import QUBOSolution

if TYPE_CHECKING:
    from position_graph import Placement, PositionGraph

logger = logging.getLogger("shaw_router.solution_decoder")


# ---------------------------------------------------------------------------
# Output data structure  (DO NOT CHANGE – congestion_handler.py will consume this)
# ---------------------------------------------------------------------------

@dataclass
class DecodedSolution:
    accepted: bool
    decoded_path: Optional[List[Any]]
    C_QUBO: int
    C_heuristic: int
    improvement: float
    violation: Optional[str]
    trajectories: Dict[Any, List[Any]] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Outcome of one validation check."""
    passed: bool
    constraint: str       # e.g. "movement_legality"
    detail: str = ""      # human-readable failure description


class SolutionDecoder:
    def __init__(self, strict_capacity: bool = True) -> None:
        self.strict_capacity = strict_capacity

    def decode_and_validate(
        self,
        solution: QUBOSolution,
        placement: "Placement",
        pg: "PositionGraph",
    ) -> DecodedSolution:
        window = solution.problem.window
        # C_heuristic: cost the existing SHAW/greedy approach would pay to
        # clear this same congestion, for a fair comparison against
        # C_QUBO. Reuses CongestionHandler's own C_est model (see
        # congestion_handler.py's _compute_rho: C_est = C_LB + 2*|blocked|)
        # rather than inventing a second, inconsistent definition here.
        # NOTE: this replaces the original placeholder
        # `len(window.blocked_path) + len(window.blocked_path)`, which
        # just doubled the blocked-position count and had no connection
        # to actual path distance. Flagged for Chaitanya to confirm.
        C_heuristic = self._compute_heuristic_cost(window, pg)

        # Step 1 – quick pre-filter: solver already flagged infeasible
        if not solution.is_feasible:
            logger.info("QUBO solution pre-rejected: solver flagged infeasible (energy=%.3f)", solution.energy)
            return DecodedSolution(
                accepted=False, decoded_path=None,
                C_QUBO=0, C_heuristic=C_heuristic,
                improvement=0.0, violation="infeasible_energy",
            )

        # Step 2 – decode trajectories
        trajectories = self._decode_trajectories(solution)
        if not trajectories:
            return DecodedSolution(
                accepted=False, decoded_path=None,
                C_QUBO=0, C_heuristic=C_heuristic,
                improvement=0.0, violation="decode_failed",
                trajectories={},
            )

        # Step 3 – run all validation checks
        checks = [
            self._check_movement_legality(trajectories, window, pg),
            self._check_occupancy(trajectories, window),
            self._check_capacity(trajectories, window, pg),
            self._check_collision_avoidance(trajectories, window),
            self._check_gate_feasibility(trajectories, window),
        ]

        for result in checks:
            if not result.passed:
                logger.info(
                    "QUBO solution rejected: %s failed – %s",
                    result.constraint, result.detail,
                )
                return DecodedSolution(
                    accepted=False, decoded_path=None,
                    C_QUBO=0, C_heuristic=C_heuristic,
                    improvement=0.0, violation=result.constraint,
                    trajectories=trajectories,
                )

        # Step 4 – extract moving ion path. A path with fewer than 2
        # nodes is degenerate (a zero-hop "solution" that claims 100%
        # improvement while routing nothing -- observed live when the
        # mover's whole trajectory collapses to one position) and is
        # rejected outright rather than compared.
        decoded_path = self._extract_moving_ion_path(trajectories, window)
        if not decoded_path or len(decoded_path) < 2:
            logger.info(
                "QUBO solution rejected: degenerate %s-node path.",
                0 if not decoded_path else len(decoded_path),
            )
            return DecodedSolution(
                accepted=False, decoded_path=None,
                C_QUBO=0, C_heuristic=C_heuristic,
                improvement=0.0, violation="empty_path",
                trajectories=trajectories,
            )
        C_QUBO = len(decoded_path) - 1

        # Step 5 – cost comparison (methodology §9)
        if C_QUBO >= C_heuristic:
            logger.info(
                "QUBO solution valid but not better: C_QUBO=%d >= C_heur=%d; "
                "falling back to heuristic.",
                C_QUBO, C_heuristic,
            )
            return DecodedSolution(
                accepted=False, decoded_path=None,
                C_QUBO=C_QUBO, C_heuristic=C_heuristic,
                improvement=0.0, violation="not_better_than_heuristic",
                trajectories=trajectories,
            )

        improvement = (C_heuristic - C_QUBO) / max(C_heuristic, 1)
        logger.info(
            "QUBO solution ACCEPTED: C_QUBO=%d < C_heur=%d (%.1f%% improvement)",
            C_QUBO, C_heuristic, improvement * 100,
        )
        return DecodedSolution(
            accepted=True,
            decoded_path=decoded_path,
            C_QUBO=C_QUBO,
            C_heuristic=C_heuristic,
            improvement=improvement,
            violation=None,
            trajectories=trajectories,
        )

    # -----------------------------------------------------------------------
    # Heuristic cost (used for the C_QUBO < C_heuristic comparison)
    # -----------------------------------------------------------------------

    def _compute_heuristic_cost(
        self,
        window: WindowInfo,
        pg: "PositionGraph",
    ) -> int:
        """
        Cost the existing greedy/SHAW path would pay to clear this
        congestion, as a baseline for comparing against C_QUBO.

        Mirrors CongestionHandler._compute_rho's own C_est model:
            C_LB  = uncongested shortest-path hop count (source -> target)
            C_est = C_LB + 2 * |blocked_path|
        (each blocked position costs ~1 hop to park aside, ~1 to undo).
        Reusing that formula keeps the two modules' notion of "heuristic
        cost" consistent rather than introducing a second, different one.
        """
        try:
            C_LB = len(pg.shortest_path(window.source, window.target)) - 1
        except Exception:
            # No path at all between source/target in the static graph
            # (shouldn't normally happen -- SHAW already routed this way
            # once). Fall back to blocked-path length as a rough estimate
            # rather than crashing the whole decode pipeline over it.
            logger.warning(
                "Could not compute shortest_path(%r, %r) for heuristic cost; "
                "falling back to blocked-path length.",
                window.source, window.target,
            )
            C_LB = len(window.blocked_path) + 1
        return C_LB + 2 * len(window.blocked_path)

    # -----------------------------------------------------------------------
    # Decoding
    # -----------------------------------------------------------------------

    def _decode_trajectories(
        self,
        solution: QUBOSolution,
    ) -> Dict[Any, List[Any]]:
        """
        Read solution.sample and solution.problem.inv_var_map to
        reconstruct trajectory[ion][t] = position for every ion in
        window.active_ions, over t = 0..T.

        Returns {} (triggering the "decode_failed" path in
        decode_and_validate) if any (ion, t) has more than one
        position bit set to 1 -- an internally inconsistent sample.
        """
        problem = solution.problem
        inv_var_map = problem.inv_var_map
        window = problem.window
        T = problem.time_horizon

        ions = list(window.active_ions.keys())

        # assigned[ion][t] -> list of positions with x_{ion,pos,t} = 1
        assigned: Dict[Any, Dict[int, List[Any]]] = {ion: {} for ion in ions}

        for label, bit in solution.sample.items():
            if bit != 1:
                continue
            mapped = inv_var_map.get(label)
            if mapped is None:
                # Not a real (ion, pos, t) variable -- e.g. a Rosenberg
                # auxiliary variable from quadratization. Not our concern
                # here; only inv_var_map entries carry physical meaning.
                continue
            ion, pos, t = mapped
            if ion not in assigned:
                # A variable for an ion outside window.active_ions --
                # shouldn't happen per the formulator's own spec, but
                # don't silently track it if it does.
                continue
            assigned[ion].setdefault(t, []).append(pos)

        trajectories: Dict[Any, List[Any]] = {}
        for ion in ions:
            current = window.active_ions.get(ion)  # known t=0 position
            traj: List[Any] = []
            for t in range(T + 1):
                positions_at_t = assigned.get(ion, {}).get(t, [])
                if len(positions_at_t) > 1:
                    logger.info(
                        "Decode failed: ion %s has %d positions set at t=%d "
                        "(expected exactly one): %s",
                        ion, len(positions_at_t), t, positions_at_t,
                    )
                    return {}
                if positions_at_t:
                    current = positions_at_t[0]
                elif current is None:
                    # No bit set yet and no known starting position --
                    # genuinely undecodable for this ion.
                    logger.info(
                        "Decode failed: ion %s has no position at t=%d and "
                        "no known starting position.", ion, t,
                    )
                    return {}
                # else: no bit set at this t -> ion stayed at `current`
                # ("If no variable is 1 for (ion, t), the ion didn't move")
                traj.append(current)
            trajectories[ion] = traj

        return trajectories

    def _extract_moving_ion_path(
        self,
        trajectories: Dict[Any, List[Any]],
        window: WindowInfo,
    ) -> Optional[List[Any]]:
        """
        Extract the deduplicated position sequence for the ion that
        starts at window.source (the ion SHAW was originally trying to
        route). Consecutive repeated positions are collapsed, since the
        ion may "stay" at one position for several timesteps before its
        next real hop.
        """
        moving_ion = self._find_moving_ion(window)
        if moving_ion is None or moving_ion not in trajectories:
            return None

        raw_path = trajectories[moving_ion]
        deduped: List[Any] = []
        for pos in raw_path:
            if not deduped or deduped[-1] != pos:
                deduped.append(pos)
        return deduped

    def _find_moving_ion(self, window: WindowInfo) -> Optional[Any]:
        """The active ion whose current position is window.source."""
        for ion, pos in window.active_ions.items():
            if pos == window.source:
                return ion
        return None

    # -----------------------------------------------------------------------
    # Validation checks
    # -----------------------------------------------------------------------

    def _check_movement_legality(
        self,
        trajectories: Dict[Any, List[Any]],
        window: WindowInfo,
        pg: "PositionGraph",
    ) -> ValidationResult:
        """Every consecutive (t, t+1) step is either a stay or a legal
        G_p edge."""
        G = getattr(pg, "graph", None)
        for ion, traj in trajectories.items():
            for t in range(len(traj) - 1):
                u, v = traj[t], traj[t + 1]
                if u == v:
                    continue  # stayed -- always legal
                if G is None or not G.has_edge(u, v):
                    return ValidationResult(
                        passed=False,
                        constraint="movement_legality",
                        detail=(
                            f"ion {ion}: illegal move {u!r} -> {v!r} at "
                            f"t={t} (not an edge in G_p)"
                        ),
                    )
        return ValidationResult(passed=True, constraint="movement_legality")

    def _check_occupancy(
        self,
        trajectories: Dict[Any, List[Any]],
        window: WindowInfo,
    ) -> ValidationResult:
        """No two *active* ions occupy the same position at the same
        timestep. (Capacity vs. fixed obstacle ions is handled
        separately in _check_capacity.)"""
        T = max((len(traj) for traj in trajectories.values()), default=0)
        for t in range(T):
            position_to_ions: Dict[Any, List[Any]] = {}
            for ion, traj in trajectories.items():
                if t < len(traj):
                    position_to_ions.setdefault(traj[t], []).append(ion)
            for pos, ions_here in position_to_ions.items():
                if len(ions_here) > 1:
                    return ValidationResult(
                        passed=False,
                        constraint="occupancy",
                        detail=(
                            f"position {pos!r} occupied by {len(ions_here)} "
                            f"active ions at t={t}: {ions_here}"
                        ),
                    )
        return ValidationResult(passed=True, constraint="occupancy")

    def _check_capacity(
        self,
        trajectories: Dict[Any, List[Any]],
        window: WindowInfo,
        pg: "PositionGraph",
    ) -> ValidationResult:
        """
        occupancy(v, t) <= cap(v) for every position v and timestep t,
        counting BOTH active ions (from trajectories) and obstacle_ions
        (fixed outside the window, but still physically present).

        src/position_graph.py's graph has no 'capacity' node attribute:
        a trap's capacity is realized by having one graph node per slot
        (e.g. "t0:0", "t0:1"), so every individual node holds at most
        one ion by construction. cap(v) therefore defaults to 1 for any
        node without an explicit 'capacity' attribute (segments included)
        -- i.e. slot-style, per the team's position_graph.py model.
        """
        G = getattr(pg, "graph", None)

        def capacity_of(pos: Any) -> int:
            if G is not None and pos in G:
                cap = G.nodes[pos].get("capacity")
                if cap is not None:
                    return cap
            return 1  # slot-style default (see docstring)

        T = max((len(traj) for traj in trajectories.values()), default=0)

        for t in range(T):
            position_to_ions: Dict[Any, List[Any]] = {}
            for ion, traj in trajectories.items():
                if t < len(traj):
                    position_to_ions.setdefault(traj[t], []).append(ion)
            # Obstacle ions are fixed boundary conditions -- same
            # position at every timestep in this window.
            for obstacle_ion, obstacle_pos in window.obstacle_ions.items():
                position_to_ions.setdefault(obstacle_pos, []).append(obstacle_ion)

            for pos, ions_here in position_to_ions.items():
                cap = capacity_of(pos)
                if len(ions_here) > cap:
                    detail = (
                        f"position {pos!r} at t={t} holds {len(ions_here)} "
                        f"ions {ions_here} > capacity {cap}"
                    )
                    if self.strict_capacity:
                        return ValidationResult(
                            passed=False, constraint="capacity", detail=detail,
                        )
                    logger.warning("Capacity violation (non-strict): %s", detail)

        return ValidationResult(passed=True, constraint="capacity")

    def _check_collision_avoidance(
        self,
        trajectories: Dict[Any, List[Any]],
        window: WindowInfo,
    ) -> ValidationResult:
        """No crossing moves: ion A going u->v while ion B goes v->u in
        the same step is an implicit swap with no explicit swap gate,
        which isn't physically valid here."""
        T = max((len(traj) for traj in trajectories.values()), default=0)
        for t in range(T - 1):
            moves: List[Tuple[Any, Any, Any]] = []
            for ion, traj in trajectories.items():
                if t + 1 < len(traj):
                    u, v = traj[t], traj[t + 1]
                    if u != v:
                        moves.append((ion, u, v))
            move_set = {(u, v) for _, u, v in moves}
            for ion, u, v in moves:
                if (v, u) in move_set:
                    return ValidationResult(
                        passed=False,
                        constraint="collision_avoidance",
                        detail=(
                            f"crossing move between {u!r} and {v!r} at "
                            f"t={t} (ion {ion} is one side of it)"
                        ),
                    )
        return ValidationResult(passed=True, constraint="collision_avoidance")

    def _check_gate_feasibility(
        self,
        trajectories: Dict[Any, List[Any]],
        window: WindowInfo,
    ) -> ValidationResult:
        """The moving ion (starting at window.source) must end at
        window.target by t=T."""
        moving_ion = self._find_moving_ion(window)
        if moving_ion is None:
            return ValidationResult(
                passed=False,
                constraint="gate_feasibility",
                detail=f"no active ion found starting at source {window.source!r}",
            )
        traj = trajectories.get(moving_ion)
        if not traj:
            return ValidationResult(
                passed=False,
                constraint="gate_feasibility",
                detail=f"no decoded trajectory for moving ion {moving_ion}",
            )
        final_pos = traj[-1]
        if final_pos != window.target:
            return ValidationResult(
                passed=False,
                constraint="gate_feasibility",
                detail=(
                    f"moving ion {moving_ion} ended at {final_pos!r}, "
                    f"expected target {window.target!r}"
                ),
            )
        return ValidationResult(passed=True, constraint="gate_feasibility")

    # -----------------------------------------------------------------------
    # Placement update (called ONLY if accepted=True)
    # -----------------------------------------------------------------------

    def apply_to_placement(
        self,
        decoded: DecodedSolution,
        placement: "Placement",
        pg: "PositionGraph",
        skip_ions: FrozenSet[Any] = frozenset(),
    ) -> None:
        """
        Apply accepted trajectories to the live Placement, in timestep
        order, skipping any ion in `skip_ions` (left exactly where it
        is). The selective-QUBO hookup skips the moving ion: its
        decoded path is walked by ShawRoutingPass instead, while
        blockers are pre-cleared here -- replaying the mover too would
        walk it twice (once to the target via apply, once backward
        through the path via the walk).

        Within a single timestep, one ion's move can depend on another
        ion vacating its target position first (e.g. ion A moves into
        the slot ion B is leaving this same step). We resolve that by
        repeatedly applying whichever moves are currently unblocked
        until no more progress can be made. A true 2-ion swap (A->B's
        spot, B->A's spot in the same step) can never resolve this way
        -- but _check_collision_avoidance already rejects that case
        before we get here. A longer rotation cycle (A->B->C->A) is NOT
        caught by that check and would deadlock here; we raise a clear
        error rather than hang, since that's a real gap in the 5
        validation checks as specified, not something to paper over.
        """
        if not decoded.accepted:
            raise ValueError(
                "apply_to_placement called on a rejected DecodedSolution "
                f"(violation={decoded.violation!r}); nothing to apply."
            )

        G = getattr(pg, "graph", None)
        T = max((len(traj) for traj in decoded.trajectories.values()), default=0)

        for t in range(1, T):
            pending: List[Tuple[Any, Any, Any]] = []
            for ion, traj in decoded.trajectories.items():
                if ion in skip_ions or t >= len(traj):
                    continue
                prev_pos, new_pos = traj[t - 1], traj[t]
                if prev_pos != new_pos:
                    pending.append((ion, prev_pos, new_pos))

            while pending:
                progressed: List[Tuple[Any, Any, Any]] = []
                still_pending: List[Tuple[Any, Any, Any]] = []

                for ion, prev_pos, new_pos in pending:
                    # Divergence is a hard invariant violation, not a
                    # scheduling concern -- check it before the occupancy
                    # deferral below, or a move that would never resolve
                    # (because the live state already disagrees with the
                    # plan) just gets silently retried forever instead of
                    # surfaced.
                    current_actual = placement.position_of(ion)
                    if current_actual != prev_pos:
                        raise ValueError(
                            f"apply_to_placement: live Placement has ion "
                            f"{ion} at {current_actual!r}, but the decoded "
                            f"trajectory expected {prev_pos!r} at t={t - 1}. "
                            "Placement has diverged from the decoded "
                            "solution; refusing to apply."
                        )

                    if placement.is_occupied(new_pos):
                        still_pending.append((ion, prev_pos, new_pos))
                        continue

                    if G is not None and not G.has_edge(prev_pos, new_pos):
                        raise ValueError(
                            f"apply_to_placement: illegal move for ion {ion}: "
                            f"{prev_pos!r} -> {new_pos!r} at t={t} is not an "
                            "edge in G_p (should have been caught by "
                            "_check_movement_legality)."
                        )

                    try:
                        placement.move(ion, new_pos)
                    except ValueError as e:
                        raise ValueError(
                            f"apply_to_placement: placement.move({ion}, "
                            f"{new_pos!r}) failed even though validation "
                            f"passed: {e}"
                        ) from e
                    progressed.append((ion, prev_pos, new_pos))

                if not progressed:
                    raise ValueError(
                        f"apply_to_placement: deadlocked at t={t} -- "
                        f"{len(still_pending)} move(s) can't proceed because "
                        f"their targets are occupied by ions also waiting to "
                        f"move this step (a rotation cycle longer than 2, "
                        f"which _check_collision_avoidance doesn't catch): "
                        f"{still_pending}"
                    )
                pending = still_pending


if __name__ == "__main__":
    print("solution_decoder.py: all 8 SolutionDecoder methods implemented.")
    print("Run: pytest tests/test_solution_decoder.py -v")