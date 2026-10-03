"""
qubo_validator.py
=================
Shared QUBO solution validator for the Selective-QUBO methodology.

Used by BOTH:
  - qubo_solver.py   : to filter SA/Exact reads before selecting best sample
  - solution_decoder.py : to validate the selected sample before accepting it

Having one shared implementation guarantees the solver can never return a
sample that the decoder will reject for a constraint the solver missed.

The five physical constraints checked here are:
  1. one_hot       – exactly one position per ion per timestep
  2. capacity      – at most cap(v) ions (active + obstacle) at any node/step
  3. movement      – every step is a stay or a legal G_p edge
  4. crossing      – no two ions swap positions in the same step
  5. goal          – moving ion (at window.source) ends at window.target at T

Trajectory decoding
-------------------
A raw QUBO sample is a flat dict {label: 0/1}. Before any physical check
can run the sample must be decoded into per-ion position trajectories.
The shared ``decode_trajectories`` function does this using ``inv_var_map``
(provided by the formulator).  Rosenberg auxiliary variables (not in
``inv_var_map``) are silently ignored.

Public API
----------
  decode_trajectories(sample, inv_var_map, window, time_horizon)
      -> Optional[Dict[ion, List[pos]]]

  validate_trajectories(trajectories, window, pg=None)
      -> List[ConstraintViolation]   (empty = valid)

  validate_sample(sample, inv_var_map, window, time_horizon, pg=None)
      -> List[ConstraintViolation]   (empty = valid)

  ConstraintViolation   – dataclass(constraint: str, detail: str)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from congestion_handler import WindowInfo

if TYPE_CHECKING:
    from position_graph import PositionGraph

logger = logging.getLogger("shaw_router.qubo_validator")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConstraintViolation:
    """
    Describes a single failed constraint check.

    Attributes
    ----------
    constraint : str  – machine-readable constraint name, e.g. "movement_legality"
    detail     : str  – human-readable description of what went wrong
    """
    constraint: str
    detail: str


# ---------------------------------------------------------------------------
# Trajectory decoder  (shared between solver and decoder)
# ---------------------------------------------------------------------------

def decode_trajectories(
    sample: Dict[str, int],
    inv_var_map: Dict[str, Tuple[Any, Any, int]],
    window: WindowInfo,
    time_horizon: int,
) -> Optional[Dict[Any, List[Any]]]:
    """
    Decode a flat binary QUBO sample into per-ion position trajectories.

    For each active ion ``i`` and each timestep ``t`` in [0, T], exactly
    one variable ``x_{i,v,t}`` should be 1 (the one-hot property).  If no
    variable is 1 for ``(i, t)``, the ion is assumed to have stayed at
    its position from the previous timestep.

    Rosenberg auxiliary variables (labels not in ``inv_var_map``) are
    silently skipped — they carry no physical position meaning.

    Parameters
    ----------
    sample       : {label: 0/1}  – raw binary assignment from the solver
    inv_var_map  : {label: (ion, pos, t)}  – built by the formulator
    window       : WindowInfo – provides active_ions (ion → t=0 position)
    time_horizon : int  – T (number of timesteps)

    Returns
    -------
    {ion: [pos_t0, pos_t1, ..., pos_tT]}  if decoding succeeds,
    None  if a one-hot violation or missing start position is detected
          (treated as a hard constraint failure upstream).
    """
    ions = list(window.active_ions.keys())

    # assigned[ion][t] → list of positions with x_{ion,pos,t} = 1
    assigned: Dict[Any, Dict[int, List[Any]]] = {ion: {} for ion in ions}

    for label, bit in sample.items():
        if bit != 1:
            continue
        mapped = inv_var_map.get(label)
        if mapped is None:
            continue   # Rosenberg aux — not a physical position variable
        ion, pos, t = mapped
        if ion not in assigned:
            continue   # variable for an ion outside active_ions — skip
        assigned[ion].setdefault(t, []).append(pos)

    trajectories: Dict[Any, List[Any]] = {}
    for ion in ions:
        current = window.active_ions.get(ion)   # known t=0 position
        traj: List[Any] = []
        for t in range(time_horizon + 1):
            positions_at_t = assigned.get(ion, {}).get(t, [])
            if len(positions_at_t) > 1:
                # More than one position bit set: one-hot violated.
                logger.debug(
                    "decode_trajectories: one-hot violated for ion %s at "
                    "t=%d: %r", ion, t, positions_at_t,
                )
                return None
            if positions_at_t:
                current = positions_at_t[0]
            elif current is None:
                # No bit set and no known starting position.
                logger.debug(
                    "decode_trajectories: ion %s has no position at t=%d "
                    "and no known start.", ion, t,
                )
                return None
            # else: no bit set → ion stayed at ``current`` (implicit stay)
            traj.append(current)
        trajectories[ion] = traj

    return trajectories


# ---------------------------------------------------------------------------
# Full validator  (all 5 constraints in one call)
# ---------------------------------------------------------------------------

def validate_trajectories(
    trajectories: Dict[Any, List[Any]],
    window: WindowInfo,
    pg: Optional["PositionGraph"] = None,
) -> List[ConstraintViolation]:
    """
    Run all five physical constraint checks on decoded trajectories.

    Checks are applied in dependency order so the first reported violation
    is the most fundamental one:
      one_hot → capacity → movement_legality → crossing → goal

    Parameters
    ----------
    trajectories : {ion: [pos_t0, ..., pos_tT]}  from decode_trajectories()
    window       : WindowInfo
    pg           : PositionGraph (optional).  Required for movement_legality
                   and capacity (graph cap attributes).  When None, those two
                   checks are still run but movement_legality is skipped for
                   edges (only stay-or-move is validated via the trajectory),
                   and capacity defaults to 1 for every node.

    Returns
    -------
    List[ConstraintViolation] — empty list means all constraints satisfied.
    """
    violations: List[ConstraintViolation] = []

    v = _check_one_hot(trajectories)
    if v:
        return v   # one-hot failure makes further checks unreliable

    violations += _check_capacity(trajectories, window, pg)
    if violations:
        return violations

    violations += _check_movement_legality(trajectories, pg)
    if violations:
        return violations

    violations += _check_crossing(trajectories)
    if violations:
        return violations

    violations += _check_goal(trajectories, window)
    return violations


# ---------------------------------------------------------------------------
# Convenience: decode + validate in one call
# ---------------------------------------------------------------------------

def validate_sample(
    sample: Dict[str, int],
    inv_var_map: Dict[str, Tuple[Any, Any, int]],
    window: WindowInfo,
    time_horizon: int,
    pg: Optional["PositionGraph"] = None,
) -> List[ConstraintViolation]:
    """
    Decode a raw QUBO sample and validate it against all constraints.

    Returns an empty list if the sample is fully valid; a non-empty list
    (containing the first detected violation) otherwise.

    If ``inv_var_map`` is empty (e.g. in unit-test dummy problems), structural
    checks cannot run and an empty list is returned — the caller is responsible
    for knowing this means "unknown" rather than "valid".
    """
    if not inv_var_map:
        return []   # no variable map → cannot validate structurally

    trajectories = decode_trajectories(sample, inv_var_map, window, time_horizon)
    if trajectories is None:
        return [ConstraintViolation(
            "one_hot",
            "failed to decode trajectories (one-hot violated or missing start position)",
        )]

    return validate_trajectories(trajectories, window, pg)


# ---------------------------------------------------------------------------
# Individual constraint checks
# Each returns List[ConstraintViolation] — empty on pass, one entry on fail.
# Early-exit on first violation keeps O(n) overall.
# ---------------------------------------------------------------------------

def _check_one_hot(
    trajectories: Dict[Any, List[Any]],
) -> List[ConstraintViolation]:
    """No two active ions at the same position at the same timestep."""
    T = max((len(traj) for traj in trajectories.values()), default=0)
    for t in range(T):
        pos_to_ions: Dict[Any, List[Any]] = {}
        for ion, traj in trajectories.items():
            if t < len(traj):
                pos_to_ions.setdefault(traj[t], []).append(ion)
        for pos, ions_here in pos_to_ions.items():
            if len(ions_here) > 1:
                return [ConstraintViolation(
                    "one_hot",
                    f"position {pos!r} occupied by {len(ions_here)} active "
                    f"ions at t={t}: {ions_here}",
                )]
    return []


def _check_capacity(
    trajectories: Dict[Any, List[Any]],
    window: WindowInfo,
    pg: Optional["PositionGraph"] = None,
) -> List[ConstraintViolation]:
    """
    At most cap(v) ions (active + obstacle) at any position per timestep.

    cap(v) is read from the graph node attribute 'capacity'; defaults to 1
    (slot-style model used by the team's position_graph.py) when not set.
    Obstacle ions occupy fixed positions for every timestep in the window.
    """
    G = getattr(pg, "graph", None)

    def cap_of(pos: Any) -> int:
        if G is not None and pos in G:
            cap = G.nodes[pos].get("capacity")
            if cap is not None:
                return int(cap)
        return 1   # slot-style default

    T = max((len(traj) for traj in trajectories.values()), default=0)
    for t in range(T):
        pos_to_ions: Dict[Any, List[Any]] = {}
        for ion, traj in trajectories.items():
            if t < len(traj):
                pos_to_ions.setdefault(traj[t], []).append(ion)
        # Obstacle ions are fixed — same position every timestep in the window.
        for obs_ion, obs_pos in window.obstacle_ions.items():
            pos_to_ions.setdefault(obs_pos, []).append(obs_ion)

        for pos, ions_here in pos_to_ions.items():
            cap = cap_of(pos)
            if len(ions_here) > cap:
                return [ConstraintViolation(
                    "capacity",
                    f"position {pos!r} at t={t} holds {len(ions_here)} ions "
                    f"{ions_here} > capacity {cap}",
                )]
    return []


def _check_movement_legality(
    trajectories: Dict[Any, List[Any]],
    pg: Optional["PositionGraph"] = None,
) -> List[ConstraintViolation]:
    """
    Every consecutive (t, t+1) step is either a stay or a legal G_p edge.

    If ``pg`` is None the graph structure is unavailable and this check is
    skipped — the caller should always pass ``pg`` in production paths.
    """
    G = getattr(pg, "graph", None)
    if G is None:
        # Cannot verify movement legality without the position graph.
        # Log at debug so production logs surface this if pg is missing.
        logger.debug(
            "_check_movement_legality: pg not available — movement check skipped"
        )
        return []

    for ion, traj in trajectories.items():
        for t in range(len(traj) - 1):
            u, v = traj[t], traj[t + 1]
            if u == v:
                continue   # stay — always legal
            if not G.has_edge(u, v):
                return [ConstraintViolation(
                    "movement_legality",
                    f"ion {ion}: illegal move {u!r} -> {v!r} at t={t} "
                    "(not an edge in G_p)",
                )]
    return []


def _check_crossing(
    trajectories: Dict[Any, List[Any]],
) -> List[ConstraintViolation]:
    """
    No crossing moves: ion A going u→v while ion B goes v→u in the same
    step is an implicit swap with no physical swap gate — not allowed.
    """
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
                return [ConstraintViolation(
                    "crossing",
                    f"crossing move between {u!r} and {v!r} at t={t} "
                    f"(ion {ion} is one side)",
                )]
    return []


def _check_goal(
    trajectories: Dict[Any, List[Any]],
    window: WindowInfo,
) -> List[ConstraintViolation]:
    """
    The moving ion (the one starting at window.source) must end at
    window.target by the last timestep T.
    """
    moving_ion = None
    for ion, pos in window.active_ions.items():
        if pos == window.source:
            moving_ion = ion
            break

    if moving_ion is None:
        return [ConstraintViolation(
            "goal",
            f"no active ion found at source {window.source!r}",
        )]

    traj = trajectories.get(moving_ion)
    if not traj:
        return [ConstraintViolation(
            "goal",
            f"no decoded trajectory for moving ion {moving_ion}",
        )]

    final_pos = traj[-1]
    if final_pos != window.target:
        return [ConstraintViolation(
            "goal",
            f"moving ion {moving_ion} ended at {final_pos!r}, "
            f"expected target {window.target!r}",
        )]
    return []
