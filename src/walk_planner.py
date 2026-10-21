"""
walk_planner.py
===============
Greedy (SHAW-style) walk of one ion along a fixed path, as a pure dry run.

Shared by ShawRoutingPass (which executes the actions) and
CongestionHandler (which prices the greedy walk to decide whether a QUBO
plan is cheaper). Keeping one implementation guarantees the estimate and
the execution can never drift apart.

Actions:
    ("move", dest)                       mover hops onto a free position
    ("vacate", occupant, hop, free_slot) occupant steps aside inside its
                                         trap, then the mover hops in
    ("swap", occupant, hop)              in-place exchange of two ions

Cost (in shuttle operations): move = 1, vacate = 2, swap = 1.

Also provides BoardView, a scratch copy of the placement for simulating
plans, and walk_ops, the greedy walk expressed as plan operations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ACTION_COST = {"move": 1, "vacate": 2, "swap": 1}

# Cost models price one operation by the edge it uses ("swap" = in-trap
# repositioning, "merge_split" = trap <-> transport, "move" = transport
# <-> transport) plus "ion_swap" for two ions exchanging places.
UNIT_COSTS = {"swap": 1.0, "merge_split": 1.0, "move": 1.0, "ion_swap": 1.0}
# Approximate primitive durations in microseconds: split/merge 80 and
# transport 10 (Cyclone, arXiv:2511.15910), gate-based ion swap 300
# (3 two-qubit gates, Moveless, arXiv:2508.03914); in-trap repositioning
# 10 is an assumption. Used for reporting and as an optional objective.
TIME_COSTS_US = {"swap": 10.0, "merge_split": 80.0, "move": 10.0, "ion_swap": 300.0}


def trap_of(pg: Any, position: Any) -> Optional[str]:
    """trap_id of a trap-slot position, or None for a segment."""
    return pg.graph.nodes[position].get("trap_id")


def plan_walk(pg: Any, placement: Any, path: List[Any], mover: Any) -> List[tuple]:
    """Simulate walking `mover` along `path` without touching `placement`."""
    # Occupancy of every OTHER ion on the path and in the traps it touches.
    occupancy: Dict[Any, Any] = {}
    for pos in path:
        positions = [pos]
        trap = trap_of(pg, pos)
        if trap is not None:
            positions += pg.slots_of(trap)
        for p in positions:
            occ = placement.occupant_of(p)
            if occ is not None and occ != mover:
                occupancy[p] = occ

    actions: List[tuple] = []
    mover_pos = path[0]
    for next_pos in path[1:]:
        if next_pos == mover_pos:
            continue
        occupant = occupancy.get(next_pos)
        if occupant is None:
            mover_pos = next_pos
            actions.append(("move", next_pos))
            continue
        next_trap = trap_of(pg, next_pos)
        if next_trap is not None:
            free_slot = next(
                (s for s in pg.slots_of(next_trap)
                 if s not in occupancy and s != mover_pos),
                None,
            )
            if free_slot is not None:
                del occupancy[next_pos]
                occupancy[free_slot] = occupant
                actions.append(("vacate", occupant, next_pos, free_slot))
                mover_pos = next_pos
                continue
        occupancy[mover_pos] = occupant
        del occupancy[next_pos]
        mover_pos = next_pos
        actions.append(("swap", occupant, next_pos))
    return actions


def walk_cost(
    pg: Any, placement: Any, path: List[Any], mover: Any, costs: Optional[Dict] = None
) -> float:
    """Cost of the greedy walk along `path`: its number of shuttle
    operations, or its price under the cost model `costs`."""
    if costs is None:
        return sum(ACTION_COST[a[0]] for a in plan_walk(pg, placement, path, mover))
    return ops_cost(pg, placement, walk_ops(pg, placement, path, mover), costs)


def op_cost(pg: Any, board: Any, op: tuple, costs: Dict) -> float:
    """Price of one plan operation on the current board."""
    if op[0] == "swap":
        return costs["ion_swap"]
    return costs[pg.graph.edges[board.position_of(op[1]), op[2]]["label"]]


def ops_cost(pg: Any, placement: Any, ops: List[tuple], costs: Dict) -> float:
    """Price of a sequence of plan operations starting from `placement`."""
    board = BoardView(pg, placement)
    total = 0.0
    for op in ops:
        total += op_cost(pg, board, op, costs)
        board.apply([op])
    return total


def walk_ops(pg: Any, placement: Any, path: List[Any], mover: Any) -> List[tuple]:
    """The greedy walk as plan operations: ("move", ion, dest) / ("swap", a, b)."""
    ops: List[tuple] = []
    for action in plan_walk(pg, placement, path, mover):
        if action[0] == "move":
            ops.append(("move", mover, action[1]))
        elif action[0] == "vacate":
            _, occupant, hop, free_slot = action
            ops += [("move", occupant, free_slot), ("move", mover, hop)]
        else:
            ops.append(("swap", mover, action[1]))
    return ops


class BoardView:
    """
    A scratch copy of who-sits-where, with the same read interface as
    Placement. Lets the planner simulate operations without touching the
    live placement.
    """

    def __init__(self, pg: Any, placement: Any) -> None:
        self._occupant = {v: placement.occupant_of(v) for v in pg.graph
                          if placement.is_occupied(v)}
        self._position = {q: v for v, q in self._occupant.items()}

    def occupant_of(self, position: Any) -> Optional[Any]:
        return self._occupant.get(position)

    def is_occupied(self, position: Any) -> bool:
        return position in self._occupant

    def position_of(self, ion: Any) -> Any:
        return self._position[ion]

    def apply(self, ops: List[tuple]) -> None:
        for kind, a, b in ops:
            if kind == "move":
                del self._occupant[self._position[a]]
                self._occupant[b] = a
                self._position[a] = b
            else:
                pa, pb = self._position[a], self._position[b]
                self._occupant[pa], self._occupant[pb] = b, a
                self._position[a], self._position[b] = pb, pa
