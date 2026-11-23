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
from collections import deque
import heapq
import itertools
import networkx as nx

ACTION_COST = {"move": 1, "vacate": 2, "swap": 1, "displace": 1}

# Cost models price one operation by the edge it uses ("swap" = in-trap
# repositioning, "merge_split" = trap <-> transport, "move" = transport
# <-> transport) plus "ion_swap" for two ions exchanging places.
UNIT_COSTS = {"swap": 1.0, "merge_split": 1.0, "move": 1.0, "ion_swap": 1.0}
# Approximate primitive durations in microseconds: split/merge 80 and
# transport 10 (Cyclone, arXiv:2511.15910). The physical ion-swap
# duration of 300 and in-trap repositioning of 10 are configurable
# modeling assumptions, not calibrated hardware measurements. A physical
# ion swap must not be confused with a three-CX logical SWAP gate.
# This reports summed movement work, not parallel execution makespan.
TIME_COSTS_US = {"swap": 10.0, "merge_split": 80.0, "move": 10.0, "ion_swap": 300.0}


def trap_of(pg: Any, position: Any) -> Optional[str]:
    """trap_id of a trap-slot position, or None for a segment."""
    return pg.graph.nodes[position].get("trap_id")


def plan_walk(pg: Any, placement: Any, path: List[Any], mover: Any) -> List[tuple]:
    """Simulate walking `mover` along `path` without touching `placement`."""
    if not path or placement.position_of(mover) != path[0]:
        raise ValueError("Walk must start at the mover's position")
    occupancy = {v: placement.occupant_of(v) for v in pg.graph
                 if placement.is_occupied(v) and placement.occupant_of(v) != mover}
    actions: List[tuple] = []
    mover_pos = path[0]
    for next_pos in path[1:]:
        if next_pos == mover_pos:
            continue
        if not pg.graph.has_edge(mover_pos, next_pos):
            raise ValueError("Walk contains a nonadjacent hop")
        occupant = occupancy.get(next_pos)
        if occupant is not None:
            next_trap = trap_of(pg, next_pos)
            free_slot = next((s for s in pg.slots_of(next_trap)
                              if s not in occupancy and s != mover_pos), None) if next_trap is not None else None
            if free_slot is not None:
                del occupancy[next_pos]
                occupancy[free_slot] = occupant
                actions.append(("vacate", occupant, next_pos, free_slot))
                mover_pos = next_pos
                continue
            if pg.graph.edges[mover_pos, next_pos]["label"] == "swap":
                occupancy[mover_pos] = occupant
                del occupancy[next_pos]
                actions.append(("swap", occupant, next_pos))
                mover_pos = next_pos
                continue
            # Find a vacancy without passing through the mover. Push the
            # occupied chain backwards into it; every hop has a free target.
            queue = deque([next_pos])
            parent = {next_pos: None}
            vacancy = None
            while queue:
                node = queue.popleft()
                if node not in occupancy:
                    vacancy = node
                    break
                for neighbor in pg.graph.neighbors(node):
                    if neighbor != mover_pos and neighbor not in parent:
                        parent[neighbor] = node
                        queue.append(neighbor)
            if vacancy is None:
                for kind, source, dest in _clear_dead_end(pg, occupancy, mover_pos, next_pos):
                    if kind == "swap":
                        ion = occupancy.pop(dest)
                        occupancy[source] = ion
                        actions.append(("swap", ion, dest))
                        mover_pos = dest
                    elif source == mover_pos:
                        actions.append(("move", dest))
                        mover_pos = dest
                    else:
                        ion = occupancy.pop(source)
                        occupancy[dest] = ion
                        actions.append(("displace", ion, dest))
                continue
            while parent[vacancy] is not None:
                source = parent[vacancy]
                ion = occupancy.pop(source)
                occupancy[vacancy] = ion
                actions.append(("displace", ion, vacancy))
                vacancy = source
        actions.append(("move", next_pos))
        mover_pos = next_pos
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
        elif action[0] == "displace":
            ops.append(("move", action[1], action[2]))
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
        self._pg = pg
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
            validate_operation(self._pg, self, (kind, a, b))
            if kind == "move":
                del self._occupant[self._position[a]]
                self._occupant[b] = a
                self._position[a] = b
            else:
                pa, pb = self._position[a], self._position[b]
                self._occupant[pa], self._occupant[pb] = b, a
                self._position[a], self._position[b] = pb, pa


def validate_operation(pg: Any, board: Any, op: tuple) -> None:
    """Reject illegal physical operations before mutating any placement."""
    kind, a, b = op
    source = board.position_of(a)
    if kind == "move":
        if not pg.graph.has_edge(source, b) or board.is_occupied(b):
            raise ValueError(f"Illegal move {op!r} from {source!r}")
    elif kind == "swap":
        target = board.position_of(b)
        if (not pg.graph.has_edge(source, target)
                or pg.graph.edges[source, target].get("label") != "swap"):
            raise ValueError(f"Illegal physical swap {op!r}")
    else:
        raise ValueError(f"Unknown physical operation {kind!r}")


def _clear_dead_end(pg, occupancy, mover, target, _radius=4):
    """A* vacancy puzzle for a saturated corridor, with interchangeable blockers.

    Only the mover's identity matters to the search. Replaying source/dest
    actions preserves every blocker's real identity. Occupied swaps are allowed
    only inside a trap. Most walks never need this fallback.
    """
    distances = nx.single_source_shortest_path_length(pg.graph, target)
    region = set(nx.single_source_shortest_path_length(pg.graph, mover, cutoff=_radius))
    start = (mover, frozenset(occupancy))
    counter = itertools.count()
    queue = [(distances[mover], 0, next(counter), start)]
    best = {start: 0}
    parents = {}
    while queue:
        _, cost, _, state = heapq.heappop(queue)
        if best[state] != cost:
            continue
        current, blockers = state
        if current == target:
            result = []
            while state != start:
                state, action = parents[state]
                result.append(action)
            return result[::-1]
        occupied = blockers | {current}
        transitions = []
        # Mover steps/swaps first; other ions only move into vacancies.
        for dest in pg.graph.neighbors(current):
            if dest not in region:
                continue
            if dest not in blockers:
                transitions.append(((dest, blockers), ("move", current, dest)))
            elif pg.graph.edges[current, dest]["label"] == "swap":
                transitions.append(((dest, (blockers - {dest}) | {current}), ("swap", current, dest)))
        for source in pg.graph:  # stable graph order for reproducibility
            if source not in region or source not in blockers:
                continue
            for dest in pg.graph.neighbors(source):
                if dest in region and dest not in occupied:
                    transitions.append(((current, (blockers - {source}) | {dest}), ("move", source, dest)))
        for following, action in transitions:
            if cost + 1 < best.get(following, float("inf")):
                best[following] = cost + 1
                parents[following] = (state, action)
                heapq.heappush(queue, (cost + 1 + distances[following[0]], cost + 1, next(counter), following))
        if len(best) > 100000:
            break
    if len(region) < len(pg.graph) and _radius < 8:
        return _clear_dead_end(pg, occupancy, mover, target, _radius + 2)
    raise ValueError("No legal route found within corridor search budget")
