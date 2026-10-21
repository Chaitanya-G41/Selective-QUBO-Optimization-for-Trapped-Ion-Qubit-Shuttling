"""
shaw_routing_pass.py
====================
Core SHAW routing loop for a trapped-ion QCCD machine, as a BQSKit pass.

Algorithm:
  1. Build (or accept) a PositionGraph + Placement; logical qudit i
     starts in trap i, slot 0.
  2. Walk the circuit DAG front layer. A two-qudit gate whose ions
     already share a trap executes directly.
  3. Otherwise pick which ion moves into the other's trap (path length
     minus a small lookahead bonus), ask the CongestionHandler how to
     get there, and execute the answer:
       - a greedy walk along a path (see walk_planner.py), or
       - an accepted selective-QUBO move plan for every ion in the
         congested window.
  4. Retire the gate and repeat.

Physical movement is recorded in the routed circuit with identity
markers, which leave the circuit's unitary unchanged:
  - IdentityGate(1) on one qudit   = one shuttle (ion hop)
  - IdentityGate(2) on two qudits  = one physical ion swap
A physical swap exchanges where two ions sit; it does NOT exchange their
quantum states, so emitting a logical SwapGate would be wrong.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import IdentityGate

from circuit_dag import CircuitDAG
from congestion_handler import CongestionHandler, Resolution
from position_graph import Placement, PositionGraph, build_linear_qccd
from walk_planner import TIME_COSTS_US, op_cost, plan_walk, trap_of, walk_cost

logger = logging.getLogger("shaw_router")

SHUTTLE_MARKER = IdentityGate(1)
ION_SWAP_MARKER = IdentityGate(2)


class ShawRoutingPass(BasePass):
    """The core SHAW routing loop. See module docstring."""

    LOOKAHEAD_SIZE = 5
    DEFAULT_TRAP_CAPACITY = 2  # room for a home ion + one visiting ion

    def __init__(
        self,
        position_graph: Optional[PositionGraph] = None,
        placement: Optional[Placement] = None,
        congestion_handler: Optional[CongestionHandler] = None,
    ) -> None:
        super().__init__()
        self._position_graph = position_graph
        self._placement = placement
        self.congestion_handler = congestion_handler or CongestionHandler()
        self.movement_time_us = 0.0  # under walk_planner.TIME_COSTS_US

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """BQSKit entry point. Mutates `circuit` in place."""
        pg, placement = self._get_or_build_machine_model(circuit)

        dag = CircuitDAG(circuit)
        program_order = dag.program_order()
        order_index = {op_id: i for i, op_id in enumerate(program_order)}
        front_layer: Deque[int] = deque(dag.initial_front_layer())
        routed = Circuit(circuit.num_qudits, circuit.radixes)
        remaining = set(dag.nodes.keys())

        stalls = 0
        stall_limit = len(dag.nodes) + 1

        def execute(op_id: int) -> None:
            nonlocal stalls
            routed.append(dag.nodes[op_id].operation)
            remaining.discard(op_id)
            front_layer.extend(dag.mark_executed(op_id))
            stalls = 0

        def defer(op_id: int, a: int, b: int, ta, tb) -> None:
            nonlocal stalls
            if self._evict_bystander(routed, pg, placement, a, b, ta, tb):
                front_layer.appendleft(op_id)  # board changed, retry now
                return
            stalls += 1
            if stalls > stall_limit:
                raise RuntimeError(
                    f"Routing deadlock: qudits {a}/{b} cannot be co-located "
                    f"(traps {ta!r}/{tb!r}) after {stalls} attempts with no "
                    "other progress."
                )
            logger.debug("Deferring gate %s (qudits %s/%s)", op_id, a, b)
            front_layer.append(op_id)

        while front_layer:
            op_id = front_layer.popleft()
            node = dag.nodes[op_id]
            if len(node.qudits) < 2:
                execute(op_id)
                continue

            a, b = node.qudits[0], node.qudits[1]
            trap_a = trap_of(pg, placement.position_of(a))
            trap_b = trap_of(pg, placement.position_of(b))
            if trap_a is not None and trap_a == trap_b:
                execute(op_id)
                continue

            lookahead = self._get_lookahead_window(
                remaining, order_index, program_order, exclude=op_id
            )
            mover, target_slot = self._choose_move(
                pg, placement, a, b, trap_a, trap_b, lookahead, dag
            )
            if mover is None:
                defer(op_id, a, b, trap_a, trap_b)
                continue

            raw_path = pg.shortest_path(placement.position_of(mover), target_slot)
            hints = self._next_partner_positions(placement, lookahead, dag)
            resolution = self.congestion_handler.resolve_path_congestion(
                raw_path, placement, pg, hints
            )
            self._execute(routed, pg, placement, resolution, mover)

            trap_a = trap_of(pg, placement.position_of(a))
            trap_b = trap_of(pg, placement.position_of(b))
            if trap_a is None or trap_a != trap_b:
                defer(op_id, a, b, trap_a, trap_b)
                continue
            execute(op_id)

        circuit.become(routed)

    # -----------------------------------------------------------------------
    # Machine model
    # -----------------------------------------------------------------------

    def _get_or_build_machine_model(
        self, circuit: Circuit
    ) -> Tuple[PositionGraph, Placement]:
        if self._position_graph is not None and self._placement is not None:
            return self._position_graph, self._placement
        pg = build_linear_qccd(
            num_traps=circuit.num_qudits,
            trap_capacity=self.DEFAULT_TRAP_CAPACITY,
        )
        placement = Placement(pg)
        for q in range(circuit.num_qudits):
            placement.place(q, pg.slots_of(f"t{q}")[0])
        return pg, placement

    @staticmethod
    def _trap_of(pg: PositionGraph, position: str) -> Optional[str]:
        return trap_of(pg, position)

    # -----------------------------------------------------------------------
    # Move selection
    # -----------------------------------------------------------------------

    def _get_lookahead_window(
        self,
        remaining: set,
        order_index: Dict[int, int],
        program_order: List[int],
        exclude: int,
    ) -> List[int]:
        window = []
        for op_id in program_order[order_index[exclude] + 1:]:
            if op_id in remaining:
                window.append(op_id)
                if len(window) >= self.LOOKAHEAD_SIZE:
                    break
        return window

    def _choose_move(
        self,
        pg: PositionGraph,
        placement: Placement,
        logical_a: int,
        logical_b: int,
        trap_a: Optional[str],
        trap_b: Optional[str],
        lookahead_ops: List[int],
        dag: CircuitDAG,
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Candidates: move A into trap_b, or B into trap_a (score = path
        length - lookahead bonus, lower is better). A qudit stranded on
        transport with no join available may instead tuck into the
        nearest free trap slot. Returns (None, None) if nothing fits.
        """
        candidates: List[Tuple[int, str]] = []
        for mover, target_trap in ((logical_a, trap_b), (logical_b, trap_a)):
            mover_pos = placement.position_of(mover)
            if target_trap is not None and self._free_slot_in_trap(pg, placement, target_trap):
                candidates.append((mover, target_trap))
            elif trap_of(pg, mover_pos) is None:
                tuck = self._nearest_free_slot(pg, placement, mover_pos)
                if tuck is not None:
                    candidates.append((mover, trap_of(pg, tuck)))

        best: Tuple[Optional[int], Optional[str]] = (None, None)
        best_score = float("inf")
        for mover, target_trap in candidates:
            slot = self._free_slot_in_trap(pg, placement, target_trap)
            mover_pos = placement.position_of(mover)
            score = len(pg.shortest_path(mover_pos, slot)) - 1 - self._parallelism_bonus(
                pg, placement, mover, slot, lookahead_ops, dag
            )
            if score < best_score:
                best_score, best = score, (mover, slot)
        return best

    @staticmethod
    def _free_slot_in_trap(
        pg: PositionGraph, placement: Placement, trap_id: str
    ) -> Optional[str]:
        for slot in pg.slots_of(trap_id):
            if not placement.is_occupied(slot):
                return slot
        return None

    def _nearest_free_slot(
        self,
        pg: PositionGraph,
        placement: Placement,
        from_pos: str,
        excluded_traps=frozenset(),
    ) -> Optional[str]:
        """Nearest free trap slot (by hops), ignoring `excluded_traps`."""
        best_slot, best_len = None, None
        seen = set()
        for _, attrs in pg.graph.nodes(data=True):
            trap_id = attrs.get("trap_id")
            if trap_id is None or trap_id in seen or trap_id in excluded_traps:
                continue
            seen.add(trap_id)
            slot = self._free_slot_in_trap(pg, placement, trap_id)
            if slot is None:
                continue
            length = len(pg.shortest_path(from_pos, slot)) - 1
            if best_len is None or length < best_len:
                best_slot, best_len = slot, length
        return best_slot

    def _evict_bystander(
        self,
        routed: Circuit,
        pg: PositionGraph,
        placement: Placement,
        logical_a: int,
        logical_b: int,
        trap_a: Optional[str],
        trap_b: Optional[str],
    ) -> bool:
        """
        Relocate one non-gate ion out of the contested traps to the
        nearest free slot elsewhere. Evictees never return to the
        contested traps, so repeated calls terminate.
        """
        contested = {t for t in (trap_a, trap_b) if t is not None}
        for trap_id in (trap_a, trap_b):
            if trap_id is None:
                continue
            for slot in pg.slots_of(trap_id):
                occupant = placement.occupant_of(slot)
                if occupant is None or occupant in (logical_a, logical_b):
                    continue
                start = placement.position_of(occupant)
                dest = self._nearest_free_slot(pg, placement, start, contested)
                if dest is None:
                    continue
                resolution = self.congestion_handler.resolve_path_congestion(
                    pg.shortest_path(start, dest), placement, pg
                )
                self._execute(routed, pg, placement, resolution, occupant)
                logger.debug("Evicted qudit %s from trap %s -> %s", occupant, trap_id, dest)
                return True
        return False

    @staticmethod
    def _next_partner_positions(placement, lookahead_ops, dag) -> Dict[int, str]:
        """{qudit: current position of its next 2-qudit gate partner}."""
        hints: Dict[int, str] = {}
        for op_id in lookahead_ops:
            qudits = dag.nodes[op_id].qudits
            if len(qudits) == 2:
                a, b = qudits
                hints.setdefault(a, placement.position_of(b))
                hints.setdefault(b, placement.position_of(a))
        return hints

    def _parallelism_bonus(
        self,
        pg: PositionGraph,
        placement: Placement,
        moved_logical: int,
        candidate_slot: str,
        lookahead_ops: List[int],
        dag: CircuitDAG,
    ) -> float:
        """Small bonus if the move also shortens an upcoming gate."""
        bonus = 0.0
        current_pos = placement.position_of(moved_logical)
        for op_id in lookahead_ops:
            qudits = dag.nodes[op_id].qudits
            if len(qudits) < 2 or moved_logical not in qudits:
                continue
            other = [q for q in qudits if q != moved_logical][0]
            other_pos = placement.position_of(other)
            current = len(pg.shortest_path(current_pos, other_pos)) - 1
            new = len(pg.shortest_path(candidate_slot, other_pos)) - 1
            if new < current:
                bonus += (current - new) * 0.1
        return bonus

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    def _execute(
        self,
        routed: Circuit,
        pg: PositionGraph,
        placement: Placement,
        resolution: Resolution,
        mover: int,
    ) -> None:
        """Execute a resolution: its QUBO plan, else the greedy walk."""
        if resolution.plan is not None:
            self.congestion_handler.stats["qubo_executed"] += 1
            self._execute_plan(routed, pg, placement, resolution.plan.ops)
        else:
            self._walk_path(routed, pg, placement, resolution.path, mover)

    def _execute_plan(
        self, routed: Circuit, pg: PositionGraph, placement: Placement, ops: List[tuple]
    ) -> None:
        """Apply a QUBO move plan: ("move", ion, dest) / ("swap", a, b)."""
        for op in ops:
            self.movement_time_us += op_cost(pg, placement, op, TIME_COSTS_US)
            if op[0] == "move":
                _, ion, dest = op
                placement.move(ion, dest)
                routed.append_gate(SHUTTLE_MARKER, (ion,))
            else:
                _, ion_a, ion_b = op
                placement.swap(ion_a, ion_b)
                routed.append_gate(ION_SWAP_MARKER, (ion_a, ion_b))

    def _walk_path(
        self,
        routed: Circuit,
        pg: PositionGraph,
        placement: Placement,
        path: List[str],
        mover: int,
    ) -> None:
        """Greedy walk of `mover` along `path` (rules in walk_planner.py)."""
        self.movement_time_us += walk_cost(pg, placement, path, mover, TIME_COSTS_US)
        for action in plan_walk(pg, placement, path, mover):
            kind = action[0]
            if kind == "move":
                placement.move(mover, action[1])
                routed.append_gate(SHUTTLE_MARKER, (mover,))
            elif kind == "vacate":
                _, occupant, hop, free_slot = action
                placement.move(occupant, free_slot)
                routed.append_gate(SHUTTLE_MARKER, (occupant,))
                placement.move(mover, hop)
                routed.append_gate(SHUTTLE_MARKER, (mover,))
            else:  # swap
                _, occupant, _hop = action
                placement.swap(mover, occupant)
                routed.append_gate(ION_SWAP_MARKER, (mover, occupant))


if __name__ == "__main__":
    import asyncio

    from bqskit.ir.gates import CNOTGate, HGate

    logging.basicConfig(level=logging.INFO)
    demo = Circuit(4)
    demo.append_gate(HGate(), (0,))
    demo.append_gate(CNOTGate(), (0, 3))
    demo.append_gate(CNOTGate(), (1, 2))
    demo.append_gate(CNOTGate(), (0, 1))
    original = demo.get_unitary()
    asyncio.run(ShawRoutingPass().run(demo, PassData(demo)))
    for op in demo.operations():
        print(f"  {op.gate.name} on {tuple(op.location)}")
    print("Unitary preserved:", original.get_distance_from(demo.get_unitary()) < 1e-8)
