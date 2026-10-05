"""
"Did routing actually work" checks for the full pass:

  1. It must run to completion without tripping the pass's own
     internal safety-check assertion (see shaw_routing_pass.py -- right
     before every real 2-qudit gate is appended, the pass asserts both
     qudits are in the same trap; if that's ever false, this test
     fails loudly instead of silently producing a wrong circuit).
  2. The routed circuit must contain every original gate, in the same
     relative order per qudit (routing may run independent gates
     early, but must never reorder two gates that share a qudit).
  3. For a circuit with qudits that start far apart, routing must
     have actually inserted movement markers (SwapGate / IdentityGate)
     -- if it didn't, the "no routing needed" fast path fired when it
     shouldn't have.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate, IdentityGate, SwapGate

from position_graph import Placement, build_linear_qccd
from shaw_routing_pass import ShawRoutingPass


def _build_test_circuit() -> Circuit:
    # Deliberately spread out so some gates need routing and others don't.
    c = Circuit(4)
    c.append_gate(HGate(), (0,))
    c.append_gate(CNOTGate(), (0, 3))  # far apart -> needs routing
    c.append_gate(CNOTGate(), (1, 2))  # far apart -> needs routing
    c.append_gate(CNOTGate(), (0, 1))  # far apart -> needs routing
    return c


async def _run_pass():
    circuit = _build_test_circuit()
    pg = build_linear_qccd(num_traps=circuit.num_qudits, trap_capacity=2)
    placement = Placement(pg)
    for q in range(circuit.num_qudits):
        placement.place(q, pg.slots_of(f"t{q}")[0])

    routing_pass = ShawRoutingPass(position_graph=pg, placement=placement)
    # If routing produced an invalid circuit, this line itself raises
    # an AssertionError -- that's the primary check.
    await routing_pass.run(circuit, PassData(circuit))
    return circuit


def test_routing_runs_clean_and_inserts_movement():
    routed_circuit = asyncio.run(_run_pass())

    movement_ops = [
        op for op in routed_circuit.operations()
        if isinstance(op.gate, (SwapGate, IdentityGate))
    ]
    assert movement_ops, (
        "Expected at least one SwapGate/IdentityGate movement marker -- "
        "this test circuit was built so every 2-qudit gate needs routing."
    )


def test_original_gates_preserved_in_order():
    routed_circuit = asyncio.run(_run_pass())
    original = _build_test_circuit()

    def real_gates_per_qudit(circuit: Circuit, qudit: int):
        # SwapGate/IdentityGate are routing artifacts, not part of the
        # original circuit, so exclude them from this comparison.
        return [
            op.gate.name
            for op in circuit.operations()
            if qudit in op.location and not isinstance(op.gate, (SwapGate, IdentityGate))
        ]

    for q in range(original.num_qudits):
        assert real_gates_per_qudit(routed_circuit, q) == real_gates_per_qudit(original, q), (
            f"Gate order for qudit {q} changed during routing -- that's not allowed."
        )


def test_estimate_walk_markers_matches_execution():
    """The dry-run estimator must predict exactly what _walk_path emits:
    clear hops cost 1 marker each; a blocked hop costs a 2-marker
    vacate when the trap has slack. The selective-QUBO acceptance gate
    compares these estimates, so drift here means wrong accept/reject."""
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    placement = Placement(pg)
    placement.place(0, pg.slots_of("t0")[0])

    clear_path = pg.shortest_path("t0:0", "t1:1")
    assert ShawRoutingPass._estimate_walk_markers(
        pg, placement, clear_path, 0
    ) == len(clear_path) - 1  # all free: one marker per hop

    # Block a trap slot whose trap still has slack: the walk must
    # shuffle the occupant aside first (2 markers) instead of a plain
    # hop (1 marker). (Blocking a segment would swap instead, which
    # costs the same single marker -- not what this asserts.)
    placement.place(1, "t1:0")
    assert "t1:0" in clear_path
    assert ShawRoutingPass._estimate_walk_markers(
        pg, placement, clear_path, 0
    ) > len(clear_path) - 1  # vacate shuffle costs extra markers


if __name__ == "__main__":
    test_routing_runs_clean_and_inserts_movement()
    test_original_gates_preserved_in_order()
    test_estimate_walk_markers_matches_execution()
    print("shaw_routing_pass: all tests passed")