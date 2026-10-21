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
     have actually inserted movement markers (IdentityGate)
     -- if it didn't, the "no routing needed" fast path fired when it
     shouldn't have.
  4. Movement markers must not change what the circuit computes.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate, IdentityGate, RZGate

import pytest

from position_graph import Placement, build_linear_qccd
from congestion_handler import CongestionHandler
from shaw_routing_pass import ShawRoutingPass
from walk_planner import walk_cost


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
        if isinstance(op.gate, IdentityGate)
    ]
    assert movement_ops, (
        "Expected at least one IdentityGate movement marker -- "
        "this test circuit was built so every 2-qudit gate needs routing."
    )


def test_original_gates_preserved_in_order():
    routed_circuit = asyncio.run(_run_pass())
    original = _build_test_circuit()

    def real_gates_per_qudit(circuit: Circuit, qudit: int):
        # IdentityGate markers are routing artifacts, not part of the
        # original circuit, so exclude them from this comparison.
        return [
            op.gate.name
            for op in circuit.operations()
            if qudit in op.location and not isinstance(op.gate, IdentityGate)
        ]

    for q in range(original.num_qudits):
        assert real_gates_per_qudit(routed_circuit, q) == real_gates_per_qudit(original, q), (
            f"Gate order for qudit {q} changed during routing -- that's not allowed."
        )


def _count_markers(circuit):
    return sum(isinstance(op.gate, IdentityGate) for op in circuit.operations())


def test_walk_cost_matches_execution():
    """walk_cost must predict exactly the markers _walk_path emits."""
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    placement = Placement(pg)
    for q in range(3):
        placement.place(q, f"t{q}:0")
    path = pg.shortest_path("t0:0", "t2:1")
    expected = walk_cost(pg, placement, path, 0)

    routed = Circuit(3)
    ShawRoutingPass()._walk_path(routed, pg, placement, path, 0)
    assert _count_markers(routed) == expected == 9
    assert placement.position_of(0) == "t2:1"


@pytest.mark.parametrize("architecture", ["linear", "junction_grid", "grid"])
@pytest.mark.parametrize("qubo_enabled", [False, True])
def test_routing_preserves_unitary(qubo_enabled, architecture):
    """Ion swaps are physical: the routed circuit must equal the input."""
    from benchmark_runner import make_machine

    c = Circuit(5)
    c.append_gate(HGate(), (0,))
    c.append_gate(RZGate(), (2,), [0.7])
    for a, b in [(0, 4), (1, 3), (0, 2), (4, 1), (2, 3)]:
        c.append_gate(CNOTGate(), (a, b))
    original = c.get_unitary()

    handler = CongestionHandler(qubo_enabled=qubo_enabled, kappa_threshold=0, rho_threshold=0)
    pg, placement = make_machine(architecture, c.num_qudits) or (None, None)
    routing = ShawRoutingPass(position_graph=pg, placement=placement, congestion_handler=handler)
    asyncio.run(routing.run(c, PassData(c)))
    assert _count_markers(c) > 0
    assert original.get_distance_from(c.get_unitary()) < 1e-8
