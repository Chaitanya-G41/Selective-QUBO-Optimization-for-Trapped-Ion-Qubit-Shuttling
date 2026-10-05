"""
test_qccd_adapter.py

Two things worth proving about the adapter:

  1. ShawRoutingPass runs *unmodified* against a real qccd
     PositionGraph via the adapter -- on LinearChain here.

  2. The same circuit on a tight 2x2 Grid -- which used to deadlock
     with a clear RuntimeError -- now routes end-to-end via
     single-level eviction (a parked bystander is relocated, then the
     gate retries). The deadlock guard itself is still covered by the
     eviction-proof remainders elsewhere in the suite.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root, for `qccd`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # for qccd_adapter, shaw_routing_pass

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate

from qccd.position_graph import GridArchitecture, LinearChainArchitecture
from qccd_adapter import build_from_qccd, seed_default_placement
from shaw_routing_pass import ShawRoutingPass


def _build_test_circuit() -> Circuit:
    c = Circuit(4)
    c.append_gate(HGate(), (0,))
    c.append_gate(CNOTGate(), (0, 3))
    c.append_gate(CNOTGate(), (1, 2))
    c.append_gate(CNOTGate(), (0, 1))
    return c


async def _run(qccd_graph):
    circuit = _build_test_circuit()
    adapter, placement = build_from_qccd(qccd_graph)
    seed_default_placement(adapter, placement, circuit.num_qudits)
    routing_pass = ShawRoutingPass(position_graph=adapter, placement=placement)
    await routing_pass.run(circuit, PassData(circuit))
    return circuit


def test_shaw_runs_unmodified_on_qccd_linear_chain():
    qccd_graph = LinearChainArchitecture.build(n_traps=4, trap_capacity=2)
    routed_circuit = asyncio.run(_run(qccd_graph))

    gate_names = [op.gate.name for op in routed_circuit.operations()]
    assert "HGate" in gate_names
    assert gate_names.count("CNOTGate") == 3  # all 3 original CNOTs survived


def test_tight_grid_routes_via_eviction():
    # A small 2x2 grid (4 traps, capacity 2 each = 8 slots for 4 ions)
    # used to deadlock here: no local slack for this circuit/seeding.
    # Single-level eviction relocates a parked bystander and retries,
    # so the same circuit now routes with all original gates intact.
    qccd_graph = GridArchitecture.build(rows=2, cols=2, trap_capacity=2)
    routed_circuit = asyncio.run(_run(qccd_graph))

    gate_names = [op.gate.name for op in routed_circuit.operations()]
    assert "HGate" in gate_names
    assert gate_names.count("CNOTGate") == 3  # all 3 original CNOTs survived


if __name__ == "__main__":
    test_shaw_runs_unmodified_on_qccd_linear_chain()
    test_tight_grid_routes_via_eviction()
    print("qccd_adapter: all tests passed")