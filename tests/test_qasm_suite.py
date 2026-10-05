"""
test_qasm_suite.py -- Saloni's QASM benchmark corpus (tests/*.qasm).

What this proves, without touching src/:

  1. Every .qasm file parses: OpenQASM 3.0 header, one qubit register,
     only h/cx ops, every qubit index in range. (Guards against a
     malformed upload silently sitting in the corpus.)
  2. verification_results.csv is exactly 1:1 with the files on disk.
  3. A curated subset routes end-to-end through ShawRoutingPass with
     every original gate preserved in per-qudit order (the same bar
     as tests/test_shaw_routing_pass.py) -- including the dense
     high_contention / repeated_interactions / large_benchmark files,
     which used to deadlock and now route via single-level eviction
     (relocate a parked bystander, retry the gate).

Deliberately NOT asserting: the CSV's PASS column (that is the
uploader's claim about their own verification setup, not about this
router), and routing all 200 files (the densest random/long-range
files can still deadlock on fully-packed boards -- that remainder is
the eviction work's benchmark set).
"""

import asyncio
import csv
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # repo root, for `qccd`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # for shaw_routing_pass

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate, IdentityGate, SwapGate

from shaw_routing_pass import ShawRoutingPass

TESTS_DIR = Path(__file__).resolve().parent

H_PAT = re.compile(r"^h q\[(\d+)\];$")
CX_PAT = re.compile(r"^cx q\[(\d+)\], q\[(\d+)\];$")
QREG_PAT = re.compile(r"^qubit\[(\d+)\] q;$")

# Circuits verified to route cleanly through the current pass
# (venv py3.10). Grow this list as routing improves.
MUST_ROUTE = [
    "basic_2qubit.qasm",
    "congestion.qasm",
    "linear_chain.qasm",
    "long_distance.qasm",
    "017_chain_4q.qasm",
    "051_star_4q.qasm",
    "170_random_4q.qasm",
    "012_chain_20q.qasm",  # scale check: 20q / 25 ops
    # Former deadlock cases, now routing via single-level eviction:
    "high_contention.qasm",
    "repeated_interactions.qasm",
    "large_benchmark.qasm",
]


def load_qasm(name: str):
    """Parse one corpus file -> (num_qudits, ops). Only h/cx supported;
    anything else is a test failure, not a skip (see test_all_qasm_parse)."""
    lines = [
        ln.strip()
        for ln in (TESTS_DIR / name).read_text().splitlines()
        if ln.strip()
    ]
    assert lines[0] == "OPENQASM 3.0;", f"{name}: bad header"
    assert lines[1] == 'include "stdgates.inc";', f"{name}: bad include"
    m = QREG_PAT.match(lines[2])
    assert m, f"{name}: bad qubit register line: {lines[2]!r}"
    num_qudits = int(m.group(1))
    ops = []
    for ln in lines[3:]:
        mh, mc = H_PAT.match(ln), CX_PAT.match(ln)
        assert mh or mc, f"{name}: unsupported statement: {ln!r}"
        if mh:
            q = int(mh.group(1))
            assert q < num_qudits, f"{name}: qubit index {q} out of range"
            ops.append(("h", q))
        else:
            a, b = int(mc.group(1)), int(mc.group(2))
            assert a < num_qudits and b < num_qudits and a != b, (
                f"{name}: bad cx: {ln!r}"
            )
            ops.append(("cx", a, b))
    return num_qudits, ops


def build_circuit(num_qudits: int, ops) -> Circuit:
    circuit = Circuit(num_qudits)
    for op in ops:
        if op[0] == "h":
            circuit.append_gate(HGate(), (op[1],))
        else:
            circuit.append_gate(CNOTGate(), (op[1], op[2]))
    return circuit


def real_gates_per_qudit(circuit: Circuit, qudit: int):
    return [
        op.gate.name
        for op in circuit.operations()
        if qudit in op.location and not isinstance(op.gate, (SwapGate, IdentityGate))
    ]


async def _route(name: str) -> Circuit:
    num_qudits, ops = load_qasm(name)
    circuit = build_circuit(num_qudits, ops)
    await ShawRoutingPass().run(circuit, PassData(circuit))
    return circuit


def test_all_qasm_parse():
    qasm_files = sorted(TESTS_DIR.glob("*.qasm"))
    assert len(qasm_files) > 0, "QASM corpus is empty -- did someone delete tests/*.qasm?"
    for f in qasm_files:
        num_qudits, ops = load_qasm(f.name)
        assert num_qudits >= 2 and ops, f"{f.name}: empty circuit"


def test_verification_csv_matches_disk():
    with open(TESTS_DIR / "verification_results.csv") as fh:
        rows = list(csv.DictReader(fh))
    csv_names = [r["Circuit"] for r in rows]
    assert len(csv_names) == len(set(csv_names)), "CSV lists a circuit twice"
    on_disk = {f.name for f in TESTS_DIR.glob("*.qasm")}
    assert set(csv_names) == on_disk, (
        f"CSV/disk mismatch: missing on disk={set(csv_names) - on_disk}, "
        f"missing in CSV={on_disk - set(csv_names)}"
    )


@pytest.mark.parametrize("name", MUST_ROUTE)
def test_qasm_routes_with_order_preserved(name):
    num_qudits, ops = load_qasm(name)
    routed = asyncio.run(_route(name))
    reference = build_circuit(num_qudits, ops)
    for q in range(num_qudits):
        assert real_gates_per_qudit(routed, q) == real_gates_per_qudit(reference, q), (
            f"{name}: gate order for qudit {q} changed during routing."
        )


if __name__ == "__main__":
    # Standalone run without pytest: same checks, plain asserts.
    test_all_qasm_parse()
    test_verification_csv_matches_disk()
    for _name in MUST_ROUTE:
        _n, _ops = load_qasm(_name)
        _routed = asyncio.run(_route(_name))
        _ref = build_circuit(_n, _ops)
        for _q in range(_n):
            assert real_gates_per_qudit(_routed, _q) == real_gates_per_qudit(_ref, _q), (
                f"{_name}: gate order for qudit {_q} changed during routing."
            )
    print("qasm_suite: all tests passed")
