"""
frontend/live_router.py
Non-invasive execution of the compiler pipeline to capture metrics,
the routed circuit, and the physical move traces for interactive simulation.
Does not alter any files in src/ or qccd/.
"""

from __future__ import annotations

import asyncio
import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate

from benchmark_runner import BenchmarkRunner, load_qasm_circuit, make_machine
from congestion_handler import CongestionHandler
from position_graph import Placement, build_linear_qccd
from shaw_routing_pass import ShawRoutingPass
from walk_planner import TIME_COSTS_US, UNIT_COSTS

TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"


def get_available_circuits() -> List[Tuple[str, str, int, int]]:
    """List all available QASM benchmark circuits with metadata (filename, family, qubits, gates)."""
    circuits = []
    from benchmark_runner import circuit_family
    for p in sorted(TESTS_DIR.glob("*.qasm")):
        try:
            c = load_qasm_circuit(p)
            fam = circuit_family(p.name)
            circuits.append((p.name, fam, c.num_qudits, c.num_operations))
        except Exception:
            pass
    return circuits


def run_live_compilation(
    circuit_name: str,
    architecture: str = "junction_grid",
    kappa_th: float = 0.0,
    rho_th: float = 0.0,
    depth_th: int = 2,
    routing_mode: str = "Full QUBO search",
    solver_engine: str = "Tabu Search (dwave-samplers)",
    tabu_effort: int = 4000,
    solver_seed: int = 0,
    objective: str = "Operations (unit count)",
    baseline_guard: bool = True,
    fast_trigger: bool = False,
) -> Dict[str, Any]:
    """
    Compile circuit with both baseline and selective QUBO, extracting full movement traces.
    """
    qasm_path = TESTS_DIR / circuit_name
    circuit = load_qasm_circuit(qasm_path)

    # Cost model
    costs = TIME_COSTS_US if "Time" in objective else UNIT_COSTS

    # Machine setup
    machine = make_machine(architecture, circuit.num_qudits)
    if machine is None:
        base_pg = build_linear_qccd(circuit.num_qudits, trap_capacity=2)
        base_pl = Placement(base_pg)
        for q in range(circuit.num_qudits):
            base_pl.place(q, base_pg.slots_of(f"t{q}")[0])
    else:
        base_pg, base_pl = machine

    # Record initial placement state for simulator
    init_phi = {q: base_pl.position_of(q) for q in range(circuit.num_qudits)}

    # Handlers
    planning_mode = "qubo" if routing_mode.startswith("Full") else "incumbent"
    solver_method = "sa" if "Simulated Annealing" in solver_engine else "tabu"

    baseline_handler = CongestionHandler(qubo_enabled=False, cost_model=costs)
    hybrid_handler = CongestionHandler(
        qubo_enabled=True,
        kappa_threshold=kappa_th,
        rho_threshold=rho_th,
        depth_threshold=int(depth_th),
        learned_trigger=fast_trigger,
        cost_model=costs,
        planning_mode=planning_mode,
        solver_options={
            "method": solver_method,
            "tabu_effort": int(tabu_effort),
        },
        seed=int(solver_seed),
    )

    # 1. Baseline Route
    pg_base, pl_base = copy.deepcopy((base_pg, base_pl))
    router_base = ShawRoutingPass(
        position_graph=pg_base,
        placement=pl_base,
        congestion_handler=baseline_handler,
        baseline_guard=False,
    )
    circuit_base = circuit.copy()
    asyncio.run(router_base.run(circuit_base, PassData(circuit_base)))

    # 2. Hybrid Route
    pg_hyb, pl_hyb = copy.deepcopy((base_pg, base_pl))
    router_hyb = ShawRoutingPass(
        position_graph=pg_hyb,
        placement=pl_hyb,
        congestion_handler=hybrid_handler,
        baseline_guard=baseline_guard,
    )
    circuit_hyb = circuit.copy()
    asyncio.run(router_hyb.run(circuit_hyb, PassData(circuit_hyb)))

    # Metrics computation
    base_shuttles = sum(1 for op in router_base.movement_trace if op[0] == "move")
    base_swaps = sum(1 for op in router_base.movement_trace if op[0] == "swap")
    base_total = base_shuttles + base_swaps

    hyb_shuttles = sum(1 for op in router_hyb.movement_trace if op[0] == "move")
    hyb_swaps = sum(1 for op in router_hyb.movement_trace if op[0] == "swap")
    hyb_total = hyb_shuttles + hyb_swaps

    gain_pct = ((base_total - hyb_total) / base_total * 100) if base_total else 0.0

    stats = hybrid_handler.stats

    return {
        "circuit_name": circuit_name,
        "num_qudits": circuit.num_qudits,
        "num_gates": circuit.num_operations,
        "base_total_ops": base_total,
        "base_shuttles": base_shuttles,
        "base_swaps": base_swaps,
        "base_time_us": router_base.movement_time_us,
        "base_depth": circuit_base.depth,
        "base_trace": router_base.movement_trace,
        "hyb_total_ops": hyb_total,
        "hyb_shuttles": hyb_shuttles,
        "hyb_swaps": hyb_swaps,
        "hyb_time_us": router_hyb.movement_time_us,
        "hyb_depth": circuit_hyb.depth,
        "hyb_trace": router_hyb.movement_trace,
        "gain_pct": gain_pct,
        "qubo_triggers": stats.get("qubo_triggers", 0),
        "qubo_accepted": stats.get("qubo_executed", 0),
        "blocked_events": stats.get("blocked_events", 0),
        "selected_route": router_hyb.portfolio_stats.get("selected", "hybrid"),
        "initial_phi": init_phi,
        "pg": base_pg,
    }
