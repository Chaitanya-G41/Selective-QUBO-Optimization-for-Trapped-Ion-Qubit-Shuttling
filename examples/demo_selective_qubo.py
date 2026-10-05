"""
examples/demo_selective_qubo.py
===============================
Interactive Prototype Demonstration: Selective QUBO Optimization for Trapped-Ion QCCD Shuttling.

Run from project root:
    python examples/demo_selective_qubo.py

Demonstrates:
  1. QCCD Hardware Graph Topology & Live Ion Placement.
  2. Severe Congestion Detection (κ, ρ, d severity metrics).
  3. Selective QUBO Window Extraction W (|vars| < 100 scaling bound).
  4. BQM Formulation & Penalty Weighting.
  5. Quantum/Simulated Annealing Solve (dimod / neal).
  6. Solution Trajectory Decoding & Physical Validation (5 Checks).
  7. Side-by-Side Comparison against Heuristic Baseline.
"""

import sys
import os
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

# Ensure Unicode output formatting works on Windows consoles
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
del _stream_name, _stream

from position_graph import build_linear_qccd, Placement
from congestion_handler import CongestionHandler
from qubo_formulator import QUBOFormulator
from qubo_solver import QUBOSolver
from solution_decoder import SolutionDecoder

SEP = "=" * 70

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main():
    print(SEP)
    print("  SELECTIVE QUBO OPTIMIZATION FOR TRAPPED-ION QUBIT SHUTTLING")
    print("  Prototype Demonstration & Compiler Pipeline Walkthrough")
    print(SEP)

    # -------------------------------------------------------------------------
    # Step 1: Hardware Graph & Initial Placement
    # -------------------------------------------------------------------------
    section("1. QCCD Hardware Topology & Initial Ion Placement")
    
    pg = build_linear_qccd(num_traps=4, trap_capacity=2)
    pl = Placement(pg)

    # Place ions to induce congestion along path t0:0 -> t1:1
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])  # blocker

    print(f"Hardware Graph  : Linear QCCD (3 traps, capacity=2 each)")
    print(f"Moving Ion      : Ion 0 starting at {pl.position_of(0)}")
    print(f"Blocker Ion     : Ion 1 at {pl.position_of(1)}")
    print(f"Desired Path    : t0:0 -> t1:1")

    # -------------------------------------------------------------------------
    # Step 2: Path & Congestion Severity Metrics
    # -------------------------------------------------------------------------
    section("2. Congestion Detection & Severity Metrics (κ, ρ, d)")

    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t1")[1])
    handler = CongestionHandler(kappa_threshold=0.2, rho_threshold=0.2, depth_threshold=0, window_max_size=6)
    
    blocked = handler._detect_blockages(path, pl)
    kappa = handler._compute_kappa(path, pl)
    rho = handler._compute_rho(blocked, len(path) - 1)
    depth = handler._compute_blockage_depth(blocked, pl, pg)
    is_triggered = handler.should_trigger_qubo(kappa, rho, depth)

    print(f"Shortest Path   : {' -> '.join(path)}")
    print(f"Blocked Nodes   : {blocked}")
    print(f"Local Contention (κ) : {kappa:.3f}  (Threshold: > {handler.kappa_threshold})")
    print(f"Routing Regret   (ρ) : {rho:.3f}  (Threshold: > {handler.rho_threshold})")
    print(f"Blockage Depth   (d) : {depth}      (Threshold: > {handler.depth_threshold})")
    print(f"QUBO Triggered? : {'YES (Severe Congestion Detected!)' if is_triggered else 'NO (Use Heuristic)'}")

    # -------------------------------------------------------------------------
    # Step 3: Local Window Extraction
    # -------------------------------------------------------------------------
    section("3. Local Congestion Window Extraction (W)")

    window = handler.extract_window(path, blocked, pl, pg, radius=1)
    print(f"Extracted Window Nodes W : {sorted(list(window.window_nodes))}")
    print(f"Active Ions (in W)       : {window.active_ions}")
    print(f"Obstacle Ions (outside)  : {window.obstacle_ions}")
    print(f"Window Node Count |V_W|   : {len(window.window_nodes)}")

    # -------------------------------------------------------------------------
    # Step 4: QUBO Matrix Formulation
    # -------------------------------------------------------------------------
    section("4. QUBO Matrix Formulation & Penalty Construction")

    formulator = QUBOFormulator(time_horizon=3)
    problem = formulator.build(window, pg)

    print(f"Time Horizon (T)         : {problem.time_horizon} timesteps")
    print(f"Decision Variables x_i,v,t: {problem.num_variables}")
    print(f"Auxiliary Variables z_k  : {problem.num_aux_variables} (Rosenberg quadratization)")
    print(f"Total BQM Variables      : {len(problem.bqm.variables)} (Budget: < 100 vars)")
    print(f"Total Quadratic Terms    : {len(problem.bqm.quadratic)}")
    print(f"Penalty Multiplier (λ)   : {problem.penalty_lambda:.2f}")

    # -------------------------------------------------------------------------
    # Step 5: QUBO Solving
    # -------------------------------------------------------------------------
    section("5. QUBO Solving via Simulated Annealing (dimod/neal)")

    solver = QUBOSolver()
    t_start = time.perf_counter()
    solution = solver.solve(problem)
    t_elapsed = (time.perf_counter() - t_start) * 1000

    print(f"Solver Engine            : {solution.solver_used}")
    print(f"Ground State Energy      : {solution.energy:.4f}")
    print(f"Feasible Ground State?   : {'YES' if solution.is_feasible else 'NO'}")
    print(f"Solve Latency            : {t_elapsed:.2f} ms")

    # -------------------------------------------------------------------------
    # Step 6: Decoding & Physical Validation
    # -------------------------------------------------------------------------
    section("6. Solution Decoding & Physical Validation (5 Checks)")

    decoder = SolutionDecoder()
    decoded = decoder.decode_and_validate(solution, pl, pg)

    print(f"Decoding Status          : {'ACCEPTED' if decoded.accepted else 'REJECTED'}")
    if not decoded.accepted:
        print(f"Rejection Reason         : {decoded.violation}")
    else:
        print(f"Moving Ion Decoded Path  : {' -> '.join(decoded.decoded_path)}")
        print(f"QUBO Route Cost (C_QUBO) : {decoded.C_QUBO} hops")
        print(f"Heuristic Baseline Cost  : {decoded.C_heuristic} hops")
        print(f"Cost Reduction (%)       : {decoded.improvement * 100:.1f}% improvement")

        print("\nTrajectory Breakdown per Ion:")
        for ion, traj in decoded.trajectories.items():
            print(f"  Ion {ion}: {' -> '.join(traj)}")

    # -------------------------------------------------------------------------
    # Step 7: Architecture Performance Summary
    # -------------------------------------------------------------------------
    section("7. Selective QUBO Architecture Key Advantages")
    print("  1. Scalability : Solves subproblem in <100 vars instead of >1000 global vars.")
    print("  2. Efficiency  : Triggers QUBO ONLY when local contention is severe.")
    print("  3. Safety      : 100% physically validated; seamless fallback to greedy router.")
    print(SEP)
    print("  Prototype Demonstration Completed Successfully!")
    print(SEP)


if __name__ == "__main__":
    main()
