"""
examples/demo_selective_qubo.py
===============================
Walks one congestion event through the whole selective-QUBO pipeline.

Run from project root:
    python examples/demo_selective_qubo.py

  1. QCCD chain with every trap holding its home ion.
  2. Severity metrics (kappa, rho, d) for a blocked path.
  3. Window extraction and the greedy walk's cost.
  4. QUBO formulation (warm-started from the greedy walk).
  5. Tabu search, decoding and per-sample validation.
  6. Greedy walk vs QUBO plan.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from congestion_handler import CongestionHandler, seed_from, walk_trajectories
from position_graph import Placement, build_linear_qccd
from qubo_formulator import QUBOFormulator
from qubo_solver import QUBOSolver
from solution_decoder import SolutionDecoder
from walk_planner import plan_walk, walk_cost

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main() -> None:
    section("1. Hardware and placement")
    pg = build_linear_qccd(num_traps=4, trap_capacity=2)
    pl = Placement(pg)
    for q in range(4):
        pl.place(q, f"t{q}:0")
    path = pg.shortest_path("t0:0", "t3:1")
    print(f"Positions: {pg.num_positions}, edges: {pg.num_edges}")
    print(f"Ion 0 must reach trap t3 along: {' -> '.join(path)}")

    section("2. Severity metrics")
    handler = CongestionHandler()
    blocked = handler._detect_blockages(path, pl)
    kappa = handler._compute_kappa(path, pl)
    rho = handler._compute_rho(blocked, len(path) - 1)
    depth = handler._compute_blockage_depth(blocked, pl, pg)
    print(f"Blocked positions: {blocked}")
    print(f"kappa={kappa:.3f}  rho={rho:.3f}  d={depth}  "
          f"trigger={handler.should_trigger_qubo(kappa, rho, depth)}")

    section("3. Window and greedy baseline")
    window = handler.extract_window(path, blocked, pl, pg)
    greedy_cost = walk_cost(pg, pl, path, window.mover)
    print(f"|W| = {len(window.window_nodes)} positions, ions: {window.active_ions}")
    print(f"Goal slots: {sorted(window.goal_nodes)}")
    print(f"Greedy walk ({greedy_cost} ops):")
    for action in plan_walk(pg, pl, path, window.mover):
        print(f"   {action}")

    section("4. QUBO formulation")
    traj = walk_trajectories(pg, pl, path, window)
    problem = QUBOFormulator().build(window, pg, min_horizon=len(traj[window.mover]) - 1)
    print(f"T = {problem.time_horizon}, variables = {problem.num_variables} "
          f"(+{problem.num_aux_variables} aux), couplings = {len(problem.bqm.quadratic)}, "
          f"P = {problem.penalty:.1f}")

    section("5. Solve, decode, validate")
    solution = QUBOSolver(seed=1).solve(problem, initial_state=seed_from(problem, traj))
    plan, violations = SolutionDecoder().best_plan(solution)
    print(f"Solver: {solution.solver_used} in {solution.solve_time_s * 1000:.1f} ms, "
          f"{len(solution.samples)} distinct candidates, {len(violations)} rejected {sorted(set(violations))}")

    section("6. Result")
    if plan is None or plan.cost >= greedy_cost:
        print(f"No cheaper plan found; greedy walk ({greedy_cost} ops) is used.")
        return
    print(f"QUBO plan ({plan.cost} ops vs greedy {greedy_cost}):")
    for op in plan.ops:
        print(f"   {op}")


if __name__ == "__main__":
    main()
