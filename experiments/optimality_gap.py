"""
optimality_gap.py
=================
How far is the heuristic QUBO solver from optimal on real windows?

Captures every QUBO the router solves on a sample of benchmark circuits,
then solves each one exactly as a MILP (standard linearisation of the
binary quadratic model, solved with SciPy's HiGHS) and compares:

  energy gap   heuristic energy - optimal energy
  plan gap     ops of the heuristic's best plan - ops of the optimal plan

    python experiments/optimality_gap.py --every 4 --time-limit 30
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import benchmark_runner as br  # noqa: E402
import congestion_handler as ch  # noqa: E402
from solution_decoder import InvalidSample, SolutionDecoder  # noqa: E402


def exact_minimum(bqm, time_limit: float):
    """(optimal sample, energy, proven_optimal) of a BINARY bqm via MILP."""
    labels = list(bqm.variables)
    index = {v: i for i, v in enumerate(labels)}
    quad = list(bqm.quadratic.items())
    n, m = len(labels), len(quad)

    # Variables: x (binary) then y_k = x_i * x_j (continuous in [0, 1]).
    c = np.zeros(n + m)
    for v, bias in bqm.linear.items():
        c[index[v]] = bias
    rows = lil_matrix((2 * m, n + m))
    lower, upper = [], []
    r = 0
    for k, ((u, v), bias) in enumerate(quad):
        i, j, y = index[u], index[v], n + k
        c[y] = bias
        if bias > 0:  # minimisation pushes y down: y >= x_i + x_j - 1
            rows[r, y], rows[r, i], rows[r, j] = 1, -1, -1
            lower.append(-1), upper.append(np.inf)
            r += 1
        else:         # minimisation pushes y up: y <= x_i, y <= x_j
            rows[r, y], rows[r, i] = 1, -1
            lower.append(-np.inf), upper.append(0)
            rows[r + 1, y], rows[r + 1, j] = 1, -1
            lower.append(-np.inf), upper.append(0)
            r += 2
    constraints = LinearConstraint(rows[:r].tocsr(), lower, upper)
    integrality = np.r_[np.ones(n), np.zeros(m)]
    result = milp(c, constraints=constraints, integrality=integrality,
                  bounds=Bounds(0, 1), options={"time_limit": time_limit})
    if result.x is None:
        return None, None, False
    sample = {v: int(round(result.x[index[v]])) for v in labels}
    return sample, bqm.energy(sample), result.status == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--every", type=int, default=4, help="use every k-th circuit")
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--arch", choices=br.ARCHITECTURES, default="linear")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    captured = []
    original_solve = ch.QUBOSolver.solve

    def capture(self, problem, initial_state=None):
        solution = original_solve(self, problem, initial_state)
        captured.append(solution)
        return solution

    ch.QUBOSolver.solve = capture
    runner = br.BenchmarkRunner(architecture=args.arch)
    for name, circuit in runner.load_all_circuits()[::args.every]:
        runner.run_hybrid(name, circuit)
    ch.QUBOSolver.solve = original_solve

    decoder = SolutionDecoder()
    energy_gaps, plan_gaps, proven = [], [], 0  # over proven-optimal windows only
    for solution in captured:
        problem = solution.problem
        heuristic_plan, _ = decoder.best_plan(solution)
        sample, energy, optimal = exact_minimum(problem.bqm, args.time_limit)
        if sample is None:
            continue
        if not optimal:
            continue  # time limit hit: no proven optimum to compare against
        proven += 1
        energy_gaps.append(solution.energy - energy)
        try:
            exact_plan = decoder.to_plan(sample, problem)
        except InvalidSample:
            continue
        if heuristic_plan is not None:
            plan_gaps.append(heuristic_plan.cost - exact_plan.cost)

    total = len(energy_gaps)
    print(f"windows: {len(captured)} captured, {proven} solved to proven optimality")
    print(f"heuristic reached the optimal energy: "
          f"{sum(g < 1e-6 for g in energy_gaps)}/{total}")
    if energy_gaps:
        print(f"mean energy gap: {statistics.mean(energy_gaps):.4f}")
    if plan_gaps:
        print(f"plans with the optimal op count: {sum(g <= 0 for g in plan_gaps)}/{len(plan_gaps)}, "
              f"ops missed in total: {sum(max(g, 0) for g in plan_gaps)}")


if __name__ == "__main__":
    main()
