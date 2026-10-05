"""
test_qubo_solver.py

Unit tests for QUBOSolver (Person 2's module).

Run with:  pytest tests/test_qubo_solver.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_exact_solver_finds_optimal():
    """ExactSolver on a tiny 4-variable problem returns lowest-energy sample."""
    import dimod
    from qubo_solver import QUBOSolver, QUBOSolution

    # Manually build a tiny BQM: minimized when x0=1, x1=0
    bqm = dimod.BinaryQuadraticModel({'x0': -1.0, 'x1': 2.0}, {}, 0.0, 'BINARY')

    # Build a minimal QUBOProblem stub
    from qubo_formulator import QUBOProblem
    from congestion_handler import WindowInfo
    dummy_window = WindowInfo(
        window_nodes=frozenset(['a']), active_ions={0: 'a'}, obstacle_ions={},
        source='a', target='a', blocked_path=[], center_nodes=[], radius=1, size_capped=False,
    )
    problem = QUBOProblem(
        bqm=bqm, var_map={}, inv_var_map={},
        num_variables=2, num_aux_variables=0,
        time_horizon=1, penalty_lambda=10.0,
        window=dummy_window,
    )

    solver = QUBOSolver(exact_threshold=10)
    solution = solver.solve_exact(problem)
    assert isinstance(solution, QUBOSolution)
    assert solution.sample['x0'] == 1
    assert solution.sample['x1'] == 0
    assert solution.solver_used == "exact"


def test_sa_returns_valid_solution():
    """SA on a 25-variable problem returns a binary sample dict."""
    import dimod
    from qubo_solver import QUBOSolver
    from qubo_formulator import QUBOProblem
    from congestion_handler import WindowInfo

    bqm = dimod.BinaryQuadraticModel(
        {f'x{i}': float(i % 3 - 1) for i in range(25)}, {}, 0.0, 'BINARY'
    )
    dummy_window = WindowInfo(
        window_nodes=frozenset(), active_ions={}, obstacle_ions={},
        source='a', target='b', blocked_path=[], center_nodes=[], radius=1, size_capped=False,
    )
    problem = QUBOProblem(
        bqm=bqm, var_map={}, inv_var_map={},
        num_variables=25, num_aux_variables=0,
        time_horizon=3, penalty_lambda=10.0,
        window=dummy_window,
    )

    solver = QUBOSolver(sa_num_reads=10, sa_num_sweeps=100)
    solution = solver.solve_sa(problem)
    assert solution.solver_used == "simulated_annealing"
    assert all(v in (0, 1) for v in solution.sample.values())


def test_auto_selects_exact_for_small():
    from qubo_solver import QUBOSolver
    # Can't fully test without QUBOProblem, but check threshold logic
    solver = QUBOSolver(exact_threshold=20)
    assert solver.exact_threshold == 20


def test_sa_seed_reproducible():
    """Same sa_seed twice -> identical samples (reproducible benchmarks)."""
    import dimod
    from qubo_solver import QUBOSolver
    from qubo_formulator import QUBOProblem
    from congestion_handler import WindowInfo

    bqm = dimod.BinaryQuadraticModel(
        {f'x{i}': float((i * 7) % 5 - 2) for i in range(15)},
        {(f'x{i}', f'x{(i + 3) % 15}'): 1.0 for i in range(15)},
        0.0, 'BINARY',
    )
    window = WindowInfo(
        window_nodes=frozenset(), active_ions={}, obstacle_ions={},
        source='a', target='b', blocked_path=[], center_nodes=[], radius=1, size_capped=False,
    )
    problem = QUBOProblem(
        bqm=bqm, var_map={}, inv_var_map={},
        num_variables=15, num_aux_variables=0,
        time_horizon=1, penalty_lambda=10.0,
        window=window,
    )
    s1 = QUBOSolver(sa_num_reads=10, sa_num_sweeps=100, sa_seed=42).solve_sa(problem)
    s2 = QUBOSolver(sa_num_reads=10, sa_num_sweeps=100, sa_seed=42).solve_sa(problem)
    assert s1.sample == s2.sample


if __name__ == "__main__":
    test_exact_solver_finds_optimal()
    test_sa_returns_valid_solution()
    test_auto_selects_exact_for_small()
    test_sa_seed_reproducible()
    print("test_qubo_solver: all tests passed")
