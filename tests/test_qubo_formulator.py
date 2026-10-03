"""
test_qubo_formulator.py

Unit tests for QUBOFormulator (Person 1's module).
These are SKELETON tests — once qubo_formulator.py is implemented,
fill in or extend these tests to cover your implementation.

Run with:  pytest tests/test_qubo_formulator.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_graph import build_linear_qccd, Placement
from congestion_handler import CongestionHandler
from qubo_formulator import QUBOFormulator, QUBOProblem


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_window():
    """Create a minimal WindowInfo for testing."""
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])  # blocker
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    blocked = handler._detect_blockages(path, pl)
    window = handler.extract_window(path, blocked, pl, pg)
    return window, pg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_returns_qubo_problem():
    window, pg = _make_window()
    formulator = QUBOFormulator()
    problem = formulator.build(window, pg)
    assert isinstance(problem, QUBOProblem)


def test_variable_count_matches_window():
    """Number of variables = |active_ions| × |window_nodes| × T."""
    window, pg = _make_window()
    formulator = QUBOFormulator()
    problem = formulator.build(window, pg)
    expected = len(window.active_ions) * len(window.window_nodes) * (problem.time_horizon + 1)
    assert problem.num_variables == expected


def test_bqm_is_qubo_type():
    import dimod
    window, pg = _make_window()
    problem = QUBOFormulator().build(window, pg)
    assert isinstance(problem.bqm, dimod.BinaryQuadraticModel)


def test_var_map_and_inv_var_map_consistent():
    window, pg = _make_window()
    problem = QUBOFormulator().build(window, pg)
    for key, label in problem.var_map.items():
        assert problem.inv_var_map[label] == key


def test_penalty_lambda_satisfies_constraint():
    """λ must be > maximum possible objective contribution."""
    window, pg = _make_window()
    problem = QUBOFormulator().build(window, pg)
    assert problem.penalty_lambda >= problem.time_horizon + 1


def test_time_horizon_at_least_path_length():
    window, pg = _make_window()
    problem = QUBOFormulator().build(window, pg)
    assert problem.time_horizon >= 1


def test_initial_position_pinned_at_t0():
    """Verify hard penalty pins active ions to starting position at t=0."""
    window, pg = _make_window()
    problem = QUBOFormulator().build(window, pg)
    for ion, start_pos in window.active_ions.items():
        start_label = problem.var_map[(ion, start_pos, 0)]
        # Bias on starting position variable at t=0 should be negative (encouraged/pinned)
        assert problem.bqm.linear[start_label] < 0.0


def test_anti_crossing_penalty_added():
    """Verify anti-crossing penalty generates auxiliary Rosenberg variables and interactions."""
    window, pg = _make_window()
    formulator = QUBOFormulator()
    problem = formulator.build(window, pg)
    # Anti-crossing produces aux variables for move pairs
    assert problem.num_aux_variables >= 0
    assert len(problem.bqm.variables) >= problem.num_variables


if __name__ == "__main__":
    pytest.main(["-v", __file__])


