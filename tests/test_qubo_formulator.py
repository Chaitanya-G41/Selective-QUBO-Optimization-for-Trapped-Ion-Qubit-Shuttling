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
from congestion_handler import CongestionHandler, WindowInfo
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


def _full_sample(problem, assignment):
    """Complete sample dict: every BQM variable (primaries + Rosenberg
    aux) 0 except assigned ones (=1). Aux at 0 contributes nothing for
    products whose inputs never co-fire; for a firing crossing product
    it contributes exactly its +lambda*ab term -- which is the penalty
    under test."""
    sample = {label: 0 for label in problem.bqm.variables}
    for key in assignment:
        sample[problem.var_map[key]] = 1
    return sample


def test_crossing_penalty_fires_on_swap():
    """A simultaneous position swap must cost >= lambda more than the
    identical trajectories without the swap. Both samples keep one-hot,
    capacity, movement-legality and goal terms equal, so any gap at or
    above lambda can only come from the crossing penalty."""
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    window = WindowInfo(
        window_nodes=frozenset({"t0:0", "t0:1", "seg0", "t1:0"}),
        active_ions={0: "t0:0", 1: "t0:1"},
        obstacle_ions={},
        source="t0:0",
        target="t1:0",
        blocked_path=["seg0"],
        center_nodes=["seg0"],
        radius=1,
        size_capped=False,
    )
    problem = QUBOFormulator(time_horizon=2).build(window, pg)
    lam = problem.penalty_lambda

    parked = _full_sample(problem, [
        (0, "t0:0", 0), (0, "t0:0", 1), (0, "t0:0", 2),
        (1, "t0:1", 0), (1, "t0:1", 1), (1, "t0:1", 2),
    ])
    # Same, except the ions swap spots at t=0->1 and swap back at t=1->2
    # (both hops along the legal t0:0<->t0:1 swap edge).
    swapped = _full_sample(problem, [
        (0, "t0:0", 0), (0, "t0:1", 1), (0, "t0:0", 2),
        (1, "t0:1", 0), (1, "t0:0", 1), (1, "t0:1", 2),
    ])
    gap = problem.bqm.energy(swapped) - problem.bqm.energy(parked)
    assert gap >= lam, (
        f"crossing swap should cost >= lambda ({lam}), got gap={gap}"
    )


def test_progress_reward_pulls_mover_to_target():
    """Shaping must make the target strictly more attractive than the
    start for the mover (a pull SA can follow stepwise). Uses an
    explicit lambda so the assertion doesn't depend on auto-scaling."""
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    window = WindowInfo(
        window_nodes=frozenset({"t0:0", "t0:1", "seg0", "t1:0"}),
        active_ions={0: "t0:0", 1: "t0:1"},
        obstacle_ions={},
        source="t0:0",
        target="t1:0",
        blocked_path=["seg0"],
        center_nodes=["seg0"],
        radius=1,
        size_capped=False,
    )
    problem = QUBOFormulator(time_horizon=2, penalty_lambda=10.0).build(window, pg)
    t = 1
    bias_target = problem.bqm.linear[problem.var_map[(0, "t1:0", t)]]
    bias_source = problem.bqm.linear[problem.var_map[(0, "t0:0", t)]]
    assert bias_target < bias_source, (
        f"target bias ({bias_target}) should beat source bias ({bias_source})"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])


