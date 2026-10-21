"""Tests for QUBOSolver."""

import importlib.util
import sys
from pathlib import Path

import dimod
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qubo_formulator import QUBOProblem, WindowInfo
from qubo_solver import QUBOSolver


def _problem(bqm):
    window = WindowInfo(frozenset(), {}, None, frozenset(), None, None)
    return QUBOProblem(bqm=bqm, var_map={}, inv_var_map={}, time_horizon=1,
                       penalty=10.0, window=window, graph=None)


SMALL = dimod.BinaryQuadraticModel({"a": -3.0, "b": 2.0, "c": 1.0}, {("a", "b"): 1.0}, 0.0, "BINARY")
LARGE = dimod.BinaryQuadraticModel({f"x{i}": float(i % 3 - 1) for i in range(25)}, {}, 0.0, "BINARY")


def test_exact_for_small_problems():
    sol = QUBOSolver().solve(_problem(SMALL))
    assert sol.solver_used == "exact"
    assert sol.sample == {"a": 1, "b": 0, "c": 0}
    assert sol.energy == -3.0
    assert sol.energies == sorted(sol.energies)
    assert len({tuple(sorted(s.items())) for s in sol.samples}) == len(sol.samples)


@pytest.mark.parametrize("method", ["tabu", "sa"])
def test_heuristics_find_separable_optimum(method):
    sol = QUBOSolver(method=method, seed=7).solve(_problem(LARGE))
    assert sol.solver_used == method
    assert sol.energy == LARGE.energy({f"x{i}": int(i % 3 == 0) for i in range(25)})


def test_warm_start_is_accepted():
    start = {f"x{i}": 0 for i in range(25)}
    for method in ("tabu", "sa"):
        sol = QUBOSolver(method=method, seed=1).solve(_problem(LARGE), initial_state=start)
        assert sol.energy <= 0.0


def test_seed_reproducible():
    a = QUBOSolver(method="sa", seed=3).solve(_problem(LARGE))
    b = QUBOSolver(method="sa", seed=3).solve(_problem(LARGE))
    assert a.samples == b.samples


@pytest.mark.skipif(importlib.util.find_spec("dwave.system") is not None,
                    reason="dwave-system installed; a real Leap call would need a token")
def test_leap_explains_missing_setup():
    with pytest.raises(RuntimeError, match="dwave-system"):
        QUBOSolver(method="leap").solve(_problem(LARGE))


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        QUBOSolver(method="quantum")
