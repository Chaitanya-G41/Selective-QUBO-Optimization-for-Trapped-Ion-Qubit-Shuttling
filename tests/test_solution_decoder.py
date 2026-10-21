"""Tests for SolutionDecoder: per-sample validation and plan scheduling."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_graph import build_linear_qccd
from qubo_formulator import QUBOFormulator, WindowInfo
from qubo_solver import QUBOSolution
from solution_decoder import InvalidSample, SolutionDecoder


@pytest.fixture
def problem():
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    window = WindowInfo(
        window_nodes=frozenset(pg.graph.nodes),
        active_ions={0: "t0:0", 1: "t1:0", 2: "t2:0"},
        mover=0,
        goal_nodes=frozenset(pg.slots_of("t2")),
        source="t0:0",
        target="t2:1",
    )
    return QUBOFormulator().build(window, pg)


GOOD = {
    0: ["t0:0", "t0:1", "seg0", "t1:0", "t1:1", "seg1", "t2:0"],
    1: ["t1:0", "t1:1", "t1:1", "t1:1", "t1:0"],
    2: ["t2:0", "t2:1"],
}


def _sample(problem, traj, drop=(), extra=()):
    s = {}
    for ion, path in traj.items():
        path = path + [path[-1]] * (problem.time_horizon + 1 - len(path))
        for t, pos in enumerate(path):
            if (ion, t) not in drop:
                s[problem.var_map[ion, pos, t]] = 1
    for key in extra:
        s[problem.var_map[key]] = 1
    return problem.complete(s)


def _violation(problem, sample):
    with pytest.raises(InvalidSample) as exc:
        SolutionDecoder().to_plan(sample, problem)
    return exc.value.constraint


def test_valid_plan_is_scheduled_with_swap(problem):
    plan = SolutionDecoder().to_plan(_sample(problem, GOOD), problem)
    assert ("swap", 0, 1) in plan.ops or ("swap", 1, 0) in plan.ops
    assert plan.cost == 8
    # ion 1 must step aside before the mover enters t1:0
    assert plan.ops.index(("move", 1, "t1:1")) < plan.ops.index(("move", 0, "t1:0"))


def test_missing_bit_is_one_hot_violation(problem):
    # Old decoder silently assumed "stayed"; now it is rejected.
    assert _violation(problem, _sample(problem, GOOD, drop={(1, 3)})) == "one_hot"


def test_two_bits_is_one_hot_violation(problem):
    assert _violation(problem, _sample(problem, GOOD, extra={(1, "t1:0", 2)})) == "one_hot"


def test_capacity_violation(problem):
    traj = {**GOOD, 1: ["t1:0"]}  # ion 1 never moves; mover runs into it
    assert _violation(problem, _sample(problem, traj)) == "capacity"


def test_exchange_over_transport_edge_rejected(problem):
    traj = {
        0: ["t0:0", "t0:1", "seg0", "t1:0", "t1:1", "seg1", "t2:0"],
        1: ["t1:0", "t1:0", "t1:0", "seg0", "t0:1", "t0:0"],
        2: ["t2:0", "t2:1"],
    }
    assert _violation(problem, _sample(problem, traj)) == "collision_avoidance"


def test_best_plan_picks_cheapest_valid(problem):
    good = _sample(problem, GOOD)
    bad = _sample(problem, GOOD, drop={(2, 4)})
    detour = _sample(problem, {**GOOD, 2: ["t2:0", "t2:1", "t2:0", "t2:1"]})
    sol = QUBOSolution([bad, detour, good], [0.0, 1.0, 2.0], "exact", 0.0, problem)
    plan, violations = SolutionDecoder().best_plan(sol)
    assert plan.cost == 8
    assert violations == ["one_hot"]
