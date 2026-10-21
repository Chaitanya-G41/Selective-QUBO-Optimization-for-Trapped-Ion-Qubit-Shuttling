"""Tests for the QUBO formulation (variables, penalties, energies)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from position_graph import build_linear_qccd
from qubo_formulator import InfeasibleWindow, QUBOFormulator, WindowInfo


def _problem(**kwargs):
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    nodes = frozenset(pg.graph.nodes)
    window = WindowInfo(
        window_nodes=nodes,
        active_ions={0: "t0:0", 1: "t1:0", 2: "t2:0"},
        mover=0,
        goal_nodes=frozenset(pg.slots_of("t2")),
        source="t0:0",
        target="t2:1",
    )
    return pg, QUBOFormulator(**kwargs).build(window, pg)


def _sample(problem, traj):
    """BQM sample for trajectories (padded with stays); aux set consistently."""
    s = {v: 0 for v in problem.bqm.variables}
    T = problem.time_horizon
    for ion, path in traj.items():
        path = path + [path[-1]] * (T + 1 - len(path))
        for t, pos in enumerate(path):
            s[problem.var_map[ion, pos, t]] = 1
    return problem.complete(s)


def test_variables_are_pruned_to_reachable_states():
    pg, p = _problem()
    for (ion, pos, t) in p.var_map:
        start = p.window.active_ions[ion]
        assert len(pg.shortest_path(start, pos)) - 1 <= t
    # start pinned: exactly one variable per ion at t=0
    for ion, start in p.window.active_ions.items():
        assert [k for k in p.var_map if k[0] == ion and k[2] == 0] == [(ion, start, 0)]
    # mover can only end in the goal trap; others only on trap slots
    assert {k[1] for k in p.var_map if k[0] == 0 and k[2] == p.time_horizon} <= p.window.goal_nodes
    assert all(not k[1].startswith("seg") for k in p.var_map if k[2] == p.time_horizon)


def test_time_horizon_defaults_to_distance_plus_slack():
    _, p = _problem(slack=2)
    assert p.time_horizon == 6 + 2  # t0:0 -> t2:0 is 6 hops
    _, p = _problem(time_horizon=9)
    assert p.time_horizon == 9


def test_valid_plan_energy_equals_hop_count():
    _, p = _problem()
    traj = {
        0: ["t0:0", "t0:1", "seg0", "t1:0", "t1:1", "seg1", "t2:0"],
        1: ["t1:0", "t1:1", "t1:1", "t1:1", "t1:0"],
        2: ["t2:0", "t2:1"],
    }
    # 6 mover hops + 2 (ion 1, one of them in the swap) + 1 (ion 2) = 9,
    # plus the settle tie-break 1/(#ions+1) because ion 2 ends off its start.
    assert p.bqm.energy(_sample(p, traj)) == pytest.approx(9.0 + 1 / 4)


def test_violations_cost_at_least_one_penalty():
    _, p = _problem()
    stay = {0: ["t0:0"], 1: ["t1:0"], 2: ["t2:0"]}
    with pytest.raises(KeyError):  # mover cannot stay: no variable at T outside goal
        _sample(p, stay)
    empty = {v: 0 for v in p.bqm.variables}
    assert p.bqm.energy(empty) >= p.penalty  # one-hot offset makes "nowhere" expensive


def test_crossing_over_transport_edge_is_penalised():
    _, p = _problem()
    # ion 0 and ion 1 trade seg0 <-> t1:0 in one step (not a swap edge)
    traj = {
        0: ["t0:0", "t0:1", "seg0", "t1:0", "t1:1", "seg1", "t2:0"],
        1: ["t1:0", "t1:0", "t1:0", "seg0", "t0:1", "t0:0"],
        2: ["t2:0", "t2:1"],
    }
    s = _sample(p, traj)
    assert p.bqm.energy(s) >= p.penalty


def test_infeasible_window_raises():
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    window = WindowInfo(
        window_nodes=frozenset({"t0:0", "t0:1"}),
        active_ions={0: "t0:0"},
        mover=0,
        goal_nodes=frozenset(pg.slots_of("t2")),
        source="t0:0",
        target="t2:0",
    )
    with pytest.raises(InfeasibleWindow):
        QUBOFormulator().build(window, pg)
