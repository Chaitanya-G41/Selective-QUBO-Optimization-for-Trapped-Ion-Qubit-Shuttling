"""
Tests for CongestionHandler: severity metrics, trigger logic, window
extraction, greedy fallback and the selective-QUBO acceptance rule.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from congestion_handler import (
    CongestionHandler,
    Resolution,
    seed_from,
    walk_trajectories,
)
from position_graph import Placement, build_linear_qccd
from qubo_formulator import QUBOFormulator
from solution_decoder import SolutionDecoder
from walk_planner import walk_cost


def _chain(num_traps, occupied):
    """Linear chain (capacity 2); occupied = {qudit: slot}."""
    pg = build_linear_qccd(num_traps=num_traps, trap_capacity=2)
    pl = Placement(pg)
    for q, slot in occupied.items():
        pl.place(q, slot)
    return pg, pl


def _home_chain(num_traps):
    return _chain(num_traps, {q: f"t{q}:0" for q in range(num_traps)})


# ---------------------------------------------------------------------------
# Severity metrics + trigger
# ---------------------------------------------------------------------------

def test_kappa():
    pg, pl = _home_chain(5)
    path = pg.shortest_path("t0:0", "t4:0")
    h = CongestionHandler()
    assert h._compute_kappa(path, pl) == sum(pl.is_occupied(p) for p in path) / len(path)
    pg2, pl2 = _chain(3, {})
    assert h._compute_kappa(pg2.shortest_path("t0:0", "t2:0"), pl2) == 0.0


def test_rho_formula():
    h = CongestionHandler()
    assert h._compute_rho([], 5) == 0.0
    assert h._compute_rho(["a", "b"], 4) == 1.0  # 2 * |B| / C_LB
    assert h._compute_rho(["a"], 0) == 0.0


def test_depth():
    h = CongestionHandler()
    pg, pl = _chain(3, {0: "t0:0"})
    assert h._compute_blockage_depth([], pl, pg) == 0
    pg, pl = _chain(3, {0: "t1:0", 1: "t1:1", 2: "seg0"})
    assert h._compute_blockage_depth(["t1:0"], pl, pg) >= 1


def test_trigger_logic():
    h = CongestionHandler(kappa_threshold=0.5, rho_threshold=0.3, depth_threshold=2)
    assert h.should_trigger_qubo(0.6, 0.4, 0)
    assert h.should_trigger_qubo(0.0, 0.0, 3)
    assert not h.should_trigger_qubo(0.6, 0.2, 0)
    assert not h.should_trigger_qubo(0.4, 0.9, 0)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def test_window_contents():
    pg, pl = _home_chain(3)
    path = pg.shortest_path("t0:0", "t2:1")
    h = CongestionHandler()
    w = h.extract_window(path, h._detect_blockages(path, pl), pl, pg)
    assert set(path) <= w.window_nodes
    assert w.goal_nodes == frozenset(pg.slots_of("t2"))
    assert w.mover == 0
    assert w.active_ions == {0: "t0:0", 1: "t1:0", 2: "t2:0"}


def test_window_too_large_is_skipped():
    pg, pl = _home_chain(8)
    path = pg.shortest_path("t0:0", "t7:1")
    h = CongestionHandler(window_max_nodes=10)
    assert h.extract_window(path, h._detect_blockages(path, pl), pl, pg) is None


def test_long_path_is_planned_in_chunks():
    """A path whose window exceeds the limit is solved window by window,
    and the combined plan still beats the greedy walk and replays legally."""
    pg, pl = _home_chain(6)
    path = pg.shortest_path("t0:0", "t5:1")
    h = CongestionHandler(window_max_nodes=10)
    res = h.resolve_path_congestion(path, pl, pg)

    assert res.outcome == "qubo_accepted"
    assert h.stats["chunked_plans"] == 1
    assert res.plan.cost < walk_cost(pg, pl, path, 0)
    for op in res.plan.ops:
        if op[0] == "move":
            assert pg.graph.has_edge(pl.position_of(op[1]), op[2])
            pl.move(op[1], op[2])
        else:
            pl.swap(op[1], op[2])
    assert pl.position_of(0) in pg.slots_of("t5")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_clear_path_unchanged():
    pg, pl = _chain(3, {0: "t0:0"})
    path = pg.shortest_path("t0:0", "t2:0")
    res = CongestionHandler().resolve_path_congestion(path, pl, pg)
    assert isinstance(res, Resolution)
    assert res.path == path and res.plan is None and res.outcome == "clear"


def test_reroute_when_alternative_exists():
    # 0 at t0:0 wants t0:1 ... occupied seg-free alternative within trap.
    pg, pl = _chain(2, {0: "t0:0", 1: "t1:0"})
    path = ["t0:0", "t0:1", "seg0", "t1:0", "t1:1"]
    h = CongestionHandler(qubo_enabled=False)
    assert h._find_unblocked_path("t0:0", "t1:1", pl, pg) is None
    res = h.resolve_path_congestion(path, pl, pg)
    assert res.outcome == "walk" and res.path == path


def test_no_pg_returns_path():
    pg, pl = _home_chain(3)
    path = pg.shortest_path("t0:0", "t2:1")
    res = CongestionHandler().resolve_path_congestion(path, pl, None)
    assert res.path == path and res.plan is None


def test_baseline_never_solves():
    pg, pl = _home_chain(3)
    path = pg.shortest_path("t0:0", "t2:1")
    h = CongestionHandler(qubo_enabled=False, kappa_threshold=0, rho_threshold=0)
    res = h.resolve_path_congestion(path, pl, pg)
    assert res.plan is None and h.stats["qubo_triggers"] == 0


def test_qubo_plan_beats_greedy_and_is_executable():
    pg, pl = _home_chain(3)
    path = pg.shortest_path("t0:0", "t2:1")
    greedy = walk_cost(pg, pl, path, 0)
    h = CongestionHandler(kappa_threshold=0, rho_threshold=0)
    res = h.resolve_path_congestion(path, pl, pg)

    assert res.outcome == "qubo_accepted"
    assert res.plan.cost < greedy
    assert h.stats["ops_saved"] == greedy - res.plan.cost

    for op in res.plan.ops:  # must replay legally on the live placement
        if op[0] == "move":
            assert pg.graph.has_edge(pl.position_of(op[1]), op[2])
            pl.move(op[1], op[2])
        else:
            a, b = op[1], op[2]
            assert pg.graph.edges[pl.position_of(a), pl.position_of(b)]["label"] == "swap"
            pl.swap(a, b)
    assert pl.position_of(0) in pg.slots_of("t2")
    assert pl.position_of(2) in pg.slots_of("t2")  # gate partner stays in its trap


def test_greedy_seed_is_a_valid_plan():
    """The warm start is the greedy walk itself and decodes to a valid plan."""
    pg, pl = _home_chain(4)
    path = pg.shortest_path("t0:0", "t3:1")
    h = CongestionHandler()
    w = h.extract_window(path, h._detect_blockages(path, pl), pl, pg)
    traj = walk_trajectories(pg, pl, path, w)
    problem = QUBOFormulator().build(w, pg, min_horizon=len(traj[0]) - 1)
    seed = seed_from(problem, traj)
    plan = SolutionDecoder().to_plan(seed, problem)
    assert plan.cost == walk_cost(pg, pl, path, 0)
    # Energy of a valid plan = hop count (each in-trap swap counts 2 hops).
    swaps = sum(op[0] == "swap" for op in plan.ops)
    assert problem.bqm.energy(seed) == pytest.approx(plan.cost + swaps)


def test_stats_and_summary():
    pg, pl = _home_chain(3)
    h = CongestionHandler(kappa_threshold=0, rho_threshold=0)
    h.resolve_path_congestion(pg.shortest_path("t0:0", "t0:1"), pl, pg)
    h.resolve_path_congestion(pg.shortest_path("t0:0", "t2:1"), pl, pg)
    assert h.stats["total_calls"] == 2
    assert h.stats["clear_paths"] == 1
    assert h.stats["blocked_events"] == 1
    assert "qubo_accepted=1" in h.summary()
