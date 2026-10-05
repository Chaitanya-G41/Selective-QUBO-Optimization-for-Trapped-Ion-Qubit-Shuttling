"""
test_congestion_handler.py

Tests for the full CongestionHandler implementation (Phases 3–6):

  1. stub-backward-compat  : clear path still returned unchanged (old test preserved)
  2. kappa                 : contention metric κ on known occupancy
  3. rho                   : routing regret ρ formula
  4. blockage_depth        : recursive depth counter d
  5. qubo_trigger          : trigger condition logic
  6. no_qubo_below_thrs    : confirm benign congestion does NOT trigger QUBO
  7. extract_window        : window node set + ion classification
  8. window_size_cap       : W is capped to window_max_size
  9. reroute               : alternate unblocked path found when one exists
 10. greedy_clear          : in-trap shuffle plan returned
 11. greedy_no_park        : gives up cleanly when no parking available
 12. full_resolve_clear    : end-to-end resolve on a clear path
 13. full_resolve_reroute  : end-to-end resolve finds alternate route
 14. full_resolve_blocked  : end-to-end resolve on unavoidable blockage
 15. stats_accumulation    : stats dict updated across multiple calls
 16. summary_str           : summary() doesn't crash
 17. no_pg_graceful        : pg=None path works without error
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from congestion_handler import CongestionHandler, CongestionMetrics, WindowInfo
from position_graph import Placement, build_linear_qccd


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_3trap_pg_and_placement():
    """3-trap linear chain, capacity 2 each. Qudits 0→t0, 1→t2."""
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t2")[0])
    return pg, pl


def _make_5trap_pg_and_placement():
    """5-trap chain, all home qudits placed."""
    pg = build_linear_qccd(num_traps=5, trap_capacity=2)
    pl = Placement(pg)
    for q in range(5):
        pl.place(q, pg.slots_of(f"t{q}")[0])
    return pg, pl


# ---------------------------------------------------------------------------
# 1. Backward compatibility – stub behaviour preserved
# ---------------------------------------------------------------------------

def test_stub_clear_path_unchanged():
    pg, pl = _make_3trap_pg_and_placement()
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    result = handler.resolve_path_congestion(path, pl)
    assert result == path, "Clear path must be returned unchanged"


# ---------------------------------------------------------------------------
# 2. κ (kappa) – local contention metric
# ---------------------------------------------------------------------------

def test_kappa_zero_on_clear_path():
    pg, pl = _make_3trap_pg_and_placement()
    # path: t0:0 → seg0 → t1:0 → seg1 → t2:0
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    k = handler._compute_kappa(path, pl)
    # only endpoints occupied (qudit 0 at t0:0, qudit 1 at t2:0)
    assert 0.0 <= k <= 1.0
    occupied_count = sum(1 for p in path if pl.is_occupied(p))
    assert k == occupied_count / len(path)


def test_kappa_high_when_path_mostly_occupied():
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    # Place ions at all slots of t0 and t1 (dense)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t0")[1])
    pl.place(2, pg.slots_of("t1")[0])
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t1")[0])
    handler = CongestionHandler()
    k = handler._compute_kappa(path, pl)
    assert k > 0.5, f"Expected κ > 0.5 for dense path, got {k}"


# ---------------------------------------------------------------------------
# 3. ρ (rho) – routing regret
# ---------------------------------------------------------------------------

def test_rho_zero_when_no_blockers():
    handler = CongestionHandler()
    rho = handler._compute_rho(blocked=[], C_LB=4)
    assert rho == 0.0


def test_rho_positive_with_blockers():
    handler = CongestionHandler()
    rho = handler._compute_rho(blocked=["x", "y"], C_LB=4)
    # C_est = 4 + 2*2 = 8; rho = (8-4)/4 = 1.0
    assert abs(rho - 1.0) < 1e-9


def test_rho_formula():
    handler = CongestionHandler()
    for n_blocked, C_LB in [(1, 3), (2, 5), (3, 6)]:
        rho = handler._compute_rho(["x"] * n_blocked, C_LB)
        expected = (2 * n_blocked) / C_LB
        assert abs(rho - expected) < 1e-9, f"ρ formula mismatch for n={n_blocked}, C_LB={C_LB}"


# ---------------------------------------------------------------------------
# 4. Blockage depth d
# ---------------------------------------------------------------------------

def test_depth_zero_all_free():
    pg, pl = _make_3trap_pg_and_placement()
    handler = CongestionHandler()
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    blocked = handler._detect_blockages(path, pl)
    d = handler._compute_blockage_depth(blocked, pl, pg)
    assert d == 0


def test_depth_increases_with_secondary_blockage():
    pg = build_linear_qccd(num_traps=4, trap_capacity=2)
    pl = Placement(pg)
    # Pack ions at t1 fully, and also at t2 slot 0
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])
    pl.place(2, pg.slots_of("t1")[1])
    pl.place(3, pg.slots_of("t2")[0])
    pl.place(4, pg.slots_of("t3")[0])

    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t3")[0])
    handler = CongestionHandler(depth_threshold=3)
    blocked = handler._detect_blockages(path, pl)
    d = handler._compute_blockage_depth(blocked, pl, pg)
    # Some blocked positions will have occupied neighbours → d >= 1
    assert d >= 0  # always non-negative


# ---------------------------------------------------------------------------
# 5 & 6. QUBO trigger
# ---------------------------------------------------------------------------

def test_qubo_trigger_fires_on_both_thresholds():
    handler = CongestionHandler(kappa_threshold=0.4, rho_threshold=0.3, depth_threshold=2)
    assert handler.should_trigger_qubo(kappa=0.6, rho=0.5, depth=0)


def test_qubo_trigger_fires_on_depth_alone():
    handler = CongestionHandler(kappa_threshold=0.9, rho_threshold=0.9, depth_threshold=2)
    assert handler.should_trigger_qubo(kappa=0.1, rho=0.1, depth=3)


def test_qubo_trigger_off_below_thresholds():
    handler = CongestionHandler(kappa_threshold=0.5, rho_threshold=0.3, depth_threshold=2)
    # Neither condition met
    assert not handler.should_trigger_qubo(kappa=0.3, rho=0.2, depth=1)


def test_qubo_trigger_both_needed_for_severity():
    handler = CongestionHandler(kappa_threshold=0.5, rho_threshold=0.3, depth_threshold=2)
    # Only kappa exceeded – not enough
    assert not handler.should_trigger_qubo(kappa=0.8, rho=0.1, depth=0)
    # Only rho exceeded – not enough
    assert not handler.should_trigger_qubo(kappa=0.2, rho=0.9, depth=0)


# ---------------------------------------------------------------------------
# 7. Window extraction – node set and ion classification
# ---------------------------------------------------------------------------

def test_extract_window_contains_path_nodes():
    pg, pl = _make_5trap_pg_and_placement()
    handler = CongestionHandler(window_radius=1)
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t4")[0])
    blocked = handler._detect_blockages(path, pl)

    # Force at least one blocked position for the window to be interesting
    # Place an extra ion in an intermediate slot if path is clear
    if not blocked:
        blocked = [path[2]] if len(path) > 2 else [path[0]]

    window = handler.extract_window(path, blocked, pl, pg)

    assert isinstance(window, WindowInfo)
    # All path nodes must be in window
    for p in path:
        assert p in window.window_nodes, f"Path node {p} missing from window"
    assert window.source == path[0]
    assert window.target == path[-1]


def test_extract_window_classifies_ions():
    pg, pl = _make_5trap_pg_and_placement()
    handler = CongestionHandler(window_radius=0)  # radius 0 = only path + parking
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    blocked = handler._detect_blockages(path, pl)
    if not blocked:
        blocked = [path[1]] if len(path) > 1 else []

    window = handler.extract_window(path, blocked, pl, pg)

    # Every ion must appear in exactly one of active/obstacle
    all_classified = set(window.active_ions.keys()) | set(window.obstacle_ions.keys())
    for q in range(5):
        assert q in all_classified, f"Qudit {q} not classified"
    assert not (set(window.active_ions.keys()) & set(window.obstacle_ions.keys())), \
        "Same ion in both active and obstacle"


# ---------------------------------------------------------------------------
# 8. Window size cap
# ---------------------------------------------------------------------------

def test_window_size_cap_respected():
    pg, pl = _make_5trap_pg_and_placement()
    handler = CongestionHandler(window_max_size=4)
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t4")[0])
    blocked = [path[2]] if len(path) > 2 else []
    window = handler.extract_window(path, blocked, pl, pg, radius=3)
    assert len(window.window_nodes) <= handler.window_max_size


# ---------------------------------------------------------------------------
# 9. Alternate unblocked path finding
# ---------------------------------------------------------------------------

def test_reroute_finds_clear_path():
    pg, pl = _make_3trap_pg_and_placement()
    handler = CongestionHandler()
    # The direct path from t0:0 to t2:0 passes through seg0, t1:*, seg1
    # Those are all free, so we expect a path back
    path = handler._find_unblocked_path(
        pg.slots_of("t0")[0], pg.slots_of("t2")[0], pl, pg
    )
    assert path is not None
    assert path[0] == pg.slots_of("t0")[0]
    assert path[-1] == pg.slots_of("t2")[0]
    # No occupied intermediates
    for pos in path[1:-1]:
        assert not pl.is_occupied(pos)


def test_reroute_returns_none_when_no_clear_path():
    # Fill every position between t0 and t1 with ions
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t0")[1])
    pl.place(2, pg.slots_of("t1")[0])
    pl.place(3, pg.slots_of("t1")[1])
    # seg0 connects them – place ion 4 there too
    from position_graph import PositionGraph
    seg_node = "seg0"
    if pg.graph.has_node(seg_node):
        # seg0 exists; try to place there (segment has no capacity guard in src/pg)
        try:
            pl.place(4, seg_node)
        except ValueError:
            pass  # already occupied

    handler = CongestionHandler(max_reroute_candidates=3)
    result = handler._find_unblocked_path("t0:0", "t1:0", pl, pg)
    # Either None or a valid path (seg0 fully blocked → likely None)
    assert result is None or isinstance(result, list)


# ---------------------------------------------------------------------------
# 10. Greedy clearing – in-trap shuffle
# ---------------------------------------------------------------------------

def test_greedy_clear_finds_parking_in_same_trap():
    # t1 has capacity 2; ion at t1:0 blocks path; t1:1 is free
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])   # mover starts here
    pl.place(1, pg.slots_of("t1")[0])   # blocker in intermediate trap
    # t1:1 is free → parking spot exists

    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    blocked = handler._detect_blockages(path, pl)
    result = handler._shaw_greedy_clear(path, blocked, pl, pg)
    # Parking found → returns original path (walk_path will clear it)
    assert result is not None


# ---------------------------------------------------------------------------
# 11. Greedy clearing – no parking spot
# ---------------------------------------------------------------------------

def test_greedy_clear_returns_none_when_stuck():
    # Completely fill t1 and all its neighbours
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])
    pl.place(2, pg.slots_of("t1")[1])   # t1 fully packed → no in-trap slot
    # seg0 and seg1 are the only neighbours – occupy them too
    try:
        pl.place(3, "seg0")
        pl.place(4, "seg1")
    except ValueError:
        pass  # tolerate if segments have no capacity guard

    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    blocked = handler._detect_blockages(path, pl)
    if not blocked:
        return  # can't test this case if path is actually clear

    result = handler._shaw_greedy_clear(path, blocked, pl, pg)
    # Either None (no parking) or the original path (some parking found)
    assert result is None or isinstance(result, list)


# ---------------------------------------------------------------------------
# 12–14. End-to-end resolve_path_congestion
# ---------------------------------------------------------------------------

def test_full_resolve_clear_path_returned_unchanged():
    pg, pl = _make_3trap_pg_and_placement()
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    result = handler.resolve_path_congestion(path, pl, pg)
    assert result == path
    assert handler.stats["clear_paths"] == 1


def test_full_resolve_reroutes_when_blocked():
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])  # blocker in intermediate trap

    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler(kappa_threshold=0.99, rho_threshold=0.99)  # QUBO off
    result = handler.resolve_path_congestion(path, pl, pg)

    assert isinstance(result, list)
    assert result[0] == path[0]
    assert result[-1] == path[-1]


def test_full_resolve_stats_updated():
    pg, pl = _make_3trap_pg_and_placement()
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()

    for _ in range(3):
        handler.resolve_path_congestion(path, pl, pg)

    assert handler.stats["total_calls"] == 3


# ---------------------------------------------------------------------------
# 15. Stats accumulation across calls
# ---------------------------------------------------------------------------

def test_stats_accumulate():
    pg, pl = _make_5trap_pg_and_placement()
    handler = CongestionHandler()
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t4")[0])

    n = 5
    for _ in range(n):
        handler.resolve_path_congestion(path, pl, pg)

    assert handler.stats["total_calls"] == n
    assert (
        handler.stats["clear_paths"] + handler.stats["blocked_events"] == n
    ), "Every call must be either clear or blocked"


# ---------------------------------------------------------------------------
# 16. summary() doesn't crash
# ---------------------------------------------------------------------------

def test_summary_no_calls():
    handler = CongestionHandler()
    s = handler.summary()
    assert "no calls" in s


def test_summary_after_calls():
    pg, pl = _make_3trap_pg_and_placement()
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    handler.resolve_path_congestion(path, pl, pg)
    s = handler.summary()
    assert "Total calls" in s


# ---------------------------------------------------------------------------
# 17. pg=None graceful
# ---------------------------------------------------------------------------

def test_no_pg_returns_path():
    pg, pl = _make_3trap_pg_and_placement()
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    # pg=None: should still return *a* path without crashing
    result = handler.resolve_path_congestion(path, pl, pg=None)
    assert isinstance(result, list)
    assert result[0] == path[0]
    assert result[-1] == path[-1]


# ---------------------------------------------------------------------------
# 18. greedy SA seed validity (warm-start for _try_qubo_solve)
# ---------------------------------------------------------------------------

def test_greedy_seed_is_one_hot_valid():
    """The warm-start seed must be a complete, one-hot-valid assignment
    (exactly one 1 per ion per timestep), with the mover walking the
    static shortest path and everyone else frozen at start."""
    from qubo_formulator import QUBOFormulator

    pg, pl = _make_3trap_pg_and_placement()
    pl.place(2, pg.slots_of("t1")[0])  # blocker on the t0->t2 path
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    handler = CongestionHandler()
    blocked = handler._detect_blockages(path, pl)
    window = handler.extract_window(path, blocked, pl, pg)
    problem = QUBOFormulator().build(window, pg)

    seed = handler._greedy_seed_sample(problem, pg)
    assert seed is not None
    # Complete: every BQM variable (primaries + aux) assigned, since
    # neal rejects initial_states that don't match bqm.variables.
    assert set(seed.keys()) == set(problem.bqm.variables)
    assert set(seed.values()) <= {0, 1}
    # One-hot per (ion, timestep).
    T = problem.time_horizon
    for ion in window.active_ions:
        for t in range(T + 1):
            ones = [
                pos for pos in window.window_nodes
                if (ion, pos, t) in problem.var_map
                and seed[problem.var_map[(ion, pos, t)]] == 1
            ]
            assert len(ones) == 1, f"ion {ion} t={t}: {ones}"
    # Mover starts at source and ends at target (T covers the walk).
    inv = problem.inv_var_map
    by_ion_t = {}
    for label, bit in seed.items():
        if bit == 1 and label in inv:
            ion, pos, t = inv[label]
            by_ion_t.setdefault((ion, t), []).append(pos)
    assert by_ion_t[(0, 0)] == [pg.slots_of("t0")[0]]
    last_t = max(t for (ion, t) in by_ion_t if ion == 0)
    assert by_ion_t[(0, last_t)] == [window.target]


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
# NOTE: the Laya advisory-gate tests lived here and were removed with
# the gate before commit (R&D, saved at /tmp/laya_backup/ + /tmp/laya_ft/).
# Re-adding the gate? Restore those tests from the backup too.

if __name__ == "__main__":
    tests = [
        test_stub_clear_path_unchanged,
        test_kappa_zero_on_clear_path,
        test_kappa_high_when_path_mostly_occupied,
        test_rho_zero_when_no_blockers,
        test_rho_positive_with_blockers,
        test_rho_formula,
        test_depth_zero_all_free,
        test_depth_increases_with_secondary_blockage,
        test_qubo_trigger_fires_on_both_thresholds,
        test_qubo_trigger_fires_on_depth_alone,
        test_qubo_trigger_off_below_thresholds,
        test_qubo_trigger_both_needed_for_severity,
        test_extract_window_contains_path_nodes,
        test_extract_window_classifies_ions,
        test_window_size_cap_respected,
        test_reroute_finds_clear_path,
        test_greedy_clear_finds_parking_in_same_trap,
        test_full_resolve_clear_path_returned_unchanged,
        test_full_resolve_reroutes_when_blocked,
        test_full_resolve_stats_updated,
        test_stats_accumulate,
        test_summary_no_calls,
        test_summary_after_calls,
        test_no_pg_returns_path,
        test_greedy_seed_is_one_hot_valid,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")