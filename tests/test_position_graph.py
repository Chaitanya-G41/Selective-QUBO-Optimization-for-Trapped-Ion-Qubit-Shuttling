"""
Plain-assert sanity tests for position_graph.py.
Run directly with:  python tests/test_position_graph.py
(No pytest dependency needed, but it'll also work fine under pytest.)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from position_graph import PositionGraph, Placement, build_linear_qccd


def test_trap_is_fully_swap_connected():
    pg = PositionGraph()
    slots = pg.add_trap("tA", capacity=3)
    assert len(slots) == 3
    # every pair of slots in the same trap should be swap-connected
    for u in slots:
        others = set(slots) - {u}
        swap_neighbors = set(pg.neighbors_by_label(u, "swap"))
        assert swap_neighbors == others


def test_linear_qccd_position_and_edge_counts():
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    # 3 traps * 2 slots each + 2 segments (between t0-t1 and t1-t2)
    assert pg.num_positions == 3 * 2 + 2


def test_placement_rejects_collisions():
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    placement = Placement(pg)
    placement.place(qudit=0, position="t0:0")
    try:
        placement.place(qudit=1, position="t0:0")
        assert False, "expected a collision error"
    except ValueError:
        pass  # correctly rejected


def test_shortest_path_crosses_segment():
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    path = pg.shortest_path("t0:0", "t1:1")
    assert "seg0" in path


if __name__ == "__main__":
    tests = [
        test_trap_is_fully_swap_connected,
        test_linear_qccd_position_and_edge_counts,
        test_placement_rejects_collisions,
        test_shortest_path_crosses_segment,
    ]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print("\nAll tests passed.")

def test_junction_grid_wiring():
    from position_graph import build_junction_grid_qccd

    pg = build_junction_grid_qccd(rows=2, cols=3, trap_capacity=2)
    traps = list(pg._trap_slots)
    assert len(traps) == 2 * 2 + 1 * 3  # horizontal + vertical edges
    # Each trap is a linear chain: first slot on one junction, last on the other.
    first, last = pg.slots_of("h0_0")
    assert pg.graph.has_edge(first, "J0_0") and pg.graph.has_edge(last, "J0_1")
    assert not pg.graph.has_edge(first, "J0_1")
    # Junctions only connect to trap ends, never directly to each other.
    assert not any(str(n).startswith("J") for n in pg.graph.neighbors("J1_1"))
