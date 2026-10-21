"""Tests for the greedy walk planner, its cost models and BoardView."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bqskit.ir.circuit import Circuit

from position_graph import Placement, build_linear_qccd
from shaw_routing_pass import ShawRoutingPass
from walk_planner import (
    TIME_COSTS_US, UNIT_COSTS, BoardView, ops_cost, walk_cost, walk_ops,
)


def _home_chain(n):
    pg = build_linear_qccd(num_traps=n, trap_capacity=2)
    pl = Placement(pg)
    for q in range(n):
        pl.place(q, f"t{q}:0")
    return pg, pl


def test_walk_ops_replay_matches_real_walk():
    pg, pl = _home_chain(3)
    path = pg.shortest_path("t0:0", "t2:1")
    ops = walk_ops(pg, pl, path, 0)
    board = BoardView(pg, pl)
    board.apply(ops)

    ShawRoutingPass()._walk_path(Circuit(3), pg, pl, path, 0)
    assert all(board.position_of(q) == pl.position_of(q) for q in range(3))


def test_unit_cost_is_operation_count():
    pg, pl = _home_chain(3)
    path = pg.shortest_path("t0:0", "t2:1")
    ops = walk_ops(pg, pl, path, 0)
    assert walk_cost(pg, pl, path, 0) == len(ops) == ops_cost(pg, pl, ops, UNIT_COSTS)


def test_time_cost_prices_each_primitive():
    pg, pl = _home_chain(2)
    ops = [("move", 0, "t0:1"), ("move", 0, "seg0")]  # in-trap move, then split
    assert ops_cost(pg, pl, ops, TIME_COSTS_US) == TIME_COSTS_US["swap"] + TIME_COSTS_US["merge_split"]
    assert ops_cost(pg, pl, [("swap", 0, 1)], TIME_COSTS_US) == TIME_COSTS_US["ion_swap"]
