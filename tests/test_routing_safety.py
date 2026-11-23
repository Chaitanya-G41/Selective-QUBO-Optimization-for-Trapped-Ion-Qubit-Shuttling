"""Physical routing and whole-circuit nonregression checks."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pytest
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, ToffoliGate
from position_graph import Placement, PositionGraph, build_linear_qccd
from congestion_handler import CongestionHandler, layout_cost
from shaw_routing_pass import ShawRoutingPass
from walk_planner import BoardView, walk_ops, UNIT_COSTS


def test_transport_blocker_is_pushed_into_vacancy():
    pg = build_linear_qccd(2, 2)
    board = Placement(pg)
    board.place(0, 't0:1')
    board.place(1, 'seg0')
    ops = walk_ops(pg, board, ['t0:1', 'seg0', 't1:0'], 0)
    assert all(op[0] == 'move' for op in ops)
    scratch = BoardView(pg, board)
    scratch.apply(ops)
    assert scratch.position_of(0) == 't1:0'
    assert board.position_of(0) == 't0:1'


def test_saturated_corridor_retreats_to_legal_swap():
    pg = build_linear_qccd(2, 2)
    board = Placement(pg)
    for ion, pos in enumerate(['seg0', 't1:0', 't1:1']):
        board.place(ion, pos)
    ops = walk_ops(pg, board, ['seg0', 't1:0'], 0)
    assert any(op[0] == 'swap' for op in ops)
    for kind, a, b in ops:
        if kind == 'move':
            board.move(a, b)
        else:
            board.swap(a, b)
    assert board.position_of(0) == 't1:0'


def test_native_and_scratch_reject_illegal_swaps_and_teleports():
    pg = build_linear_qccd(2, 2)
    board = Placement(pg)
    board.place(0, 't0:0')
    board.place(1, 't1:0')
    with pytest.raises(ValueError):
        board.swap(0, 1)
    with pytest.raises(ValueError):
        board.move(0, 't1:1')
    with pytest.raises(ValueError):
        BoardView(pg, board).apply([('swap', 0, 1)])
    assert board.position_of(0) == 't0:0'


def test_depth_reaches_cap_on_occupied_chain():
    pg = PositionGraph()
    pg.add_trap('t', 5)
    pg.graph.remove_edges_from(list(pg.graph.edges))
    board = Placement(pg)
    for i in range(5):
        board.place(i, f't:{i}')
        if i:
            pg.connect(f't:{i-1}', f't:{i}', 'swap')
    assert CongestionHandler()._compute_blockage_depth(['t:0'], board, pg) == 3


def test_partner_layout_uses_both_final_positions():
    pg = build_linear_qccd(2, 2)
    board = Placement(pg)
    board.place(0, 't0:0')
    board.place(1, 't0:1')
    stale = {0: 't1:0', 1: 't1:1'}
    assert layout_cost(pg, board, stale, {0: 1, 1: 0}) == 2
    assert layout_cost(pg, board, stale) > 2


def test_guard_replays_selected_trace_on_external_placement():
    pg = build_linear_qccd(4, 2)
    board = Placement(pg)
    for q in range(4):
        board.place(q, f't{q}:0')
    initial = BoardView(pg, board)
    circuit = Circuit(4)
    for a, b in [(0, 3), (1, 2), (0, 1), (2, 3)]:
        circuit.append_gate(CNOTGate(), (a, b))
    routing = ShawRoutingPass(pg, board)
    asyncio.run(routing.run(circuit, PassData(circuit)))
    costs = routing.portfolio_stats['candidate_costs']
    assert routing.routing_cost <= costs['baseline']
    initial.apply(routing.movement_trace)
    assert all(initial.position_of(q) == board.position_of(q) for q in range(4))
    assert routing.routing_cost == len(routing.movement_trace)


def test_rejects_undecomposed_three_qudit_gate():
    circuit = Circuit(3)
    circuit.append_gate(ToffoliGate(), (0, 1, 2))
    with pytest.raises(ValueError, match='Decompose'):
        asyncio.run(ShawRoutingPass().run(circuit, PassData(circuit)))


def test_variable_budget_skips_solver():
    pg = build_linear_qccd(3, 2)
    board = Placement(pg)
    for q in range(3):
        board.place(q, f't{q}:0')
    handler = CongestionHandler(max_bqm_variables=1)
    result = handler.resolve_path_congestion(pg.shortest_path('t0:0', 't2:1'), board, pg)
    assert result.plan is None
    assert handler.stats['qubo_skips']['variable_budget'] == 1
    assert handler.stats['qubo_solved'] == 0


def test_incumbent_mode_never_builds_or_solves_qubo(monkeypatch):
    from qubo_formulator import QUBOFormulator
    def forbidden(*args, **kwargs):
        raise AssertionError('Fast mode must not build a BQM')
    monkeypatch.setattr(QUBOFormulator, 'build', forbidden)
    pg = build_linear_qccd(3, 2)
    board = Placement(pg)
    for q in range(3):
        board.place(q, f't{q}:0')
    handler = CongestionHandler(planning_mode='incumbent')
    result = handler.resolve_path_congestion(pg.shortest_path('t0:0', 't2:1'), board, pg)
    assert result.plan is not None
    BoardView(pg, board).apply(result.plan.ops)
    assert handler.stats['qubo_solved'] == 0
    assert handler.stats['incumbent_plans'] > 0


def test_guard_uses_selected_time_objective():
    from walk_planner import TIME_COSTS_US, ops_cost
    pg = build_linear_qccd(3, 2)
    board = Placement(pg)
    for q in range(3):
        board.place(q, f't{q}:0')
    initial = BoardView(pg, board)
    circuit = Circuit(3)
    circuit.append_gate(CNOTGate(), (0, 2))
    router = ShawRoutingPass(pg, board, CongestionHandler(cost_model=TIME_COSTS_US, planning_mode='incumbent'))
    asyncio.run(router.run(circuit, PassData(circuit)))
    assert router.routing_cost <= router.portfolio_stats['candidate_costs']['baseline']
    assert router.routing_cost == ops_cost(pg, initial, router.movement_trace, TIME_COSTS_US)
