"""
test_solution_decoder.py

Unit tests for SolutionDecoder.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from congestion_handler import WindowInfo
from position_graph import Placement, build_linear_qccd
from qubo_formulator import QUBOProblem
from qubo_solver import QUBOSolution
from solution_decoder import DecodedSolution, SolutionDecoder, ValidationResult


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _make_dummy_solution(is_feasible: bool = True, energy: float = 1.0):
    dummy_window = WindowInfo(
        window_nodes=frozenset(["t0:0", "seg0", "t1:0"]),
        active_ions={0: "t0:0", 1: "t1:0"},
        obstacle_ions={},
        source="t0:0",
        target="t1:0",
        blocked_path=["seg0"],
        center_nodes=["seg0"],
        radius=1,
        size_capped=False,
    )
    problem = QUBOProblem(
        bqm=None,
        var_map={},
        inv_var_map={},
        num_variables=6,
        num_aux_variables=0,
        time_horizon=2,
        penalty_lambda=10.0,
        window=dummy_window,
    )
    return QUBOSolution(
        sample={},
        energy=energy,
        is_feasible=is_feasible,
        solver_used="exact",
        solve_time_s=0.01,
        num_reads=1,
        problem=problem,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_decoder_initialization():
    decoder = SolutionDecoder(strict_capacity=True)
    assert decoder.strict_capacity is True


def test_reject_when_infeasible():
    solution = _make_dummy_solution(is_feasible=False, energy=100.0)
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])

    decoder = SolutionDecoder()
    decoded = decoder.decode_and_validate(solution, pl, pg)
    assert isinstance(decoded, DecodedSolution)
    assert decoded.accepted is False
    assert decoded.violation == "infeasible_energy"


def test_decode_trajectories_reconstructs_path():
    decoder = SolutionDecoder()
    solution = _make_dummy_solution(is_feasible=True, energy=2.0)
    trajectories = decoder._decode_trajectories(solution)
    assert isinstance(trajectories, dict)


def test_validation_movement_legality():
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    window = _make_dummy_solution().problem.window
    trajectories = {0: ["t0:0", "seg0", "t1:0"]}
    res = decoder._check_movement_legality(trajectories, window, pg)
    assert isinstance(res, ValidationResult)


def test_cost_comparison_rejection_if_worse():
    """Verify methodology §9: QUBO solution rejected if C_QUBO >= C_heuristic."""
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    solution = _make_dummy_solution(is_feasible=True, energy=50.0)
    decoded = decoder.decode_and_validate(solution, pl, pg)
    if decoded.accepted:
        assert decoded.C_QUBO < decoded.C_heuristic


# ---------------------------------------------------------------------------
# One test per rejection reason, plus one accept-path test
# ---------------------------------------------------------------------------

def _window_for(source, target, active_ions, blocked_path):
    return WindowInfo(
        window_nodes=frozenset(active_ions.values()) | frozenset(blocked_path) | {source, target},
        active_ions=dict(active_ions),
        obstacle_ions={},
        source=source,
        target=target,
        blocked_path=list(blocked_path),
        center_nodes=list(blocked_path),
        radius=1,
        size_capped=False,
    )


def test_rejects_on_movement_legality_violation():
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)  # t0:0,t0:1,seg0,t1:0,t1:1,seg1,t2:0,t2:1
    window = _window_for("t0:0", "t2:0", {0: "t0:0"}, blocked_path=[])
    # t0:0 -> t2:0 is NOT a legal single edge (must go through seg0/t1 first).
    trajectories = {0: ["t0:0", "t2:0"]}
    res = decoder._check_movement_legality(trajectories, window, pg)
    assert res.passed is False
    assert res.constraint == "movement_legality"


def test_rejects_on_occupancy_violation():
    decoder = SolutionDecoder()
    window = _window_for("t0:0", "t1:0", {0: "t0:0", 1: "t1:0"}, blocked_path=[])
    # Both ions land on the same position at t=1.
    trajectories = {0: ["t0:0", "seg0"], 1: ["t1:0", "seg0"]}
    res = decoder._check_occupancy(trajectories, window)
    assert res.passed is False
    assert res.constraint == "occupancy"


def test_rejects_on_capacity_violation():
    decoder = SolutionDecoder(strict_capacity=True)
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    # A segment node in this model holds exactly 1 ion (slot-style), so
    # 2 distinct active ions both sitting on "seg0" at the same t exceeds
    # capacity even though _check_occupancy already catches the same-ion
    # case -- here we instead combine an active ion with an obstacle ion
    # already fixed on that position, which only _check_capacity sees.
    window = WindowInfo(
        window_nodes=frozenset({"t0:0", "seg0", "t1:0"}),
        active_ions={0: "t0:0"},
        obstacle_ions={99: "seg0"},  # fixed ion already parked on seg0
        source="t0:0",
        target="t1:0",
        blocked_path=["seg0"],
        center_nodes=["seg0"],
        radius=1,
        size_capped=False,
    )
    trajectories = {0: ["t0:0", "seg0"]}  # tries to move onto the obstacle's spot
    res = decoder._check_capacity(trajectories, window, pg)
    assert res.passed is False
    assert res.constraint == "capacity"


def test_rejects_on_collision_avoidance_violation():
    decoder = SolutionDecoder()
    window = _window_for("t0:0", "t1:0", {0: "t0:0", 1: "t1:0"}, blocked_path=["seg0"])
    # Ion 0 goes t0:0->seg0 while ion 1 goes seg0->t0:0 in the same step:
    # a crossing move with no explicit swap.
    trajectories = {0: ["t0:0", "seg0"], 1: ["seg0", "t0:0"]}
    res = decoder._check_collision_avoidance(trajectories, window)
    assert res.passed is False
    assert res.constraint == "collision_avoidance"


def test_rejects_on_gate_feasibility_violation():
    decoder = SolutionDecoder()
    window = _window_for("t0:0", "t1:0", {0: "t0:0"}, blocked_path=["seg0"])
    # Moving ion never reaches the target.
    trajectories = {0: ["t0:0", "seg0", "seg0"]}
    res = decoder._check_gate_feasibility(trajectories, window)
    assert res.passed is False
    assert res.constraint == "gate_feasibility"


def test_rejects_when_not_better_than_heuristic():
    """End-to-end: a valid but longer-than-heuristic QUBO path is rejected."""
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])

    window = WindowInfo(
        window_nodes=frozenset({"t0:0", "seg0", "t1:0"}),
        active_ions={0: "t0:0"},
        obstacle_ions={},
        source="t0:0",
        target="t1:0",
        blocked_path=["seg0"],
        center_nodes=["seg0"],
        radius=1,
        size_capped=False,
    )
    # Deliberately pad the path with a pointless detour that still only
    # uses legal edges (t0:0<->t0:1 swap, t0:1<->seg0 merge_split,
    # seg0<->t1:0 merge_split), so C_QUBO ends up >= the heuristic cost.
    var_map = {}
    inv_var_map = {
        "x_0_t00_0": (0, "t0:0", 0),
        "x_0_t01_1": (0, "t0:1", 1),
        "x_0_t00_2": (0, "t0:0", 2),
        "x_0_t01_3": (0, "t0:1", 3),
        "x_0_seg0_4": (0, "seg0", 4),
        "x_0_t10_5": (0, "t1:0", 5),
    }
    sample = {k: 1 for k in inv_var_map}

    problem = QUBOProblem(
        bqm=None, var_map=var_map, inv_var_map=inv_var_map,
        num_variables=6, num_aux_variables=0, time_horizon=5,
        penalty_lambda=10.0, window=window,
    )
    solution = QUBOSolution(
        sample=sample, energy=1.0, is_feasible=True,
        solver_used="exact", solve_time_s=0.01, num_reads=1, problem=problem,
    )

    decoded = decoder.decode_and_validate(solution, pl, pg)
    assert decoded.accepted is False
    assert decoded.violation == "not_better_than_heuristic"


def test_accepts_a_valid_better_solution():
    """End-to-end accept path: a legal, shorter-than-heuristic solution
    passes all 5 checks and gets accepted."""
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])

    window = WindowInfo(
        window_nodes=frozenset({"t0:0", "seg0", "t1:0"}),
        active_ions={0: "t0:0"},
        obstacle_ions={},
        source="t0:0",
        target="t1:0",
        blocked_path=["seg0"],
        center_nodes=["seg0"],
        radius=1,
        size_capped=False,
    )
    # Direct, legal, minimal path: t0:0 -> t0:1 -> seg0 -> t1:0 (3 hops;
    # seg0 only connects to the trap's boundary slot t0:1, not t0:0).
    inv_var_map = {
        "x_0_t00_0": (0, "t0:0", 0),
        "x_0_t01_1": (0, "t0:1", 1),
        "x_0_seg0_2": (0, "seg0", 2),
        "x_0_t10_3": (0, "t1:0", 3),
    }
    sample = {k: 1 for k in inv_var_map}

    problem = QUBOProblem(
        bqm=None, var_map={}, inv_var_map=inv_var_map,
        num_variables=4, num_aux_variables=0, time_horizon=3,
        penalty_lambda=10.0, window=window,
    )
    solution = QUBOSolution(
        sample=sample, energy=1.0, is_feasible=True,
        solver_used="exact", solve_time_s=0.01, num_reads=1, problem=problem,
    )

    decoded = decoder.decode_and_validate(solution, pl, pg)
    assert decoded.accepted is True
    assert decoded.decoded_path == ["t0:0", "t0:1", "seg0", "t1:0"]
    assert decoded.C_QUBO == 3

    # apply_to_placement should move ion 0 all the way to t1:0.
    decoder.apply_to_placement(decoded, pl, pg)
    assert pl.position_of(0) == "t1:0"


def test_apply_to_placement_skip_ions_leaves_mover():
    """skip_ions: blockers move per plan, the skipped mover stays put
    (selective-QUBO hookup walks it separately -- replaying it too
    would walk it twice)."""
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])

    decoded = DecodedSolution(
        accepted=True,
        decoded_path=["t0:0", "t0:1", "seg0", "t1:1"],
        C_QUBO=3, C_heuristic=5, improvement=0.4, violation=None,
        trajectories={
            0: ["t0:0", "t0:1", "seg0", "t1:1"],
            1: ["t1:0", "t1:0", "t1:0", "t1:1"],
        },
    )
    decoder.apply_to_placement(decoded, pl, pg, skip_ions=frozenset({0}))
    assert pl.position_of(0) == "t0:0"  # mover untouched
    assert pl.position_of(1) == "t1:1"  # blocker followed plan


# ---------------------------------------------------------------------------
# apply_to_placement raise-paths
# ---------------------------------------------------------------------------

def test_apply_to_placement_raises_on_rejected_solution():
    """Calling apply_to_placement on a rejected DecodedSolution is a
    programming error in the caller -- should fail loudly, not silently
    no-op or partially apply anything."""
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])

    rejected = DecodedSolution(
        accepted=False, decoded_path=None,
        C_QUBO=0, C_heuristic=5,
        improvement=0.0, violation="movement_legality",
        trajectories={},
    )
    with pytest.raises(ValueError, match="rejected"):
        decoder.apply_to_placement(rejected, pl, pg)


def test_apply_to_placement_raises_on_placement_divergence():
    """If the live Placement doesn't match what the decoded trajectory
    expected at t=0 (e.g. something else moved the ion in the meantime),
    apply_to_placement must refuse rather than silently apply a move
    that no longer reflects reality."""
    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    pl = Placement(pg)
    # Trajectory below claims ion 0 starts at "t0:0", but we actually
    # place it at "t0:1" -- a deliberate mismatch.
    pl.place(0, pg.slots_of("t0")[1])

    decoded = DecodedSolution(
        accepted=True,
        decoded_path=["t0:0", "seg0", "t1:0"],
        C_QUBO=2, C_heuristic=5, improvement=0.6, violation=None,
        trajectories={0: ["t0:0", "t0:1", "seg0", "t1:0"]},
    )
    with pytest.raises(ValueError, match="diverged"):
        decoder.apply_to_placement(decoded, pl, pg)


def test_apply_to_placement_raises_on_rotation_deadlock():
    """A 3-ion rotation cycle (A->B's spot, B->C's spot, C->A's spot, all
    in the same step) is NOT caught by _check_collision_avoidance (which
    only detects direct 2-ion crossing pairs), so apply_to_placement must
    detect the resulting deadlock itself and raise clearly instead of
    hanging in its unblocked-first retry loop."""
    decoder = SolutionDecoder()

    # Build a small 3-node triangle graph directly -- build_linear_qccd's
    # chain topology has no cycles, so this needs manual wiring.
    pg = build_linear_qccd(num_traps=1, trap_capacity=1)  # placeholder, replaced below
    from position_graph import PositionGraph
    pg = PositionGraph()
    pg.add_trap("A", 1)  # "A:0"
    pg.add_trap("B", 1)  # "B:0"
    pg.add_trap("C", 1)  # "C:0"
    pg.connect("A:0", "B:0", label="move")
    pg.connect("B:0", "C:0", label="move")
    pg.connect("C:0", "A:0", label="move")

    pl = Placement(pg)
    pl.place("X", "A:0")
    pl.place("Y", "B:0")
    pl.place("Z", "C:0")

    decoded = DecodedSolution(
        accepted=True,
        decoded_path=None,  # not used by apply_to_placement
        C_QUBO=1, C_heuristic=5, improvement=0.8, violation=None,
        trajectories={
            "X": ["A:0", "B:0"],
            "Y": ["B:0", "C:0"],
            "Z": ["C:0", "A:0"],
        },
    )
    with pytest.raises(ValueError, match="deadlocked"):
        decoder.apply_to_placement(decoded, pl, pg)


def test_rejects_degenerate_single_node_path():
    """A sample whose mover trajectory collapses to one position must
    be rejected as empty_path -- never accepted with C_QUBO=0 (a
    zero-hop 'solution' claiming 100% improvement routes nothing).
    Observed live from SA samples before this guard existed."""
    from qubo_formulator import QUBOProblem
    from qubo_solver import QUBOSolution

    decoder = SolutionDecoder()
    pg = build_linear_qccd(num_traps=2, trap_capacity=2)
    pl = Placement(pg)
    pl.place(0, pg.slots_of("t0")[0])

    window = WindowInfo(
        window_nodes=frozenset({"t0:0", "seg0", "t1:0"}),
        active_ions={0: "t0:0"},
        obstacle_ions={},
        source="t0:0",
        target="t1:0",
        blocked_path=["seg0"],
        center_nodes=["seg0"],
        radius=1,
        size_capped=False,
    )
    # Mover "already" at the target for the whole episode: every check
    # passes (no moves at all), but the extracted path is one node.
    inv_var_map = {f"x_0_t10_{t}": (0, "t1:0", t) for t in range(3)}
    problem = QUBOProblem(
        bqm=None, var_map={}, inv_var_map=inv_var_map,
        num_variables=3, num_aux_variables=0, time_horizon=2,
        penalty_lambda=10.0, window=window,
    )
    solution = QUBOSolution(
        sample={k: 1 for k in inv_var_map}, energy=1.0, is_feasible=True,
        solver_used="exact", solve_time_s=0.01, num_reads=1, problem=problem,
    )
    decoded = decoder.decode_and_validate(solution, pl, pg)
    assert decoded.accepted is False
    assert decoded.violation == "empty_path"


if __name__ == "__main__":
    print("test_solution_decoder.py: all SolutionDecoder methods implemented.")
    print("Run: pytest tests/test_solution_decoder.py -v")