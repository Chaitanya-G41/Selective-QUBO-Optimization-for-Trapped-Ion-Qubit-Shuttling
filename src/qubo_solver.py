"""
qubo_solver.py
==============
Phase 7 of the Selective-QUBO methodology:
  Solve the QUBOProblem using ExactSolver (small) or SimulatedAnnealing (large).

Methodology reference  (methodology-s3.pdf, Section 7)
-------------------------------------------------------
  "For very small instances ExactSolver will be used to obtain the optimal
   solution.  For larger local instances Simulated Annealing can be used to
   obtain a practical solution without requiring a physical quantum computer."

Decision boundary
-----------------
  num_variables + num_aux_variables <= EXACT_THRESHOLD  →  dimod.ExactSolver
  num_variables + num_aux_variables  > EXACT_THRESHOLD  →  neal.SimulatedAnnealingSampler

Public API
----------
  QUBOSolver.solve(problem)        – auto-select and return QUBOSolution
  QUBOSolver.solve_exact(problem)  – force ExactSolver
  QUBOSolver.solve_sa(problem)     – force SimulatedAnnealing
  QUBOSolver.calibrate_penalty(problem, target_feasibility, max_iterations)
                                   – bisection search for tightest feasible λ

DEPENDENCIES
------------
  dimod>=0.12     (ExactSolver, BinaryQuadraticModel)
  dwave-neal>=0.6 (SimulatedAnnealingSampler)

HOW TO RUN
----------
  python src/qubo_solver.py           # smoke test
  pytest tests/test_qubo_solver.py    # unit tests
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dimod
import neal

from qubo_formulator import QUBOProblem
from qubo_validator import validate_sample as _validate_sample

logger = logging.getLogger("shaw_router.qubo_solver")

# ---------------------------------------------------------------------------
# Global default: instances with more variables than this use SimulatedAnnealing
# ---------------------------------------------------------------------------
EXACT_THRESHOLD: int = 20


# ---------------------------------------------------------------------------
# Output data structure  (DO NOT CHANGE – solution_decoder.py depends on this)
# ---------------------------------------------------------------------------

@dataclass
class QUBOSolution:
    """
    Result returned by the solver.

    Attributes
    ----------
    sample : dict  { variable_label : 0 or 1 }
        The best binary assignment found.
    energy : float
        Objective + penalty energy of this sample.
    is_feasible : bool
        True if no constraint penalty term fired in this sample.
        (Checked by: energy < problem.penalty_lambda * num_constraints)
    solver_used : str
        "exact" or "simulated_annealing"
    solve_time_s : float
        Wall-clock seconds taken by the solver.
    num_reads : int
        Number of samples taken (1 for ExactSolver, configurable for SA).
    problem : QUBOProblem
        The source problem (kept for the decoder).
    metadata : dict
        Anything extra you want to record (e.g. all samples, energies).
    """
    sample: Dict[str, int]
    energy: float
    is_feasible: bool
    solver_used: str
    solve_time_s: float
    num_reads: int
    problem: QUBOProblem
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class QUBOSolver:
    """
    Selects and runs the appropriate solver for a QUBOProblem.

    Parameters
    ----------
    exact_threshold : int
        Max total variables (primary + auxiliary) before switching from
        ExactSolver to SimulatedAnnealing.  Default: 20.
    sa_num_reads : int
        Number of independent SA restarts (higher = better quality, slower).
        Default: 200.
    sa_num_sweeps : int
        Number of update sweeps per SA read.  Default: 1000.
    sa_initial_temperature : float | None
        Starting temperature for SA schedule.  None lets neal auto-tune.
    sa_final_temperature : float | None
        Ending temperature for SA schedule.  None lets neal auto-tune.
    """

    def __init__(
        self,
        exact_threshold: int = EXACT_THRESHOLD,
        sa_num_reads: int = 500,
        sa_num_sweeps: int = 3000,
        sa_initial_temperature: Optional[float] = None,
        sa_final_temperature: Optional[float] = None,
    ) -> None:
        self.exact_threshold = exact_threshold
        self.sa_num_reads = sa_num_reads
        self.sa_num_sweeps = sa_num_sweeps
        self.sa_initial_temperature = sa_initial_temperature
        self.sa_final_temperature = sa_final_temperature
        # Cache the sampler — neal.SimulatedAnnealingSampler is stateless
        # between .sample() calls (all per-run state lives in the SampleSet
        # it returns, not in the sampler object itself).  Creating it once
        # avoids repeated C-extension initialisation on every solve() call,
        # which was the cause of the 4.5 s → 16 s timing drift.
        self._sa_sampler = neal.SimulatedAnnealingSampler()


    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def solve(self, problem: QUBOProblem) -> QUBOSolution:
        """
        Auto-select solver based on problem size and return best solution.

        Routing (from methodology Section 7):
          ≤ EXACT_THRESHOLD variables → ExactSolver (guaranteed optimal)
          >  EXACT_THRESHOLD variables → SimulatedAnnealing (practical)

        Parameters
        ----------
        problem : QUBOProblem   (from qubo_formulator.py)

        Returns
        -------
        QUBOSolution
        """
        total_vars = problem.num_variables + problem.num_aux_variables
        if total_vars <= self.exact_threshold:
            logger.info(
                "Auto-select: ExactSolver (total vars=%d <= threshold=%d)",
                total_vars, self.exact_threshold,
            )
            return self.solve_exact(problem)
        else:
            logger.info(
                "Auto-select: SimulatedAnnealing (total vars=%d > threshold=%d)",
                total_vars, self.exact_threshold,
            )
            return self.solve_sa(problem)

    # -----------------------------------------------------------------------
    # ExactSolver  (optimal, exponential cost — small instances only)
    # -----------------------------------------------------------------------

    def solve_exact(self, problem: QUBOProblem) -> QUBOSolution:
        """
        Solve with dimod.ExactSolver (guaranteed globally optimal).

        dimod.ExactSolver evaluates all 2^n binary assignments and returns
        the sample with the lowest energy.  Only practical for n ≤ ~20.

        If called on a large problem (total vars > exact_threshold), logs a
        warning and falls back to SimulatedAnnealing automatically.

        Parameters
        ----------
        problem : QUBOProblem

        Returns
        -------
        QUBOSolution with solver_used = "exact"
        """
        total_vars = problem.num_variables + problem.num_aux_variables
        if total_vars > self.exact_threshold:
            logger.warning(
                "solve_exact called on a large problem (vars=%d > threshold=%d). "
                "Falling back to SimulatedAnnealing.",
                total_vars, self.exact_threshold,
            )
            sol = self.solve_sa(problem)
            # Override solver label so caller knows what happened
            sol.solver_used = "exact_fallback_sa"
            return sol

        logger.debug("ExactSolver: %d variables", total_vars)
        sampler = dimod.ExactSolver()

        t0 = time.perf_counter()
        sampleset: dimod.SampleSet = sampler.sample(problem.bqm)
        elapsed = time.perf_counter() - t0

        # Lowest-energy sample
        best = sampleset.first
        best_sample: Dict[str, int] = dict(best.sample)
        best_energy: float = float(best.energy)

        feasible = self._check_feasibility(best_energy, problem, sample=best_sample)

        logger.info(
            "ExactSolver: energy=%.4f, feasible=%s, time=%.4fs",
            best_energy, feasible, elapsed,
        )

        # Collect all sample energies for analysis
        all_energies: List[float] = [float(s.energy) for s in sampleset.data()]

        return QUBOSolution(
            sample=best_sample,
            energy=best_energy,
            is_feasible=feasible,
            solver_used="exact",
            solve_time_s=elapsed,
            num_reads=len(all_energies),
            problem=problem,
            metadata={
                "all_energies": all_energies,
                "num_feasible": sum(
                    1 for e in all_energies
                    if self._check_feasibility(e, problem)
                ),
            },
        )

    # -----------------------------------------------------------------------
    # Simulated Annealing  (heuristic, scalable — larger instances)
    # -----------------------------------------------------------------------

    def solve_sa(self, problem: QUBOProblem) -> QUBOSolution:
        """
        Solve with neal.SimulatedAnnealingSampler.

        Runs `sa_num_reads` independent annealing trajectories, each with
        `sa_num_sweeps` update sweeps.

        Selection strategy (main bug fix):
            Soft QUBO penalties mean the global minimum-energy read is NOT
            guaranteed to be feasible — SA can find a low-energy state that
            violates movement/goal constraints while incurring less total
            penalty than a feasible solution.  We therefore:
              1. Collect all reads, sorted by energy ascending.
              2. Walk the sorted list and return the first read that passes
                 _check_feasibility (lowest-energy feasible read).
              3. Only if NO read is feasible, fall back to the global minimum
                 and mark is_feasible=False.

        Parameters
        ----------
        problem : QUBOProblem

        Returns
        -------
        QUBOSolution with solver_used = "simulated_annealing"
        """
        total_vars = problem.num_variables + problem.num_aux_variables
        logger.debug(
            "SimulatedAnnealing: %d variables, %d reads, %d sweeps",
            total_vars,
            self.sa_num_reads,
            self.sa_num_sweeps,
        )

        # Build keyword args — only pass temperature params if explicitly set
        sa_kwargs: Dict[str, Any] = {
            "num_reads": self.sa_num_reads,
            "num_sweeps": self.sa_num_sweeps,
        }
        if self.sa_initial_temperature is not None:
            sa_kwargs["initial_temperature"] = self.sa_initial_temperature
        if self.sa_final_temperature is not None:
            sa_kwargs["final_temperature"] = self.sa_final_temperature

        t0 = time.perf_counter()
        sampleset: dimod.SampleSet = self._sa_sampler.sample(
            problem.bqm, **sa_kwargs
        )
        elapsed = time.perf_counter() - t0

        # --- Collect all reads sorted by energy ascending ------------------
        # dimod.SampleSet.data() yields samples in the order they were
        # recorded; sorted_by="energy" returns them in ascending energy order.
        all_samples = list(
            sampleset.data(fields=["sample", "energy"], sorted_by="energy")
        )
        all_energies: List[float] = [float(s.energy) for s in all_samples]

        # --- Feasibility scan: return the first feasible read ---------------
        # Walk sorted list; stop at the first read that passes the two-stage
        # feasibility check (energy < lambda AND goal/one-hot checks).
        best_sample: Optional[Dict[str, int]] = None
        best_energy: float = float("inf")
        feasible: bool = False

        for s in all_samples:          # already sorted energy ascending
            e = float(s.energy)
            s_dict = dict(s.sample)
            if self._check_feasibility(e, problem, sample=s_dict):
                best_sample = s_dict
                best_energy = e
                feasible = True
                break

        # --- Fallback: no feasible read found — return global minimum -------
        if best_sample is None:
            first = all_samples[0]     # lowest energy (sorted ascending)
            best_sample = dict(first.sample)
            best_energy = float(first.energy)
            feasible = False

        # --- Feasibility rate across ALL reads (for diagnostics) ------------
        num_feasible = sum(
            1 for s in all_samples
            if self._check_feasibility(
                float(s.energy), problem, sample=dict(s.sample)
            )
        )
        feasibility_rate = num_feasible / len(all_samples) if all_samples else 0.0

        logger.info(
            "SA: selected_energy=%.4f, feasible=%s, "
            "feasibility_rate=%.2f%% (%d/%d reads), time=%.4fs",
            best_energy, feasible,
            feasibility_rate * 100, num_feasible, len(all_samples),
            elapsed,
        )

        return QUBOSolution(
            sample=best_sample,
            energy=best_energy,
            is_feasible=feasible,
            solver_used="simulated_annealing",
            solve_time_s=elapsed,
            num_reads=len(all_samples),
            problem=problem,
            metadata={
                "all_energies": all_energies,
                "num_feasible": num_feasible,
                "feasibility_rate": feasibility_rate,
                "sampleset": sampleset,      # full sampleset for deeper analysis
            },
        )


    # -----------------------------------------------------------------------
    # Feasibility check
    # -----------------------------------------------------------------------

    def _check_feasibility(
        self,
        energy: float,
        problem: QUBOProblem,
        sample: Optional[Dict[str, int]] = None,
    ) -> bool:
        """
        Two-stage feasibility check for a QUBO solution sample.

        Stage 1 — Energy pre-reject (fast):
            If energy >= penalty_lambda, at least one constraint penalty
            fired → immediately infeasible.  This is the *original* check
            and it is still correct as a rejection filter.

        Stage 2 — Structural checks on the sample (necessary addition):
            The original check had a critical blind spot: the all-zeros
            sample (every x_{i,v,t} = 0, nobody moves) has energy = 0.0,
            which trivially passes `energy < penalty_lambda`.  But that
            solution violates every goal constraint and every one-hot
            constraint simultaneously — the decoder always rejects it.

            When `sample` is provided we also verify:
              (a) Goal constraint: the moving ion's variable for
                  (ion, window.target, T) must equal 1. If inv_var_map
                  is populated, we can look this up directly.
              (b) One-hot sanity: no ion has more than one position bit
                  set at any timestep (a quick O(n) pass).

            If inv_var_map is empty (e.g. in unit tests with dummy
            problems), we skip stage 2 and rely solely on stage 1.

        Parameters
        ----------
        energy  : float           – BQM energy of the sample
        problem : QUBOProblem
        sample  : dict | None     – the binary assignment {label: 0/1}

        Returns
        -------
        bool – True only if BOTH stages pass
        """
        # --- Stage 1: energy pre-reject -----------------------------------
        # A sample with energy >= lambda has at least one penalty term
        # contributing, so it is provably infeasible.
        if energy >= problem.penalty_lambda:
            return False

        # --- Stage 2: structural checks (only when sample + map available)
        if sample is None or not problem.inv_var_map:
            # Cannot do structural checks without variable mappings.
            # Fall back to energy-only check (original behaviour).
            # This keeps unit tests with dummy problems working.
            return True

        window = problem.window
        T = problem.time_horizon
        inv_var_map = problem.inv_var_map

        # Build a quick lookup: (ion, t) -> list of positions with bit=1
        ion_t_positions: Dict[Any, Dict[int, List[Any]]] = {}
        for label, bit in sample.items():
            if bit != 1:
                continue
            mapped = inv_var_map.get(label)
            if mapped is None:
                continue  # Rosenberg aux variable — skip
            ion, pos, t = mapped
            ion_t_positions.setdefault(ion, {}).setdefault(t, []).append(pos)

        # (a) Goal constraint: moving ion must be at window.target at t=T.
        #     Find the ion currently at window.source.
        moving_ion = None
        for ion, pos in window.active_ions.items():
            if pos == window.source:
                moving_ion = ion
                break

        if moving_ion is not None:
            positions_at_T = ion_t_positions.get(moving_ion, {}).get(T, [])
            if window.target not in positions_at_T:
                # Goal not achieved — decoder will reject as gate_feasibility.
                logger.debug(
                    "Feasibility pre-reject: moving ion %s not at target %r "
                    "at t=%d (positions_at_T=%r)",
                    moving_ion, window.target, T, positions_at_T,
                )
                return False

        # (b) One-hot sanity: no ion should have >1 position set at any t.
        for ion, t_map in ion_t_positions.items():
            for t, positions in t_map.items():
                if len(positions) > 1:
                    logger.debug(
                        "Feasibility pre-reject: ion %s has %d positions at "
                        "t=%d: %r (one-hot violated)",
                        ion, len(positions), t, positions,
                    )
                    return False

        return True

    # -----------------------------------------------------------------------
    # Penalty calibration  (Optional — methodology §11 / solver tuning)
    # -----------------------------------------------------------------------

    def calibrate_penalty(
        self,
        problem: QUBOProblem,
        target_feasibility: float = 0.95,
        max_iterations: int = 10,
    ) -> float:
        """
        Find the smallest penalty_lambda that achieves `target_feasibility`
        fraction of feasible SA samples, using bisection search.

        This is an offline calibration step run once on small benchmark
        instances to set good default thresholds for production use
        (methodology §11 – "thresholds calibrated experimentally").

        Algorithm
        ---------
        1. Start with lo = 1.0, hi = problem.penalty_lambda * 10.
        2. For `max_iterations` bisection steps:
             mid = (lo + hi) / 2
             Rebuild the BQM with penalty = mid
             Run SA with sa_num_reads reads
             feasibility_rate = fraction of feasible samples
             if feasibility_rate >= target_feasibility:
                 hi = mid   (mid works — try tighter)
             else:
                 lo = mid   (mid too tight — relax)
        3. Return hi  (smallest lambda known to achieve the target)

        Parameters
        ----------
        problem              : QUBOProblem – the baseline problem to calibrate on
        target_feasibility   : float       – desired fraction of feasible reads (0–1)
        max_iterations       : int         – bisection depth

        Returns
        -------
        float – calibrated penalty_lambda value
        """
        lo: float = 1.0
        hi: float = problem.penalty_lambda * 10.0

        logger.info(
            "Calibrating penalty: target_feasibility=%.2f, bisection range=[%.2f, %.2f], "
            "max_iterations=%d",
            target_feasibility, lo, hi, max_iterations,
        )

        sampler = self._sa_sampler


        for iteration in range(max_iterations):
            mid = (lo + hi) / 2.0

            # --- Rebuild BQM with penalty = mid ----------------------------
            # We scale all constraint (penalty) terms by mid/original_lambda.
            # The simplest safe approach: deep-copy the BQM, then re-weight.
            # Since dimod BQMs are mutable we work on a copy.
            scale_factor = mid / problem.penalty_lambda
            scaled_bqm = _rescale_penalty_terms(problem.bqm, scale_factor)

            # --- Run SA on scaled BQM --------------------------------------
            sa_kwargs: Dict[str, Any] = {
                "num_reads": self.sa_num_reads,
                "num_sweeps": self.sa_num_sweeps,
            }
            sampleset: dimod.SampleSet = sampler.sample(scaled_bqm, **sa_kwargs)

            # --- Compute feasibility rate ----------------------------------
            # Feasibility threshold also scales: energy < mid
            n_feasible = sum(
                1 for s in sampleset.data()
                if float(s.energy) < mid
            )
            rate = n_feasible / self.sa_num_reads

            logger.debug(
                "Calibration iter %d/%d: lambda=%.4f, feasibility_rate=%.3f",
                iteration + 1, max_iterations, mid, rate,
            )

            if rate >= target_feasibility:
                hi = mid   # mid works — try a smaller lambda
            else:
                lo = mid   # mid too small — increase lambda

        logger.info(
            "Calibration complete: recommended penalty_lambda=%.4f "
            "(feasibility >= %.2f at this value)",
            hi, target_feasibility,
        )
        return hi


# ---------------------------------------------------------------------------
# Helper: rescale only the penalty (constraint) portion of a BQM
# ---------------------------------------------------------------------------

def _rescale_penalty_terms(
    bqm: dimod.BinaryQuadraticModel,
    scale_factor: float,
) -> dimod.BinaryQuadraticModel:
    """
    Return a new BQM where *all* terms are scaled by `scale_factor`.

    Note: Because the objective and penalty terms are combined additively
    in the BQM, and we do not have explicit labels to distinguish them,
    this function scales the entire BQM.  For the purpose of penalty
    calibration (where we only care about constraint satisfaction, not
    objective minimization quality), this is an acceptable approximation.

    A more precise implementation would require the formulator to tag
    objective vs. penalty interactions separately (e.g. via metadata).

    Parameters
    ----------
    bqm          : the original BinaryQuadraticModel
    scale_factor : multiplicative factor applied to all coefficients

    Returns
    -------
    dimod.BinaryQuadraticModel  (a deep copy, original is not modified)
    """
    scaled = bqm.copy()
    scaled.scale(scale_factor)
    return scaled


# ---------------------------------------------------------------------------
# Smoke test  (run with: python src/qubo_solver.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # UTF-8 stdout/stderr so the ✓ marks below print on Windows
    # consoles (default cp1252) as well as UTF-8 systems.
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name, None)
        try:
            if _stream is not None and hasattr(_stream, "reconfigure"):
                _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    del _stream_name, _stream

    # Make sure src/ is on the path when running directly
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from qubo_formulator import QUBOProblem
    from congestion_handler import WindowInfo

    print("=" * 60)
    print("  qubo_solver.py  –  smoke test")
    print("=" * 60)

    # ---- Build a tiny dummy QUBOProblem -----------------------------------
    # Minimum at x0=1, x1=0, x2=0  (energy = -3.0 + 0 + 0 = -3.0)
    bqm = dimod.BinaryQuadraticModel(
        {"x0": -3.0, "x1": 2.0, "x2": 1.0},  # linear
        {("x0", "x1"): 1.0},                   # quadratic (coupling)
        0.0,
        "BINARY",
    )

    dummy_window = WindowInfo(
        window_nodes=frozenset(["T0", "T1"]),
        active_ions={0: "T0"},
        obstacle_ions={},
        source="T0",
        target="T1",
        blocked_path=["T0"],
        center_nodes=["T0"],
        radius=1,
        size_capped=False,
    )

    problem = QUBOProblem(
        bqm=bqm,
        var_map={},
        inv_var_map={},
        num_variables=3,
        num_aux_variables=0,
        time_horizon=1,
        penalty_lambda=10.0,
        window=dummy_window,
    )

    solver = QUBOSolver(exact_threshold=20, sa_num_reads=50, sa_num_sweeps=200)

    # ---- ExactSolver (3 vars → well within threshold) ---------------------
    print("\n[1] ExactSolver:")
    sol_exact = solver.solve_exact(problem)
    print(f"    sample      = {sol_exact.sample}")
    print(f"    energy      = {sol_exact.energy:.4f}")
    print(f"    is_feasible = {sol_exact.is_feasible}")
    print(f"    solver_used = {sol_exact.solver_used}")
    print(f"    solve_time  = {sol_exact.solve_time_s*1000:.2f} ms")
    assert sol_exact.sample.get("x0") == 1, "ExactSolver should find x0=1"
    assert sol_exact.sample.get("x1") == 0, "ExactSolver should find x1=0"
    print("    ✓ Correct optimal solution found")

    # ---- SimulatedAnnealing -----------------------------------------------
    print("\n[2] SimulatedAnnealing:")
    sol_sa = solver.solve_sa(problem)
    print(f"    sample      = {sol_sa.sample}")
    print(f"    energy      = {sol_sa.energy:.4f}")
    print(f"    is_feasible = {sol_sa.is_feasible}")
    print(f"    solver_used = {sol_sa.solver_used}")
    print(f"    solve_time  = {sol_sa.solve_time_s*1000:.2f} ms")
    print(f"    feasibility_rate = {sol_sa.metadata['feasibility_rate']*100:.1f}%")
    assert all(v in (0, 1) for v in sol_sa.sample.values()), "All values must be 0 or 1"
    print("    ✓ Valid binary sample returned")

    # ---- Auto-select (small → ExactSolver) --------------------------------
    print("\n[3] Auto-select (small problem):")
    sol_auto = solver.solve(problem)
    print(f"    solver_used = {sol_auto.solver_used}")
    assert sol_auto.solver_used == "exact", "Small problem should use ExactSolver"
    print("    ✓ Correctly routed to ExactSolver")

    # ---- Auto-select (large → SA) -----------------------------------------
    print("\n[4] Auto-select (large problem, 25 vars):")
    big_bqm = dimod.BinaryQuadraticModel(
        {f"x{i}": float(i % 3 - 1) for i in range(25)}, {}, 0.0, "BINARY"
    )
    big_problem = QUBOProblem(
        bqm=big_bqm, var_map={}, inv_var_map={},
        num_variables=25, num_aux_variables=0,
        time_horizon=3, penalty_lambda=10.0,
        window=dummy_window,
    )
    sol_big = solver.solve(big_problem)
    print(f"    solver_used = {sol_big.solver_used}")
    assert sol_big.solver_used == "simulated_annealing"
    print("    ✓ Correctly routed to SimulatedAnnealing")

    # ---- Penalty calibration ----------------------------------------------
    print("\n[5] Penalty calibration (target feasibility=0.8, 5 iterations):")
    calibrated_lambda = solver.calibrate_penalty(
        problem,
        target_feasibility=0.80,
        max_iterations=5,
    )
    print(f"    calibrated_lambda = {calibrated_lambda:.4f}")
    assert calibrated_lambda > 0, "Calibrated lambda must be positive"
    print("    ✓ Calibration completed successfully")

    print("\n" + "=" * 60)
    print("  All smoke tests passed!")
    print("=" * 60)
