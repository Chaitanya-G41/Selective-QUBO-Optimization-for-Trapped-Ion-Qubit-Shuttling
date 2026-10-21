"""
qubo_solver.py
==============
Solves a QUBOProblem and returns the lowest-energy *distinct* samples.

  exact  dimod.ExactSolver, for problems with <= exact_threshold variables
  tabu   dwave-samplers TabuSearch (default for larger problems)
  sa     neal simulated annealing
  leap   D-Wave Leap hybrid solver (quantum + classical, cloud). Optional:
         needs `pip install dwave-system` and a Leap API token. Untested
         here; note each cloud call takes seconds, versus milliseconds
         for tabu on these small windows.

Tabu is the default because single-bit-flip annealing copes badly with
this model: moving one ion means leaving a one-hot state, which costs a
full penalty P, so annealing either scrambles a warm start or never
leaves it. Tabu search accepts those uphill steps deliberately and,
warm-started from the greedy walk, reliably finds cheaper plans.

Feasibility is decided by the decoder, which checks every physical
constraint explicitly instead of guessing from the energy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import dimod
import neal
from dwave.samplers import TabuSampler

from qubo_formulator import QUBOProblem

EXACT_THRESHOLD = 16  # ExactSolver enumerates 2^n states


@dataclass
class QUBOSolution:
    samples: List[Dict[str, int]]    # distinct samples, lowest energy first
    energies: List[float]
    solver_used: str                 # "exact" | "tabu" | "sa"
    solve_time_s: float
    problem: QUBOProblem
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample(self) -> Dict[str, int]:
        return self.samples[0]

    @property
    def energy(self) -> float:
        return self.energies[0]


class QUBOSolver:
    """
    Parameters
    ----------
    method          : "tabu", "sa" or "leap" for problems above exact_threshold
    num_reads       : independent restarts
    tabu_effort     : tabu iterations per BQM variable. An iteration
                      budget instead of a wall-clock timeout keeps runs
                      reproducible; 4000 matched the best plan found by
                      any tested setting on 299 benchmark windows.
    sa_num_sweeps   : sweeps per annealing read
    seed            : RNG seed (reproducible runs)
    max_candidates  : distinct samples handed to the decoder
    """

    def __init__(
        self,
        method: str = "tabu",
        exact_threshold: int = EXACT_THRESHOLD,
        num_reads: int = 2,
        tabu_effort: int = 4000,
        sa_num_sweeps: int = 1000,
        seed: Optional[int] = None,
        max_candidates: int = 16,
    ) -> None:
        if method not in ("tabu", "sa", "leap"):
            raise ValueError(f"method must be 'tabu', 'sa' or 'leap', got {method!r}")
        self.method = method
        self.exact_threshold = exact_threshold
        self.num_reads = num_reads
        self.tabu_effort = tabu_effort
        self.sa_num_sweeps = sa_num_sweeps
        self.seed = seed
        self.max_candidates = max_candidates

    def solve(
        self,
        problem: QUBOProblem,
        initial_state: Optional[Dict[str, int]] = None,
    ) -> QUBOSolution:
        """
        Solve `problem`. An `initial_state` (e.g. the greedy walk) warm
        starts the heuristic samplers; it is ignored by ExactSolver.
        """
        bqm = problem.bqm
        t0 = time.perf_counter()
        if len(bqm.variables) <= self.exact_threshold:
            used, sampleset = "exact", dimod.ExactSolver().sample(bqm)
        elif self.method == "tabu":
            used, sampleset = "tabu", self._tabu(bqm, initial_state)
        elif self.method == "sa":
            used, sampleset = "sa", self._sa(bqm, initial_state)
        else:
            used, sampleset = "leap", self._leap(bqm)
        elapsed = time.perf_counter() - t0

        best = list(sampleset.aggregate().truncate(self.max_candidates).data(["sample", "energy"]))
        return QUBOSolution(
            samples=[dict(row.sample) for row in best],
            energies=[float(row.energy) for row in best],
            solver_used=used,
            solve_time_s=elapsed,
            problem=problem,
        )

    def _initial_states(self, bqm, initial_state, reads):
        labels = list(bqm.variables)
        row = [initial_state.get(v, 0) for v in labels]
        return dimod.SampleSet.from_samples(([row] * reads, labels), dimod.BINARY, 0)

    def _tabu(self, bqm, initial_state):
        kwargs: Dict[str, Any] = {
            "num_reads": self.num_reads,
            "num_restarts": 0,
            "lower_bound_z": 1,
            "coefficient_z_first": self.tabu_effort,
            "timeout": 2_000,  # safety net only; the iteration budget ends the search
        }

        if self.seed is not None:
            kwargs["seed"] = self.seed
        if initial_state is not None:
            kwargs["initial_states"] = self._initial_states(bqm, initial_state, self.num_reads)
        return TabuSampler().sample(bqm, **kwargs)

    @staticmethod
    def _leap(bqm):
        try:
            from dwave.system import LeapHybridSampler
        except ImportError as exc:
            raise RuntimeError(
                "method='leap' needs the D-Wave Ocean SDK: `pip install dwave-system`, "
                "then configure a Leap API token with `dwave config create`."
            ) from exc
        return LeapHybridSampler().sample(bqm, label="selective-qubo-shuttling")

    def _sa(self, bqm, initial_state):
        kwargs: Dict[str, Any] = {"num_reads": self.num_reads, "num_sweeps": self.sa_num_sweeps}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        sampler = neal.SimulatedAnnealingSampler()
        if initial_state is None:
            return sampler.sample(bqm, **kwargs)
        # Warm reads polish the start state at low temperature;
        # cold reads search globally with the default schedule.
        warm = sampler.sample(
            bqm,
            initial_states=self._initial_states(bqm, initial_state, self.num_reads),
            initial_states_generator="none",
            beta_range=(1.0, 10.0),
            **kwargs,
        )
        return dimod.concatenate([warm, sampler.sample(bqm, **kwargs)])
