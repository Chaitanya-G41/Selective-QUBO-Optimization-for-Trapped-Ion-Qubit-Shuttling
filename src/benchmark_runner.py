"""
benchmark_runner.py
===================
Benchmarks plain SHAW (baseline) against SHAW with selective QUBO on the
OpenQASM circuits in tests/.

    python src/benchmark_runner.py                    # linear chain, all circuits
    python src/benchmark_runner.py --arch junction_grid  # traps between X-junctions
    python src/benchmark_runner.py --jobs 8           # parallel workers
    python src/benchmark_runner.py --seeds 5          # solver-seed statistics
    python src/benchmark_runner.py --fast             # learned trigger (faster)
    python src/benchmark_runner.py --objective time   # QUBO minimises shuttling time

Cost = physical operations (shuttles + ion swaps). "Best of both" is the
portfolio strategy that compiles both ways and keeps the cheaper circuit,
which can never be worse than the baseline.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import math
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate, IdentityGate

from congestion_handler import CongestionHandler
from shaw_routing_pass import ShawRoutingPass
from walk_planner import TIME_COSTS_US

logger = logging.getLogger("shaw_router.benchmark")
TESTS_DIR = ROOT / "tests"
RESULTS_DIR = ROOT / "experiments" / "results"
PLOTS_DIR = ROOT / "experiments" / "plots"
ARCHITECTURES = ("linear", "junction_grid", "grid", "grid_dense")


@dataclass
class RunMetrics:
    circuit_name: str
    mode: str
    num_qudits: int
    num_gates_original: int
    total_shuttle_ops: int
    swap_count: int
    compile_time_s: float
    congestion_events: int
    qubo_triggers: int
    qubo_accepted: int
    qubo_acceptance_rate: float
    kappa_mean: float
    rho_mean: float
    depth_mean: float
    routing_cost_improvement: float
    success: bool
    error_msg: str = ""
    movement_time_us: float = 0.0   # shuttling time under TIME_COSTS_US
    depth: int = 0                  # routed-circuit depth (markers included)

    @property
    def total_ops(self) -> int:
        return self.total_shuttle_ops + self.swap_count


@dataclass
class ComparisonResult:
    circuit_name: str
    baseline: RunMetrics
    hybrid: RunMetrics
    shuttle_improvement: float
    swap_improvement: float
    time_overhead: float
    verdict: str

    @property
    def best_of_both_ops(self) -> int:
        if not self.hybrid.success:
            return self.baseline.total_ops
        return min(self.baseline.total_ops, self.hybrid.total_ops)


# ---------------------------------------------------------------------------
# Circuits and machines
# ---------------------------------------------------------------------------

_QREG = re.compile(r"qubit\s*\[\s*(\d+)\s*\]\s+(\w+)\s*;")
_H = re.compile(r"h\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;")
_CX = re.compile(r"cx\s+(\w+)\s*\[\s*(\d+)\s*\]\s*,\s*(\w+)\s*\[\s*(\d+)\s*\]\s*;")


def load_qasm_circuit(path: Path) -> Circuit:
    """Parse one OpenQASM 3.0 file containing only h / cx gates."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 3 or lines[0] != "OPENQASM 3.0;" or lines[1] != 'include "stdgates.inc";':
        raise ValueError("expected an OpenQASM 3.0 header with stdgates.inc")
    match = _QREG.fullmatch(lines[2])
    if not match:
        raise ValueError("invalid qubit declaration")
    n, reg = int(match.group(1)), match.group(2)
    circuit = Circuit(n)
    for line in lines[3:]:
        if (m := _H.fullmatch(line)) and m.group(1) == reg and int(m.group(2)) < n:
            circuit.append_gate(HGate(), (int(m.group(2)),))
        elif ((m := _CX.fullmatch(line)) and m.group(1) == reg == m.group(3)
              and int(m.group(2)) < n and int(m.group(4)) < n and m.group(2) != m.group(4)):
            circuit.append_gate(CNOTGate(), (int(m.group(2)), int(m.group(4))))
        else:
            raise ValueError(f"unsupported statement: {line}")
    return circuit


def make_machine(architecture: str, num_qudits: int) -> Optional[Tuple[Any, Any]]:
    """
    (position_graph, placement) for `architecture`, one home ion per trap.
    None for "linear": ShawRoutingPass builds its default linear chain.
    "junction_grid" puts linear capacity-2 traps on the edges of a square
    grid of X-junctions (see build_junction_grid_qccd), one home ion each.
    "grid" is the near-square qccd grid (traps joined by channels in a
    row and junctions between rows), capacity 2 per trap.
    "grid_dense" packs two home ions into each capacity-3 trap, so ions
    passing through a trap genuinely collide.
    """
    if architecture == "linear":
        return None
    if architecture == "junction_grid":
        from position_graph import Placement, build_junction_grid_qccd

        side = 2  # smallest square junction grid with >= num_qudits traps
        while 2 * side * (side - 1) < num_qudits:
            side += 1
        pg = build_junction_grid_qccd(side, side, trap_capacity=2)
        placement = Placement(pg)
        traps = list(pg._trap_slots)
        for q in range(num_qudits):
            placement.place(q, pg.slots_of(traps[q])[0])
        return pg, placement
    if architecture in ("grid", "grid_dense"):
        from qccd.position_graph import GridArchitecture
        from qccd_adapter import build_from_qccd

        per_trap, capacity = (2, 3) if architecture == "grid_dense" else (1, 2)
        traps = math.ceil(num_qudits / per_trap)
        rows = math.ceil(math.sqrt(traps))
        cols = math.ceil(traps / rows)
        pg, placement = build_from_qccd(GridArchitecture.build(rows, cols, trap_capacity=capacity))
        trap_ids = [n for n, attrs in pg.graph.nodes(data=True) if attrs.get("trap_id") is not None]
        trap_ids = list(dict.fromkeys(pg.graph.nodes[n]["trap_id"] for n in trap_ids))
        for q in range(num_qudits):
            placement.place(q, pg.slots_of(trap_ids[q // per_trap])[q % per_trap])
        return pg, placement
    raise ValueError(f"unknown architecture {architecture!r}; use one of {ARCHITECTURES}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(
        self,
        qubo_enabled: bool = True,
        kappa_threshold: float = 0.0,
        rho_threshold: float = 0.0,
        architecture: str = "linear",
        handler_options: Optional[Dict[str, Any]] = None,
        pass_options: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        jobs: int = 1,
    ) -> None:
        self.qubo_enabled = qubo_enabled
        self.kappa_threshold = kappa_threshold
        self.rho_threshold = rho_threshold
        self.architecture = architecture
        self.handler_options = handler_options or {}
        self.pass_options = pass_options or {}
        self.seed = seed
        self.jobs = jobs
        self.last_baseline_results: List[RunMetrics] = []

    @staticmethod
    def _mean(values) -> float:
        return float(statistics.mean(values)) if values else 0.0

    @staticmethod
    def _relative_improvement(baseline: int, hybrid: int) -> float:
        if baseline == 0:
            return 0.0 if hybrid == 0 else -1.0
        return (baseline - hybrid) / baseline

    def _route(self, circuit_name: str, circuit: Circuit, *, hybrid: bool) -> RunMetrics:
        if hybrid:
            handler = CongestionHandler(
                kappa_threshold=self.kappa_threshold,
                rho_threshold=self.rho_threshold,
                seed=self.seed,
                **self.handler_options,
            )
        else:
            handler = CongestionHandler(qubo_enabled=False)
        machine = make_machine(self.architecture, circuit.num_qudits)
        pg, placement = machine if machine else (None, None)
        shaw = ShawRoutingPass(position_graph=pg, placement=placement, congestion_handler=handler,
                               **(self.pass_options if hybrid else {}))

        routed = circuit.copy()  # ShawRoutingPass.run mutates its input.
        started = time.perf_counter()
        success, error_msg = True, ""
        try:
            asyncio.run(shaw.run(routed, PassData(routed)))
        except RuntimeError as exc:
            success, error_msg = False, str(exc)
            logger.warning("Routing failed for %s: %s", circuit_name, exc)
        elapsed = time.perf_counter() - started

        shuttles, swaps = self._count_ops(routed) if success else (0, 0)
        stats = handler.stats
        triggers, accepted = stats["qubo_triggers"], stats["qubo_executed"]
        return RunMetrics(
            circuit_name=circuit_name,
            mode="hybrid_qubo" if hybrid else "baseline",
            num_qudits=circuit.num_qudits,
            num_gates_original=circuit.num_operations,
            total_shuttle_ops=shuttles,
            swap_count=swaps,
            compile_time_s=elapsed,
            congestion_events=stats["blocked_events"],
            qubo_triggers=triggers,
            qubo_accepted=accepted,
            qubo_acceptance_rate=accepted / triggers if triggers else 0.0,
            kappa_mean=self._mean(stats["kappa_values"]),
            rho_mean=self._mean(stats["rho_values"]),
            depth_mean=self._mean(stats["depth_values"]),
            routing_cost_improvement=0.0,
            success=success,
            error_msg=error_msg,
            movement_time_us=shaw.movement_time_us if success else 0.0,
            depth=routed.depth if success else 0,
        )

    def run_baseline(self, circuit_name: str, circuit: Circuit) -> RunMetrics:
        """Plain SHAW: QUBO disabled."""
        return self._route(circuit_name, circuit, hybrid=False)

    def run_hybrid(self, circuit_name: str, circuit: Circuit) -> RunMetrics:
        """SHAW with selective QUBO."""
        return self._route(circuit_name, circuit, hybrid=True)

    def compare(self, baseline: RunMetrics, hybrid: RunMetrics) -> ComparisonResult:
        """Verdict on total physical operations (shuttles + ion swaps)."""
        if baseline.circuit_name != hybrid.circuit_name:
            raise ValueError("Baseline and hybrid must use the same circuit")
        if not baseline.success:
            verdict = "baseline_failed"
            shuttle_improvement = swap_improvement = ops_improvement = 0.0
        elif not hybrid.success:
            verdict = "qubo_loses"
            shuttle_improvement = swap_improvement = ops_improvement = -1.0
        else:
            shuttle_improvement = self._relative_improvement(
                baseline.total_shuttle_ops, hybrid.total_shuttle_ops)
            swap_improvement = self._relative_improvement(baseline.swap_count, hybrid.swap_count)
            ops_improvement = self._relative_improvement(baseline.total_ops, hybrid.total_ops)
            if ops_improvement > 0.05:
                verdict = "qubo_wins"
            elif ops_improvement < -0.05:
                verdict = "qubo_loses"
            else:
                verdict = "tie"
        hybrid.routing_cost_improvement = ops_improvement
        time_overhead = (hybrid.compile_time_s / baseline.compile_time_s - 1.0
                         if baseline.compile_time_s > 0 else 0.0)
        return ComparisonResult(
            circuit_name=baseline.circuit_name,
            baseline=baseline,
            hybrid=hybrid,
            shuttle_improvement=shuttle_improvement,
            swap_improvement=swap_improvement,
            time_overhead=time_overhead,
            verdict=verdict,
        )

    def _run_one(self, name: str) -> Tuple[RunMetrics, Optional[ComparisonResult]]:
        circuit = load_qasm_circuit(TESTS_DIR / name)
        baseline = self.run_baseline(name, circuit)
        if not self.qubo_enabled:
            return baseline, None
        return baseline, self.compare(baseline, self.run_hybrid(name, circuit))

    def run_all(
        self,
        max_circuits: Optional[int] = None,
        write_outputs: bool = True,
    ) -> List[ComparisonResult]:
        """Run the QASM circuits (in parallel with jobs > 1) and write outputs."""
        if max_circuits is not None and max_circuits < 1:
            raise ValueError("max_circuits must be positive")
        names = self.circuit_names()[:max_circuits]
        if not names:
            raise FileNotFoundError(f"No .qasm files found under {TESTS_DIR}")

        if self.jobs > 1:
            with ProcessPoolExecutor(self.jobs) as pool:
                outcomes = list(pool.map(self._run_one, names))
        else:
            outcomes = []
            for index, name in enumerate(names, start=1):
                logger.info("[%d/%d] %s", index, len(names), name)
                outcomes.append(self._run_one(name))

        self.last_baseline_results = [b for b, _ in outcomes]
        comparisons = [c for _, c in outcomes if c is not None]
        if write_outputs:
            self.save_csv(comparisons)
            self.plot_results(comparisons)
            self.print_summary(comparisons)
        return comparisons

    # -----------------------------------------------------------------------
    # Outputs
    # -----------------------------------------------------------------------

    def _suffix(self) -> str:
        return "" if self.architecture == "linear" else f"_{self.architecture}"

    def save_csv(self, results: List[ComparisonResult]) -> Path:
        """One row per circuit with every metric of both runs."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"comparison{self._suffix()}.csv"
        metric_names = [f.name for f in fields(RunMetrics) if f.name != "circuit_name"]
        extra = ["shuttle_improvement", "swap_improvement", "time_overhead", "verdict",
                 "best_of_both_ops"]
        columns = (["circuit_name"] + [f"baseline_{m}" for m in metric_names]
                   + ([f"hybrid_{m}" for m in metric_names] + extra if self.qubo_enabled else []))
        rows = []
        if self.qubo_enabled:
            for result in results:
                row = {"circuit_name": result.circuit_name}
                for prefix, metrics in (("baseline", result.baseline), ("hybrid", result.hybrid)):
                    row.update({f"{prefix}_{k}": v for k, v in asdict(metrics).items()
                                if k != "circuit_name"})
                row.update({k: getattr(result, k) for k in extra})
                rows.append(row)
        else:
            for baseline in self.last_baseline_results:
                row = {"circuit_name": baseline.circuit_name}
                row.update({f"baseline_{k}": v for k, v in asdict(baseline).items()
                            if k != "circuit_name"})
                rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %d rows to %s", len(rows), path)
        return path

    def plot_results(self, results: List[ComparisonResult]) -> None:
        """Per-family operation counts, per-circuit change, and compile time."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not results:
            return
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = self._suffix()

        families: Dict[str, List[ComparisonResult]] = {}
        for r in results:
            families.setdefault(circuit_family(r.circuit_name), []).append(r)
        names = sorted(families)
        xs = range(len(names))
        fig, ax = plt.subplots(figsize=(9, 5))
        for offset, label, value in (
            (-0.27, "SHAW baseline", lambda r: r.baseline.total_ops),
            (0.0, "Selective QUBO", lambda r: r.hybrid.total_ops),
            (0.27, "Best of both", lambda r: r.best_of_both_ops),
        ):
            ax.bar([x + offset for x in xs], [sum(map(value, families[f])) for f in names],
                   width=0.27, label=label)
        ax.set_xticks(list(xs), names)
        ax.set_ylabel("Physical operations (shuttles + ion swaps)")
        ax.set_title(f"Operations per circuit family ({self.architecture})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"ops_by_family{suffix}.png", dpi=160)
        plt.close(fig)

        ordered = sorted(results, key=lambda r: r.hybrid.routing_cost_improvement)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(range(len(ordered)), [100 * r.hybrid.routing_cost_improvement for r in ordered],
               color=["#c0392b" if r.hybrid.routing_cost_improvement < 0 else "#2e86c1"
                      for r in ordered])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Circuits (sorted)")
        ax.set_ylabel("Fewer operations than baseline (%)")
        ax.set_title(f"Per-circuit change with selective QUBO ({self.architecture})")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"per_circuit_change{suffix}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([r.baseline.num_gates_original for r in results],
                   [r.baseline.compile_time_s for r in results], s=12, label="SHAW baseline")
        ax.scatter([r.hybrid.num_gates_original for r in results],
                   [r.hybrid.compile_time_s for r in results], s=12, label="Selective QUBO")
        ax.set_xlabel("Original circuit gate count")
        ax.set_ylabel("Compile time (seconds)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"compile_time{suffix}.png", dpi=160)
        plt.close(fig)

    def print_summary(self, results: List[ComparisonResult]) -> None:
        baseline_runs = [r.baseline for r in results] if results else self.last_baseline_results
        print(f"\n========== Benchmark Summary ({self.architecture}) ==========")
        print(f"Circuits tested: {len(baseline_runs)}")
        print(f"Baseline successes: {sum(r.success for r in baseline_runs)}")
        if not self.qubo_enabled:
            print("Mode: baseline only")
            return
        valid = [r for r in results if r.baseline.success and r.hybrid.success]
        print(f"Hybrid failures: {sum(r.baseline.success and not r.hybrid.success for r in results)}")
        better = sum(r.hybrid.total_ops < r.baseline.total_ops for r in valid)
        worse = sum(r.hybrid.total_ops > r.baseline.total_ops for r in valid)
        print(f"Circuits better / equal / worse: {better} / {len(valid) - better - worse} / {worse}")
        base_ops = sum(r.baseline.total_ops for r in valid)
        qubo_ops = sum(r.hybrid.total_ops for r in valid)
        best_ops = sum(r.best_of_both_ops for r in valid)
        print(f"Total physical ops: baseline {base_ops}, selective QUBO {qubo_ops} "
              f"({self._relative_improvement(base_ops, qubo_ops):+.1%}), best of both {best_ops} "
              f"({self._relative_improvement(base_ops, best_ops):+.1%})")
        base_time = sum(r.baseline.movement_time_us for r in valid)
        qubo_time = sum(r.hybrid.movement_time_us for r in valid)
        base_depth = sum(r.baseline.depth for r in valid)
        qubo_depth = sum(r.hybrid.depth for r in valid)
        print(f"Shuttling time: baseline {base_time / 1e3:.1f} ms, selective QUBO "
              f"{qubo_time / 1e3:.1f} ms ({(base_time - qubo_time) / base_time:+.1%}); "
              f"circuit depth {base_depth} -> {qubo_depth} "
              f"({(base_depth - qubo_depth) / base_depth:+.1%})")
        print(f"Mean per-circuit improvement: "
              f"{self._mean([r.hybrid.routing_cost_improvement for r in valid]):+.1%}")
        base_t = sum(r.baseline.compile_time_s for r in valid)
        qubo_t = sum(r.hybrid.compile_time_s for r in valid)
        print(f"Total compile time: baseline {base_t:.1f}s, selective QUBO {qubo_t:.1f}s")
        triggers = sum(r.hybrid.qubo_triggers for r in results)
        accepted = sum(r.hybrid.qubo_accepted for r in results)
        print(f"Blocked paths considered: {triggers}; QUBO plans accepted: {accepted}")

    # -----------------------------------------------------------------------
    # Circuit discovery
    # -----------------------------------------------------------------------

    @staticmethod
    def circuit_names() -> List[str]:
        return [str(p.relative_to(TESTS_DIR)) for p in sorted(TESTS_DIR.rglob("*.qasm"))]

    @classmethod
    def load_all_circuits(cls) -> List[tuple]:
        """[(name, Circuit)] for every readable QASM file in tests/."""
        circuits = []
        for name in cls.circuit_names():
            try:
                circuits.append((name, load_qasm_circuit(TESTS_DIR / name)))
            except ValueError as exc:
                logger.warning("Skipping unreadable QASM file %s: %s", name, exc)
        return circuits

    @staticmethod
    def _count_ops(circuit: Circuit) -> tuple:
        """(shuttles, ion swaps): 1- and 2-qudit identity markers."""
        markers = [op.gate.num_qudits for op in circuit.operations()
                   if isinstance(op.gate, IdentityGate)]
        return markers.count(1), markers.count(2)


def circuit_family(name: str) -> str:
    match = re.search(r"\d+_([a-z_]+)_\d+q", name)
    return match.group(1) if match else "other"


def seed_study(seeds: int, **runner_kwargs) -> None:
    """Total-ops improvement over several solver seeds (mean +- stdev)."""
    totals = []
    for seed in range(seeds):
        runner = BenchmarkRunner(seed=seed, **runner_kwargs)
        results = runner.run_all(write_outputs=False)
        valid = [r for r in results if r.baseline.success and r.hybrid.success]
        base = sum(r.baseline.total_ops for r in valid)
        qubo = sum(r.hybrid.total_ops for r in valid)
        totals.append(BenchmarkRunner._relative_improvement(base, qubo))
        print(f"seed {seed}: {totals[-1]:+.2%} ({base} -> {qubo})")
    spread = statistics.stdev(totals) if len(totals) > 1 else 0.0
    print(f"mean {statistics.mean(totals):+.2%}, stdev {spread:.2%} over {seeds} seeds")


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SHAW vs SHAW + selective QUBO")
    parser.add_argument("--circuits", type=int, default=None)
    parser.add_argument("--mode", choices=["both", "baseline_only"], default="both")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="linear")
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=0,
                        help="run a solver-seed study instead of a single benchmark")
    parser.add_argument("--fast", action="store_true",
                        help="learned trigger: skip low-slack windows (much faster)")
    parser.add_argument("--objective", choices=["ops", "time"], default="ops",
                        help="what the QUBO minimises: operation count or shuttling time")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()
    handler_options: Dict[str, Any] = {"learned_trigger": args.fast}
    if args.objective == "time":
        handler_options["cost_model"] = TIME_COSTS_US
    kwargs = dict(qubo_enabled=args.mode == "both", kappa_threshold=args.kappa,
                  rho_threshold=args.rho, architecture=args.arch, jobs=args.jobs,
                  handler_options=handler_options)
    if args.seeds:
        seed_study(args.seeds, **kwargs)
    else:
        BenchmarkRunner(**kwargs).run_all(max_circuits=args.circuits)
