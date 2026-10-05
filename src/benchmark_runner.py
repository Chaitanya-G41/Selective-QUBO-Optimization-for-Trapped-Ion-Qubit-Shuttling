"""Benchmark SHAW routing against the current selective-QUBO integration.

The present CongestionHandler detects QUBO triggers but uses greedy fallback.
Until the solver/decoder is integrated, hybrid_qubo is a trigger-enabled
routing mode, NOT evidence of QUBO optimization or accepted QUBO solutions.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import IdentityGate, SwapGate
from congestion_handler import CongestionHandler
from shaw_routing_pass import ShawRoutingPass

logger = logging.getLogger("shaw_router.benchmark")
TESTS_DIR = ROOT / "tests"
RESULTS_DIR = ROOT / "experiments" / "results"
PLOTS_DIR = ROOT / "experiments" / "plots"


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


@dataclass
class ComparisonResult:
    circuit_name: str
    baseline: RunMetrics
    hybrid: RunMetrics
    shuttle_improvement: float
    swap_improvement: float
    time_overhead: float
    verdict: str


class BenchmarkRunner:
    def __init__(
        self,
        qubo_enabled: bool = True,
        kappa_threshold: float = 0.5,
        rho_threshold: float = 0.3,
    ) -> None:
        self.qubo_enabled = qubo_enabled
        self.kappa_threshold = kappa_threshold
        self.rho_threshold = rho_threshold
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
            )
        else:
            # Both the kappa/rho condition AND independent depth trigger
            # must be disabled for a genuine SHAW-only baseline.
            handler = CongestionHandler(
                kappa_threshold=float("inf"),
                rho_threshold=float("inf"),
                depth_threshold=float("inf"),
            )
        shaw = ShawRoutingPass(congestion_handler=handler)
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
        # Triggers count escalations; accepted counts decoded QUBO paths
        # actually applied (0 when everything fell back to greedy).
        triggers = stats["qubo_triggers"]
        accepted = stats.get("qubo_accepted", 0)
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
        )

    def run_baseline(self, circuit_name: str, circuit: Circuit) -> RunMetrics:
        """Run SHAW with all QUBO triggers disabled."""
        return self._route(circuit_name, circuit, hybrid=False)

    def run_hybrid(self, circuit_name: str, circuit: Circuit) -> RunMetrics:
        """Run trigger-enabled SHAW (selective QUBO with greedy fallback)."""
        return self._route(circuit_name, circuit, hybrid=True)

    def compare(self, baseline: RunMetrics, hybrid: RunMetrics) -> ComparisonResult:
        """Compare runs; verdict describes measured routing, not QUBO efficacy."""
        if baseline.circuit_name != hybrid.circuit_name:
            raise ValueError("Baseline and hybrid must use the same circuit")
        if not baseline.success:
            verdict = "baseline_failed"
            shuttle_improvement = swap_improvement = 0.0
        elif not hybrid.success:
            verdict = "qubo_loses"
            shuttle_improvement = swap_improvement = -1.0
        else:
            shuttle_improvement = self._relative_improvement(
                baseline.total_shuttle_ops, hybrid.total_shuttle_ops
            )
            swap_improvement = self._relative_improvement(
                baseline.swap_count, hybrid.swap_count
            )
            if shuttle_improvement > 0.05:
                verdict = "qubo_wins"
            elif shuttle_improvement < -0.05:
                verdict = "qubo_loses"
            else:
                verdict = "tie"
        time_overhead = (
            hybrid.compile_time_s / baseline.compile_time_s - 1.0
            if baseline.compile_time_s > 0 else 0.0
        )
        hybrid.routing_cost_improvement = shuttle_improvement if (
            baseline.success and hybrid.success
        ) else 0.0
        return ComparisonResult(
            circuit_name=baseline.circuit_name,
            baseline=baseline,
            hybrid=hybrid,
            shuttle_improvement=shuttle_improvement,
            swap_improvement=swap_improvement,
            time_overhead=time_overhead,
            verdict=verdict,
        )

    def run_all(self, max_circuits: Optional[int] = None) -> List[ComparisonResult]:
        """Run the selected QASM circuits and write outputs."""
        if max_circuits is not None and max_circuits < 1:
            raise ValueError("max_circuits must be positive")
        circuits = self.load_all_circuits()
        if max_circuits is not None:
            circuits = circuits[:max_circuits]
        if not circuits:
            raise FileNotFoundError(f"No .qasm files found under {TESTS_DIR}")
        comparisons: List[ComparisonResult] = []
        self.last_baseline_results = []
        for index, (name, circuit) in enumerate(circuits, start=1):
            logger.info("[%d/%d] %s", index, len(circuits), name)
            baseline = self.run_baseline(name, circuit)
            self.last_baseline_results.append(baseline)
            if self.qubo_enabled:
                hybrid = self.run_hybrid(name, circuit)
                comparisons.append(self.compare(baseline, hybrid))
        self.save_csv(comparisons)
        self.plot_results(comparisons)
        self.print_summary(comparisons)
        return comparisons

    def save_csv(self, results: List[ComparisonResult]) -> Path:
        """Export every run metric; baseline-only mode has one row per circuit."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "comparison.csv"
        metric_names = [f.name for f in fields(RunMetrics)]
        columns = (
            ["circuit_name"]
            + [f"baseline_{name}" for name in metric_names if name != "circuit_name"]
            + [f"hybrid_{name}" for name in metric_names if name != "circuit_name"]
            + ["shuttle_improvement", "swap_improvement", "time_overhead", "verdict"]
        )
        rows = []
        if self.qubo_enabled:
            for result in results:
                row = {"circuit_name": result.circuit_name}
                for prefix, metrics in (("baseline", result.baseline), ("hybrid", result.hybrid)):
                    row.update({f"{prefix}_{key}": value for key, value in asdict(metrics).items()
                                if key != "circuit_name"})
                row.update({key: getattr(result, key) for key in
                            ("shuttle_improvement", "swap_improvement", "time_overhead", "verdict")})
                rows.append(row)
        else:
            for baseline in self.last_baseline_results:
                row = {"circuit_name": baseline.circuit_name}
                row.update({f"baseline_{key}": value for key, value in asdict(baseline).items()
                            if key != "circuit_name"})
                rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %d rows to %s", len(rows), path)
        return path

    def plot_results(self, results: List[ComparisonResult]) -> None:
        """Write the comparison plots; baseline-only mode writes time plot only."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        if not results and not self.last_baseline_results:
            return
        baseline_runs = [r.baseline for r in results] if results else self.last_baseline_results
        hybrid_runs = [r.hybrid for r in results]

        if results:
            # Many circuits: show only first 25 in per-circuit bar charts.
            visible = results[:25]
            names = [r.circuit_name for r in visible]
            xs = list(range(len(visible)))
            for attr, filename, ylabel in (
                ("total_shuttle_ops", "shuttling_ops.png", "Shuttle hops"),
                ("swap_count", "swap_count.png", "SWAP operations"),
            ):
                fig, ax = plt.subplots(figsize=(max(9, len(visible) * 0.48), 5))
                ax.bar([x - 0.2 for x in xs], [getattr(r.baseline, attr) for r in visible],
                       width=0.4, label="SHAW baseline")
                ax.bar([x + 0.2 for x in xs], [getattr(r.hybrid, attr) for r in visible],
                       width=0.4, label="Trigger-enabled SHAW")
                ax.set_xticks(xs, names, rotation=75, ha="right")
                ax.set_ylabel(ylabel)
                ax.legend()
                fig.tight_layout()
                fig.savefig(PLOTS_DIR / filename, dpi=160)
                plt.close(fig)

            # The RunMetrics dataclass stores per-circuit means, not individual
            # event values; label these as distributions of CIRCUIT MEANS.
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, attr, title in zip(axes,
                                       ("kappa_mean", "rho_mean", "depth_mean"),
                                       ("Mean κ", "Mean ρ", "Mean d")):
                ax.hist([getattr(r, attr) for r in baseline_runs if r.congestion_events],
                        alpha=0.65, label="Baseline")
                ax.hist([getattr(r, attr) for r in hybrid_runs if r.congestion_events],
                        alpha=0.65, label="Trigger-enabled")
                ax.set_title(title)
                ax.set_xlabel("Per-circuit mean")
                ax.set_ylabel("Circuits")
                ax.legend()
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "congestion_dist.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter([r.qubo_triggers for r in hybrid_runs],
                       [r.qubo_acceptance_rate for r in hybrid_runs])
            ax.set_xlabel("QUBO triggers (not solves)")
            ax.set_ylabel("QUBO acceptance rate")
            ax.set_title("Trigger count vs acceptance")
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "qubo_acceptance.png", dpi=160)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for runs, label in ((baseline_runs, "SHAW baseline"),
                            (hybrid_runs, "Trigger-enabled SHAW")):
            if runs:
                ordered = sorted(runs, key=lambda r: r.num_gates_original)
                ax.plot([r.num_gates_original for r in ordered],
                        [r.compile_time_s for r in ordered], marker="o", label=label)
        ax.set_xlabel("Original circuit gate count")
        ax.set_ylabel("Compile time (seconds)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "compile_time.png", dpi=160)
        plt.close(fig)

    def print_summary(self, results: List[ComparisonResult]) -> None:
        baseline_runs = [r.baseline for r in results] if results else self.last_baseline_results
        print("\n========== Benchmark Summary ==========")
        print(f"Circuits tested: {len(baseline_runs)}")
        print(f"Baseline successes: {sum(r.success for r in baseline_runs)}")
        if not self.qubo_enabled:
            print("Mode: baseline only")
            print("=======================================")
            return
        valid = [r for r in results if r.baseline.success and r.hybrid.success]
        print(f"Both modes successful: {len(valid)}")
        print(f"Baseline failures: {sum(not r.baseline.success for r in results)}")
        print(f"Hybrid failures: {sum(r.baseline.success and not r.hybrid.success for r in results)}")
        for verdict, label in (("qubo_wins", "Trigger-enabled wins"),
                               ("tie", "Ties"), ("qubo_loses", "Trigger-enabled losses")):
            print(f"{label}: {sum(r.verdict == verdict for r in results)}")
        print(f"Mean shuttle improvement: {self._mean([r.shuttle_improvement for r in valid]):+.1%}")
        print(f"Mean SWAP improvement: {self._mean([r.swap_improvement for r in valid]):+.1%}")
        print(f"Mean compile-time overhead: {self._mean([r.time_overhead for r in valid]):+.1%}")
        triggers = sum(r.hybrid.qubo_triggers for r in results)
        accepted = sum(r.hybrid.qubo_accepted for r in results)
        print(f"QUBO triggers: {triggers}; accepted: {accepted}")
        print(f"Acceptance rate: {accepted / triggers if triggers else 0:.1%}")
        print("NOTE: Trigger-enabled mode currently uses greedy fallback, not QUBO solving.")
        print("=======================================")

    @staticmethod
    def load_all_circuits() -> List[tuple]:
        """Load the project's OpenQASM 3 benchmark circuits."""
        import re
        from bqskit.ir.gates import CNOTGate, HGate

        qreg_pat = re.compile(
            r"qubit\s*\[\s*(\d+)\s*\]\s+(\w+)\s*;"
        )
        h_pat = re.compile(
            r"h\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;"
        )
        cx_pat = re.compile(
            r"cx\s+(\w+)\s*\[\s*(\d+)\s*\]\s*,\s*"
            r"(\w+)\s*\[\s*(\d+)\s*\]\s*;"
        )

        circuits = []

        for path in sorted(TESTS_DIR.rglob("*.qasm")):
            try:
                lines = [
                    line.strip()
                    for line in path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]

                if len(lines) < 3:
                    raise ValueError("Incomplete QASM file")

                if lines[0] != "OPENQASM 3.0;":
                    raise ValueError("Expected OpenQASM 3.0")

                if lines[1] != 'include "stdgates.inc";':
                    raise ValueError("Unexpected include statement")

                match = qreg_pat.fullmatch(lines[2])
                if not match:
                    raise ValueError("Invalid qubit declaration")

                num_qudits = int(match.group(1))
                register = match.group(2)
                circuit = Circuit(num_qudits)

                for line in lines[3:]:
                    h_match = h_pat.fullmatch(line)
                    cx_match = cx_pat.fullmatch(line)

                    if h_match:
                        reg = h_match.group(1)
                        q = int(h_match.group(2))

                        if reg != register or q >= num_qudits:
                            raise ValueError(
                                f"Invalid qubit: {line}"
                            )

                        circuit.append_gate(HGate(), (q,))

                    elif cx_match:
                        reg1 = cx_match.group(1)
                        a = int(cx_match.group(2))
                        reg2 = cx_match.group(3)
                        b = int(cx_match.group(4))

                        if (
                            reg1 != register
                            or reg2 != register
                            or a >= num_qudits
                            or b >= num_qudits
                            or a == b
                        ):
                            raise ValueError(
                                f"Invalid CX gate: {line}"
                            )

                        circuit.append_gate(
                            CNOTGate(), (a, b)
                        )

                    else:
                        raise ValueError(
                            f"Unsupported statement: {line}"
                        )

                circuits.append(
                    (
                        str(path.relative_to(TESTS_DIR)),
                        circuit,
                    )
                )

            except Exception as exc:
                logger.warning(
                    "Skipping unreadable QASM file %s: %s",
                    path,
                    exc,
                )

        return circuits

    @staticmethod
    def _count_ops(circuit: Circuit) -> tuple:
        shuttles = sum(isinstance(op.gate, IdentityGate) for op in circuit.operations())
        swaps = sum(isinstance(op.gate, SwapGate) for op in circuit.operations())
        return shuttles, swaps


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SHAW vs trigger-enabled SHAW")
    parser.add_argument("--circuits", type=int, default=None)
    parser.add_argument("--mode", choices=["both", "baseline_only"], default="both")
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    runner = BenchmarkRunner(
        qubo_enabled=args.mode == "both",
        kappa_threshold=args.kappa,
        rho_threshold=args.rho,
    )
    runner.run_all(max_circuits=args.circuits)
