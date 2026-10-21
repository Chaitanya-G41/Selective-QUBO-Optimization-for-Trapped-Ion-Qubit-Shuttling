"""
learn_trigger.py
================
Can a cheap learned rule predict which QUBO windows are worth solving?

Logs, for every window the router solves, features available *before*
solving (window size, ions, path length, blockers, greedy slack) and
whether the QUBO beat the greedy walk. Trains a shallow decision tree on
half of the circuits (even indices) and reports, on the other half, how
many solves it would skip and how many improvements it would lose.

    python experiments/learn_trigger.py --arch linear
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import benchmark_runner as br  # noqa: E402
import congestion_handler as ch  # noqa: E402
from walk_planner import ops_cost, walk_ops  # noqa: E402

FEATURES = ["ions", "window_nodes", "path_hops", "blocked", "greedy_cost", "slack"]


def collect(arch: str, names):
    rows = []
    original = ch.CongestionHandler._solve_window

    def logged(self, chunk, board, pg, hints):
        mover = board.occupant_of(chunk[0])
        greedy = walk_ops(pg, board, chunk, mover)
        greedy_cost = ops_cost(pg, board, greedy, self.costs)
        solved_before = self.stats["qubo_solved"]
        ops = original(self, chunk, board, pg, hints)
        if self.stats["qubo_solved"] > solved_before:  # a QUBO was actually solved
            window = self.extract_window(chunk, self._detect_blockages(chunk, board), board, pg)
            rows.append({
                "ions": len(window.active_ions),
                "window_nodes": len(window.window_nodes),
                "path_hops": len(chunk) - 1,
                "blocked": len(window.blocked_path),
                "greedy_cost": greedy_cost,
                "slack": greedy_cost - self._lower_bound(pg, chunk[0], window.goal_nodes),
                "improved": ops_cost(pg, board, ops, self.costs) < greedy_cost - 1e-9,
            })
        return ops

    ch.CongestionHandler._solve_window = logged
    runner = br.BenchmarkRunner(architecture=arch)
    for name in names:
        runner.run_hybrid(name, br.load_qasm_circuit(br.TESTS_DIR / name))
    ch.CongestionHandler._solve_window = original
    return rows


def main() -> None:
    from sklearn.tree import DecisionTreeClassifier, export_text

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--arch", choices=br.ARCHITECTURES, default="linear")
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    names = br.BenchmarkRunner.circuit_names()
    train, test = collect(args.arch, names[::2]), collect(args.arch, names[1::2])
    X = lambda rows: [[r[f] for f in FEATURES] for r in rows]
    y = lambda rows: [r["improved"] for r in rows]

    model = DecisionTreeClassifier(max_depth=args.depth, class_weight="balanced", random_state=0)
    model.fit(X(train), y(train))
    print(export_text(model, feature_names=FEATURES))

    keep = model.predict(X(test))
    improved = y(test)
    skipped = sum(not k for k in keep)
    lost = sum(i and not k for i, k in zip(improved, keep))
    print(f"train solves: {len(train)} ({sum(y(train))} improved)")
    print(f"test solves:  {len(test)} ({sum(improved)} improved)")
    print(f"rule skips {skipped}/{len(test)} solves ({skipped / len(test):.0%}), "
          f"losing {lost}/{sum(improved)} improvements")


if __name__ == "__main__":
    main()
