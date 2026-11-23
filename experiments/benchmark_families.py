"""Benchmark structured circuit families with reproducible configuration manifests."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from benchmark_circuits import FAMILIES, circuit_for
from benchmark_runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--sizes', nargs='+', type=int, default=[8, 12])
    parser.add_argument('--architectures', nargs='+', choices=['linear', 'junction_grid', 'grid', 'grid_dense'], default=['linear', 'junction_grid'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--planner', choices=['qubo', 'incumbent'], default='qubo')
    parser.add_argument('--efforts', nargs='+', type=int, default=[4000])
    parser.add_argument('--graph-seed', type=int, default=17)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/results/family_benchmark.json')
    args = parser.parse_args()
    if any(e <= 0 for e in args.efforts):
        parser.error('efforts must be positive')
    manifest = {'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'python': platform.python_version(),
                'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'working_tree_modified': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                'packages': {p: importlib.metadata.version(p) for p in ['bqskit', 'dimod', 'dwave-samplers', 'networkx']},
                'objective': 'physical_operations', 'baseline_guard': True,
                'max_time_horizon': 24, 'max_bqm_variables': 3000, 'staged_effort': 0}
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for arch in args.architectures:
        for n in args.sizes:
            for family in args.families:
                circuit = circuit_for(family, n, graph_seed=args.graph_seed)
                baseline = BenchmarkRunner(architecture=arch)._route(family, circuit, hybrid=False)
                for seed in args.seeds:
                    for effort in args.efforts:
                        runner = BenchmarkRunner(architecture=arch, seed=seed,
                            handler_options={'planning_mode': args.planner, 'solver_options': {'tabu_effort': effort}})
                        hybrid = runner._route(family, circuit, hybrid=True)
                        rows.append({'architecture': arch, 'family': family, 'n': n, 'seed': seed,
                                     'effort': effort, 'baseline': asdict(baseline), 'hybrid': asdict(hybrid)})
                        args.output.write_text(json.dumps({'manifest': manifest, 'rows': rows}, indent=2), encoding='utf-8')
                print(arch, n, family, 'finished', flush=True)
    failures = sum(not r['baseline']['success'] or not r['hybrid']['success'] for r in rows)
    regressions = sum(r['baseline']['success'] and r['hybrid']['success'] and
        r['hybrid']['total_shuttle_ops'] + r['hybrid']['swap_count'] > r['baseline']['total_shuttle_ops'] + r['baseline']['swap_count'] for r in rows)
    print(f'{len(rows)} pairs; {failures} failures; {regressions} regressions; output: {args.output}')
    if failures or regressions:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
