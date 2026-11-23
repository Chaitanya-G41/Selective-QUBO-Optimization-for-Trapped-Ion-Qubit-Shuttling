"""Attribution control: retain only the deterministic warm-start candidate."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from benchmark_circuits import FAMILIES, circuit_for
from benchmark_runner import BenchmarkRunner
from qubo_solver import QUBOSolver, QUBOSolution


def incumbent(self, problem, initial_state=None):
    sample = problem.complete(initial_state or {})
    return QUBOSolution([sample], [float(problem.bqm.energy(sample))], 'incumbent_only', 0., problem)


def main():
    rows = []
    with patch.object(QUBOSolver, 'solve', incumbent):
        for arch in ['linear', 'junction_grid']:
            for n in [8, 12]:
                for family in FAMILIES:
                    result = BenchmarkRunner(architecture=arch)._route(family, circuit_for(family, n), hybrid=True)
                    rows.append(dict(architecture=arch, family=family, n=n, **asdict(result)))
    out = ROOT / 'experiments/results/implementation_validation/incumbent_control.json'
    out.write_text(json.dumps({'description': 'No tabu/annealing; same model and acceptance, only warm incumbent returned', 'rows': rows}, indent=2), encoding='utf-8')
    assert all(r['success'] for r in rows)
    print(len(rows), 'incumbent-only control routes')

if __name__ == '__main__':
    main()
