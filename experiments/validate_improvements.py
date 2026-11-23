"""Paired local validation after implementation; preserves historical audit evidence."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from research_audit import ROOT
from benchmark_circuits import circuit_for
from benchmark_runner import BenchmarkRunner, make_machine
from bqskit.compiler.passdata import PassData
from congestion_handler import CongestionHandler
from shaw_routing_pass import ShawRoutingPass

OUT = ROOT / 'experiments/results/implementation_validation'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mode = 'incumbent' if '--incumbent' in sys.argv else 'qubo'
    if '--existing' in sys.argv:
        from benchmark_runner import load_qasm_circuit
        rows = []
        for arch in ['linear', 'junction_grid', 'grid', 'grid_dense']:
            for name in ['019_chain_18q.qasm', '177_random_19q.qasm', '184_random_17q.qasm']:
                for hybrid in [False, True]:
                    circuit = load_qasm_circuit(ROOT / 'tests' / name)
                    result = BenchmarkRunner(architecture=arch, handler_options={'planning_mode': mode})._route(name, circuit, hybrid=hybrid)
                    rows.append(dict(architecture=arch, **asdict(result)))
                print(arch, name, 'finished', flush=True)
        (OUT / ('existing_incumbent.json' if mode == 'incumbent' else 'existing.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        assert all(r['success'] for r in rows)
        print('Validated', len(rows), 'existing-corpus routes')
        return
    families = ['ghz_chain', 'qft', 'qaoa_ring', 'qaoa_cubic', 'vqe_brickwork', 'parity_check_proxy']
    rows = []
    for arch in ([] if '--unitaries-only' in sys.argv else ['linear', 'junction_grid']):
        for n in [8, 12]:
            for family in families:
                circuit = circuit_for(family, n)
                for policy in ['baseline', 'raw', 'full', 'staged']:
                    runner = BenchmarkRunner(architecture=arch, handler_options={'staged_effort': 200 if policy == 'staged' else 0}, pass_options={'baseline_guard': policy != 'raw'})
                    result = runner._route(family, circuit, hybrid=policy != 'baseline')
                    rows.append(dict(architecture=arch, policy=policy, **asdict(result)))
                print(arch, n, family, [(r['policy'], r['success'], r['total_shuttle_ops'] + r['swap_count'], round(r['compile_time_s'], 3)) for r in rows[-4:]], flush=True)
                (OUT / 'screen.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    checks = []
    for arch in ['linear', 'junction_grid', 'grid', 'grid_dense']:
        for family in families:
            for enabled in [False, True]:
                circuit = circuit_for(family, 4)
                before = np.asarray(circuit.get_unitary())
                pg, placement = make_machine(arch, 4) or (None, None)
                routing = ShawRoutingPass(pg, placement, CongestionHandler(qubo_enabled=enabled, planning_mode=mode))
                asyncio.run(routing.run(circuit, PassData(circuit)))
                after = np.asarray(circuit.get_unitary())
                overlap = np.vdot(before, after)
                error = float(np.max(np.abs(after - before * overlap / abs(overlap))))
                checks.append(dict(architecture=arch, family=family, hybrid=enabled, error=error, passed=error < 1e-10))
    (OUT / ('unitaries_incumbent.json' if mode == 'incumbent' else 'unitaries.json')).write_text(json.dumps(checks, indent=2), encoding='utf-8')
    assert all(r['success'] for r in rows)
    assert all(r['passed'] for r in checks)
    print('Validated', len(rows), 'routes and', len(checks), 'unitaries')

if __name__ == '__main__':
    main()
