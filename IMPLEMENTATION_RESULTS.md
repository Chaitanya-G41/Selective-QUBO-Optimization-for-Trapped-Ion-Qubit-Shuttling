# Implementation results: legal routing, complete-route guard, and latency controls

Date: 23 November 2026. Changes are local and uncommitted; nothing was pushed. This implements the supported correctness and performance changes from [the research plan](SELECTIVE_QUBO_RESEARCH_PLAN.md). It does not implement every speculative research direction.

## What changed

- **Legal physical routing.** Both policies now clear occupied transport positions by moving blockers into vacancies. Occupied exchanges require an in-trap swap edge. Saturated corridors can require retreating before passing an ion; a bounded A* fallback uses interchangeable blocker states and expands its search radius from 4 to 6 to 8. Native placement, adapter placement, and scratch-board replay all validate moves. Entire candidate plans are checked before execution.
- **Whole-circuit protection.** Selective routing compiles an independent baseline and hybrid candidate, compares their complete physical cost, and replays only the chosen trace onto the caller's placement. Baseline wins ties. This prevents local improvements from making the final route more expensive. Both compilations, copying, validation, and replay count toward compile latency. The guarantee applies to the configured cost model when the baseline succeeds, not simultaneously to operations, movement time, circuit depth, and latency.
- **Aligned costs.** The QUBO corrects the two individual hop costs of an occupied in-trap exchange to the cost of one physical ion swap. Unit and weighted-cost tests check agreement with scheduled operations, apart from explicit terminal penalties. Physical ion swaps are not logical three-CX SWAP gates. The default 300 microsecond physical-swap time is explicitly a modeling assumption. Reported movement time is summed movement work, not hardware makespan.
- **Correct congestion and validation.** Recursive blockage depth now traverses occupied chains. The decoder checks final trap occupancy and that original goal-trap residents remain in the goal trap. Gates with more than two operands require decomposition. The dashboard uses the backend's strict QASM loader instead of silently ignoring unknown statements.
- **Better incumbents.** Valid goal-feasible prefixes can stop when the mover reaches the destination trap. The sampler cannot silently discard the warm incumbent. Vacate-and-enter actions share one model step and are serialized legally by the decoder, avoiding artificial horizon growth. Windows whose incumbent leaves the local region are skipped conservatively.
- **Better latency controls.** One BFS through free positions replaces repeated k-shortest-path enumeration. Default limits skip windows above horizon 24 or BQMs above 3,000 variables, including auxiliaries. The variable limit is applied after construction; it does not eliminate construction time. Build, solve, decode, model size, and stage timings are recorded. The operation-trained trigger is not applied to microsecond objectives.
- **Dynamic layout checks.** Acceptance resolves both partners' identities against each candidate board rather than comparing to stale partner positions. The QUBO terminal objective still uses snapshot position hints; it is an approximation. The layout check is aggregate, not a per-ion nonregression proof.
- **Configuration and visibility.** Dashboard solver choice, effort, and depth settings now reach the live compiler. Metrics expose the selected complete route and planning mode. The router exposes its executable movement trace and portfolio candidate costs.

## Two available policies

**Full QUBO search** remains the default: `CongestionHandler(planning_mode="qubo")`. It retains the original default tabu effort of 4,000, with optional effort 200 for a lower-latency search. A short-then-full solver stage is implemented as `staged_effort=200`, but disabled by default: it increased total time without improving the tested outputs.

**Fast greedy improvement** is `CongestionHandler(planning_mode="incumbent")`, or **Fast greedy improvement** in the dashboard. It constructs a settled, gate-feasible greedy prefix directly, without building a BQM or calling a sampler. It retains the same aggregate layout acceptance and complete-route safeguard. This is a deterministic heuristic improvement, not evidence of QUBO superiority.

`ShawRoutingPass(baseline_guard=False)` disables the complete-route comparison for attribution experiments. Raw selective routing can regress, so this is not the recommended application setting. The historical CSV mode name `hybrid_qubo` is retained for compatibility; use `planning_mode` to distinguish full QUBO from fast incumbent routing. Build/solve counters distinguish sampler activity from accepted deterministic plans.

## Final measured screen

Six families, 8 and 12 qubits, linear and junction-grid architectures: 24 paired workloads per policy. Solver seed 0, cubic-QAOA graph seed 17. Inputs and effective configuration are saved with the results. These are single-machine screening runs, not confidence intervals or a general performance guarantee.

| Policy | Total physical operations | Saving against corrected baseline | Total compile seconds |
|---|---:|---:|---:|
| Corrected baseline | 9,593 | — | 3.45 in the fast-policy screen |
| Guarded full QUBO, effort 4,000 | 8,997 | 6.21% | 46.81 |
| Guarded full QUBO, effort 200 | 8,997 | 6.21% | 34.92 |
| Guarded fast incumbent | 8,922 | 6.99% | 9.96 |

Fast incumbent was approximately **4.7 times faster than guarded full QUBO** on this screen. Effort 200 reduced full-search time by approximately **25%** while retaining its measured savings. These comparisons include the portfolio safeguard. Even fast routing remains slower than compiling only the baseline; the extra time purchases the measured operation savings. Runs occurred separately and timings have normal machine noise; repeated, randomized-order timing studies remain necessary for a stronger latency claim.

Full QUBO improved 17 of 24 workloads and tied the rest after the safeguard. Fast mode also had no failures or regressions. Fast mode can outperform full mode because it is not restricted by QUBO horizon, model-size, or representability filters; these settings do not form a monotonic quality ladder.

Full-QUBO family savings, aggregated over both sizes:

| Family | Linear | Junction grid |
|---|---:|---:|
| GHZ chain | 34.09% | 1.92% |
| QFT | 0.00% | 2.53% |
| Ring QAOA | 10.63% | 22.39% |
| Cubic QAOA | 0.75% | 2.83% |
| VQE brickwork | 22.32% | 7.52% |
| Parity interaction proxy | 3.26% | 6.58% |

VQE brickwork and ring QAOA are useful next workloads. QFT on the linear machine did not improve after the complete-route guard. Cubic QAOA is graph-dependent. The VQE circuit is a generic ansatz, not a chemistry simulation; the parity workload has no measurement/reset and is not a full QEC cycle. QFT omits final output reversal.

### Attribution matters

A separate control retained the QUBO warm incumbent but disabled stochastic search. Its total was **8,998 operations**, versus **8,997** with full tabu search. Thus almost all full-policy improvement in this screen comes from incumbent/prefix improvements and route selection, with only one further operation gained by tabu. The control still constructs the model, unlike the new fast application mode. It would be misleading to present the 6.21% total as an isolated benefit of QUBO search.

Original baseline routes allowed illegal transport exchanges. Their old operation counts and timings are not valid comparators for the corrected implementation. There is no claim here that the new full search is faster than the old physically invalid version, nor a claim of quantum advantage.

## Verification completed

- **84 tests passed**, including new physical legality, saturated-corridor retreat, exact swap pricing under two cost models, recursive depth, dynamic partner distance, external-placement replay, chosen time objective, model budget, warm-incumbent preservation, unsupported gate arity, and no-BQM fast-mode tests.
- **400 baseline routes passed:** all 200 existing QASM circuits on linear and junction-grid machines, with physical-operation checks active. One initial radius-limited failure was repaired before this final run.
- **48 structured pairs passed:** full QUBO at efforts 4,000 and 200, each compared with baseline, with no failures or regressions.
- **24 additional fast-policy pairs passed**, with no failures or regressions.
- **48 existing-corpus route checks passed:** baseline and hybrid on three 17–19 qubit circuits across all four architectures, for full and fast policies. These include `177_random_19q.qasm`, which exposed the corridor-radius issue.
- **96 small unitary checks passed:** six families, four architectures, baseline/hybrid, separately for full and fast policy. Phase-aligned matrix error is checked directly. Physical legality is checked independently; unitary equality alone cannot validate identity-marker routing.
- Python compilation and `git diff --check` passed. The frontend module compiles; an interactive browser walkthrough was not performed.

The final baseline corpus check covers all 200 inputs, but the complete hybrid corpus has not been rerun across multiple seeds. Statistical family/placement holdouts and hardware calibration remain open validation work. Local corridor search is bounded and may still reject a feasible arrangement outside its search budget; a recorded failure is not a proof of physical infeasibility.

## Reproduce

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m pytest -q

# Structured families with full QUBO; defaults include three solver seeds.
.venv/Scripts/python.exe experiments/benchmark_families.py --efforts 4000 200

# Low-latency deterministic policy, explicitly labeled in the manifest.
.venv/Scripts/python.exe experiments/benchmark_families.py --planner incumbent --seeds 0

# Existing QASM corpus CLI; these commands write normal benchmark outputs.
.venv/Scripts/python.exe src/benchmark_runner.py --planner incumbent --arch junction_grid
.venv/Scripts/python.exe src/benchmark_runner.py --tabu-effort 200 --arch junction_grid

# Larger-circuit and independent semantic checks.
.venv/Scripts/python.exe experiments/validate_improvements.py --existing
.venv/Scripts/python.exe experiments/validate_improvements.py --existing --incumbent
.venv/Scripts/python.exe experiments/validate_improvements.py --unitaries-only
.venv/Scripts/python.exe experiments/validate_improvements.py --unitaries-only --incumbent

# Separate stochastic-search attribution control.
.venv/Scripts/python.exe experiments/incumbent_control.py
```

Final evidence is in [experiments/results/implementation_validation](experiments/results/implementation_validation): `final_family_benchmark.json`, `fast_incumbent_benchmark.json`, `incumbent_control.json`, `corpus_baseline.json`, `existing.json`, `existing_incumbent.json`, `unitaries.json`, and `unitaries_incumbent.json`. Other `screen`, `development`, and `effort_sweep` files are intermediate measurements and are superseded by the final policy comparisons. Historical research outputs were preserved.

Path-candidate QUBOs, learned admission, initial-placement optimization, and resource-aware makespan scheduling remain future research. Current measurements support a correct baseline safeguard and an explicit fast deterministic option; they do not establish substantial incremental QUBO-search benefit beyond that stronger control.
