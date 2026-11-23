# Selective-QUBO research review and improvement plan

Research date: **19 November 2026**. Reviewed commit: `fd0068c139b84520888d96c1d768aad01bd79787`.

This is a research and implementation decision document. Production compiler code has not been changed. The accompanying experiment scripts and raw results make the new screening measurements reproducible. Existing comparison CSVs and user-created experiment-history files were preserved.

## 1. Recommended direction

**Fix physical legality before optimizing or claiming better routing results. Then keep selective QUBO, but improve what it optimizes, when it runs, and how its decisions affect the remaining circuit. Increasing tabu effort is not the first priority.**

The independent replay audit found **791 illegal transport-edge swaps in 38 of 48 generated-circuit routing runs**, across both baseline and hybrid. Another 44 occurred in small unitary-check runs. All 24 small unitary comparisons nevertheless passed. Thus, most new screening gains below are measurements of the current abstract software behavior, **not validated physical routing improvements**. The hybrid inherits the issue through its greedy fallback; this does not mean the QUBO decoder accepts those swaps.

The problem is also present in the existing corpus: a separate replay of two existing circuits on both native architectures and both policies found **254 illegal swaps in six of eight runs**. Consequently, the stored benchmark tables also need physical replay and regeneration after repair; this review did not validate all 200 existing circuits physically.

The most useful next steps are:

1. Establish one physical legality contract for greedy routing, QUBO routing, scratch simulation, and execution. Correct the blockage-depth calculation and distinguish physical ion swaps from gate-based SWAPs.
2. Add per-window measurements and reproducible benchmark manifests. Optimize compile latency and physical execution latency as separate quantities.
3. Develop a quality-aware fast mode with model-size budgets, retention of the validated greedy incumbent, and selective extra solver effort. The existing `--fast` mode is a useful starting point, but is not uniformly safe for quality.
4. Improve placement evaluation: use partner **identities** and candidate final positions, then test short deterministic downstream rollouts for difficult decisions. Do not simply increase the existing lookahead weight.
5. Expand evaluation first to brickwork VQE/TFIM-like workloads, ring-QAOA, and QFT. Measure their gains against a stronger heuristic as well as the current one.
6. Investigate path-candidate QUBOs and boundary-preserving local repair after the correctness and measurement foundations exist.

**What is verified:** the existing suite passes; saved operation totals can be reproduced by aggregation; new small circuit instances show provisional gains and failures; independent replay exposes a physical-model violation; runtime profiles identify expensive stages; two proposed micro-optimizations were tested and did not establish substantial speedups.

**What is not verified:** a universal selective-QUBO advantage, gains on large application suites, physical-device makespan or fidelity improvements, or the performance of the architectural changes proposed below. Each proposal has a test and a stopping condition.

## 2. Review scope and current architecture

Reviewed the README, package configuration, all existing Python modules under `src/`, `qccd/`, `frontend/`, `experiments/`, `examples/`, and `tests/`, the QASM corpus format, and stored comparison results. The existing 200 QASM files are synthetic H/CX interaction workloads; they are not 200 independent application algorithms.

| Component | Current behavior | Implication |
|---|---|---|
| `circuit_dag.py` | Per-qudit dependency DAG and front queue | Preserves dependencies, but offers no commutation-aware scheduling |
| `shaw_routing_pass.py` | Chooses mover using hop distance and a small five-operation lookahead bonus; takes the first free target slot | Important routing choices are fixed before the local QUBO |
| `walk_planner.py` | Dry-run greedy movement plus scratch occupancy | Shared costing is useful, but scratch replay lacks physical assertions |
| `congestion_handler.py` | Congestion trigger, lower-bound filter, windows/chunks, local acceptance | Good selective structure; local acceptance cannot guarantee global quality |
| `qubo_formulator.py` | Reachability-pruned occupancy/time variables, crossing auxiliaries, terminal preferences | Already has several sensible reductions; position count alone does not bound BQM size tightly |
| `qubo_solver.py` | Exact enumeration at at most 16 total BQM variables; otherwise two warm-started tabu reads by default | Classical optimization; no quantum advantage has been demonstrated |
| `solution_decoder.py` | Explicit trajectory validation and sequential scheduling | Keep this validation boundary; extend it rather than trusting energy |
| `position_graph.py` | Slot-level native graph | Static native paths are cacheable if topology is immutable |
| `qccd/position_graph.py`, adapter | Richer real-node model expanded into virtual slots | Adapter routing has different connectivity and occupancy-dependent path selection |
| `benchmark_runner.py` | Baseline/hybrid comparisons, worker processes, seeds | Useful foundation; output filenames omit configuration and can overwrite results |
| `optimality_gap.py` | MILP linearization of the existing BQM | Certifies that surrogate model, not general physical routing optimality |
| `learn_trigger.py` | Shallow classifier on solved windows | Labels describe local improvement, not final circuit savings or savings per second |
| Frontend/examples | Visualization and demonstrations of the same components | Some UI controls do not reach full-circuit compilation; duplicate parsers can diverge |

This is a **SHAW-style implementation**, not a demonstrated reproduction of the complete published SHAW/SHAPER evaluation. The current position-graph paper includes broader mapping, scheduling, and architectural semantics. Use precise baseline names in future comparisons. [Position-graph paper, v3](https://arxiv.org/html/2501.12470v3)

## 3. Existing evidence, checked again

Command executed in the project's existing environment:

```powershell
.venv/Scripts/python.exe -m pytest -q
```

Result: **71 passed in 17.96 seconds**. The system `python` points to Python 3.13 without pytest; the project environment is Python 3.10.11. Versions recorded in `experiments/results/research_audit/audit.json`: BQSKit 1.2.1, dimod 0.12.22, dwave-samplers 1.8.0, NetworkX 3.4.2, SciPy 1.15.3.

Re-aggregating the existing CSVs gives:

| Architecture | Baseline operations | Hybrid operations | Reduction | Better / equal / worse circuits | Stored total compile-time ratio |
|---|---:|---:|---:|---|---:|
| Linear | 59,945 | 58,281 | 2.78% | 156 / 22 / 22 | 31.80x |
| Junction grid | 31,222 | 28,886 | 7.48% | 165 / 20 / 15 | 8.41x |
| Adapter grid | 20,569 | 20,563 | 0.029% | 2 / 198 / 0 | 1.09x |
| Adapter dense grid | 13,707 | 13,707 | 0% | 0 / 200 / 0 | 0.99x |

These are **stored-run results**, not a fresh rerun of all 800 architecture/circuit combinations. The grid result rounds to 0.0% in the README but is not exactly zero. All stored rows report success; that flag alone is not an independent physical legality certificate.

The README's existing MILP optimality percentages were not fully rerun here. More importantly, even a proven BQM minimum does not prove minimum executable operations: the BQM counts exchanged ions separately, includes terminal preferences, restricts the mover/window/horizon, and can represent trajectories the sequential scheduler rejects. `optimality_gap.py` compares with one minimum-energy sample and excludes unresolved cases. Rename its claims to **restricted BQM energy optimality**, and separately measure physical-plan optimality with an independent state-space or constrained oracle.

Also correct the README complexity statement: `O(N_ions * |V| * T)` is a polynomial variable-count bound, not exponential scaling. Exhaustive search over those binary variables is exponential; couplings, auxiliaries, and solver conversion memory introduce additional scaling.

## 4. Correctness and measurement issues to address first

### 4.1 Greedy and QUBO physical rules differ — reproduced

`plan_walk` falls back to a swap whenever an occupied next position cannot be vacated. It does not require a `swap`-labeled edge. Native `Placement.swap` does not check adjacency or edge type either.

A two-trap probe with ion 0 at `t0:1`, ion 1 at `seg0`, and path `t0:1 -> seg0 -> t1:0` produces `swap(0,1)` across transport. The QUBO decoder rejects the equivalent exchange.

This also happens in ordinary generated workloads. `research_validation.py` instruments actual native placement mutations and records, without stopping execution, every nonadjacent move or swap on a non-swap edge. For seed 0, 8/12 qubits, six families and both architectures/policies, **38 of 48 runs contain violations**: 625 illegal swaps on linear and 166 on junction grid. Only five baseline/hybrid pairs passed this particular movement check: GHZ at both sizes on both architectures, and 8-qubit QFT on junction grid. This is not a complete hardware-resource validation, and other seeds were not replay-audited.

Separately, all 24 four-qubit unitary checks passed with maximum phase-aligned matrix error `3.33e-16`, despite 44 illegal swaps in those runs. BQSKit's overlap-based distance can suffer near-zero cancellation, so the audit records it but uses direct phase-aligned matrix error for its small-unitary check. Full details are in `validation.json`.

Existing-corpus reproduction (`existing_validation.json`), counting forbidden swaps:

| Circuit / architecture | Baseline | Hybrid |
|---|---:|---:|
| `019_chain_18q.qasm`, linear | 0 | 0 |
| `019_chain_18q.qasm`, junction grid | 4 | 4 |
| `184_random_17q.qasm`, linear | 102 | 100 |
| `184_random_17q.qasm`, junction grid | 23 | 21 |

This directly limits the interpretation of both the historical comparisons and the measured profiles: they describe the current implementation, and the workload/latency distribution can change after the legality fix.

**Plan:** a shared operation validator must require an adjacent empty destination for moves and a permitted exchange edge for swaps. Add gate-location assertions, occupancy-map consistency checks, and completion checks. For transport blockage, find a legal evacuation/reroute or defer; never silently reinterpret it as an in-trap swap. Apply this to both baseline and hybrid before evaluating new dense workloads.

**Verification:** replay traces independently; include occupied segments, fully occupied intermediate traps, capacity-one traps, cycles, and deadlocks. Unitary equality is necessary but insufficient: identity movement markers can preserve the unitary even when the movement is physically illegal.

### 4.2 Blockage depth stops prematurely — reproduced

`_compute_blockage_depth` passes `visited | set(secondary)` into recursion, then immediately skips those secondary nodes as already visited. A chain of five occupied positions returns **1**, although the configured capped depth should reach **3**. Existing tests only demand depth at least one.

**Plan:** define depth precisely and use a bounded BFS/DFS that marks nodes when traversed. Test depths 0, 1, 2, 3+, branching, and cycles. Re-evaluate triggers after fixing it. Defaults already trigger on almost all blocked paths, so do not promise a large default-quality gain from this fix alone.

### 4.3 Cost objective and executed cost disagree

In unit mode, an in-trap exchange costs two ion hops in the BQM but one executable operation. In time mode it costs `2 * 10 = 20` in the BQM while the decoded swap is charged `300`. Actual candidate pricing prevents acceptance of an immediately more expensive plan, but cannot recover a good plan that the solver never returns.

There is also a physical interpretation issue: the cited Moveless paper explicitly distinguishes **GateSWAP**, three CX gates taking 300 microseconds in its model, from **IonSWAP**, a physical rearrangement with a different duration model. The project emits physical-swap identity markers. Mixing those semantics makes the current time weights a hypothetical proxy. [Moveless, shuttling overhead discussion](https://arxiv.org/html/2508.03914v1)

**Plan:** specify supported native operations first. Make all paths through cost estimation use that same model. A transition/action formulation can count a physical exchange once; alternatively use validated complete-plan costs in a candidate-selection model. Re-derive objective bounds and auxiliary penalties after changes. The README reports that a previous swap-pricing attempt worsened results: recover its configuration if available, and test the revised model as a new ablation rather than assuming an objective correction automatically improves tabu search.

Also make threshold units explicit: the learned slack thresholds 2/4 were motivated by operation counts, but the same comparisons run against weighted costs under `--objective time`. Retrain or normalize them for the time objective; `min_saving` likewise needs an objective-specific unit.

### 4.4 “Never worse layout” is an aggregate, stale-position proxy

`_next_partner_positions` stores positions before optimization. `layout_cost` sums distance to those fixed positions; it does not look up the partner's position after the candidate moves both ions. It also compares an aggregate, not a per-ion guarantee. `LOOKAHEAD_SIZE=5` counts single-qubit operations too, and scanning only after the current operation can omit other relevant ready operations.

**Plan:** store partner IDs and gate IDs, score candidate positions on the candidate board, deduplicate gate pairs, and collect the next relevant two-qubit dependencies. Measure whether this reduces final regressions. Keep current behavior as an ablation. The correction is conceptually justified; its effect on final routing must be measured.

Use a gate-rendezvous estimate that is zero when both ions already share an executable trap; raw slot-to-slot distance can remain positive even when no routing is needed.

### 4.5 Architecture, imports, and frontend restrictions

- Adapter `shortest_path` consults its attached live placement. During chunk planning, that can differ from the scratch `BoardView`. Pass the planning board explicitly; do not cache complete adapter paths as topology-only data.
- The adapter makes cross-node connections between every virtual slot and discards directed-edge distinctions when building an undirected graph. This is an optimistic architecture abstraction, not equivalent to ordered boundary-access traps. It should remain an explicit control configuration.
- `ShawRoutingPass` treats the first two operands of any gate with two or more qudits as the routing pair. Require decomposition to one-/two-qubit gates, or explicitly reject unsupported higher-arity gates before routing.
- Backend QASM loading intentionally accepts only H/CX. The frontend parser silently ignores unmatched statements; reuse the strict backend import path and reject unsupported input visibly.
- In full-circuit frontend mode, solver choice/effort and depth threshold are not forwarded to `BenchmarkRunner`; `--fast` is not exposed there. Wire controls through `handler_options` and display the effective configuration.
- Research dependencies such as scikit-learn are not declared as experiment extras. Pin a research environment, and record optional dependency versions.

## 5. New circuit-family screening

Ran **24 distinct generated circuits**: six families, 8 and 12 qubits, two architectures. Each has three solver seeds for normal mode and three for `--fast`: **144 hybrid runs**, plus 24 baseline runs. Circuits were built directly as BQSKit circuits, allowing RX/RY/RZ without modifying the H/CX parser.

The following are reductions in **aggregate reported operation counts** across the two sizes and three seeds. **Most families fail the independent movement-legality audit described above; rerun this table after repair before interpreting it as physical savings.** Each baseline is repeated in the denominator to match the seeded hybrid runs. These are exploratory results for these specific circuits and placements, not confidence intervals or population estimates.

| Family | Linear normal | Linear fast | Junction normal | Junction fast | Normal-mode better/equal/worse, linear; junction |
|---|---:|---:|---:|---:|---|
| GHZ, chain preparation | 34.09% | 34.09% | 1.92% | 1.92% | 6/0/0; 3/3/0 |
| QFT, decomposed controlled phases | 10.57% | 10.45% | 2.89% | 1.24% | 6/0/0; 6/0/0 |
| Ring-QAOA, 3 layers | 8.44% | 8.44% | 8.54% | 7.45% | 6/0/0; 6/0/0 |
| Cubic-graph QAOA, 3 layers | 4.61% | 4.36% | 5.06% | 4.27% | 3/0/3; 6/0/0 |
| Brickwork RY/CX VQE ansatz, 6 layers | 12.53% | 12.53% | 13.20% | 13.20% | 6/0/0; 6/0/0 |
| Repeated bipartite parity-check interaction proxy | 0.51% | -1.34% | 8.92% | 8.75% | 4/0/2; 3/0/3 |

Definitions and limitations:

- GHZ adds H followed by a CX chain. It is a useful application sanity check, but closely resembles an existing synthetic family.
- QFT uses a standard controlled-phase decomposition into CX/RZ, omits the final output reversal, and is equivalent up to global phase to that convention. Native controlled-phase/ZZ compilation can change its routing pressure; test both representations.
- QAOA uses a fixed ring or one random 3-regular graph per size, graph seed 17, fixed nonzero angles, and three cost/mixer layers. It does not establish typical QAOA performance across graph distributions.
- The VQE circuit is an ansatz workload, not a chemistry or energy-convergence experiment.
- The parity-check proxy has no measurement, reset, feedforward, noise model, or code-distance guarantee. It is **not** a functioning QEC benchmark. Its mixed results specifically argue against promising immediate QEC gains.
- Circuit generation, graph construction, imports, and UI time are outside `compile_time_s`; this metric times routing. Timing observations use this machine and are exploratory.

**Priority expansion after legality repair:** VQE brickwork and TFIM/TFXY Trotter layers; ring, grid, cubic, and dense QAOA; full/approximate QFT and QPE kernels; then Cuccaro and Draper adders. These are promising candidates, not established physical winners. Test random label permutations and good initial mappings, since natural chain ordering can dominate outcomes. MQT Bench provides relevant application families, including QFT, QAOA, VQE ansätze and arithmetic circuits. Pin its version and export a consistent gate basis. [MQT Bench algorithm catalog](https://mqt.readthedocs.io/projects/bench/en/stable/benchmark_selection.html)

**Longer-term QEC expansion:** real repetition/Steane/surface-code extraction, then small HGP/BB instances. Add measurement/reset and role-aware placement first. Compare against ancilla-only routing and structured schedules, not only a generic greedy baseline. Moveless and Cyclone motivate exploiting repeated structure and hardware resources; their published improvements are not evidence that this selective QUBO will match them. [Moveless](https://arxiv.org/abs/2508.03914), [Cyclone](https://arxiv.org/abs/2511.15910)

**Negative controls:** already-local repeated two-qubit gates, single-qubit-only circuits, permissive adapter grids, and circuits with near-optimal initial placement. A useful trigger should spend almost no optimization time there.

## 6. Where compile latency actually goes

Profiled `184_random_17q.qasm` with default hybrid routing on both native architectures. `cProfile` adds overhead, so use these measurements for attribution, not a speedup claim. Stage wrappers time the whole public solve call, including candidate postprocessing.

| Measurement | Linear | Junction grid |
|---|---:|---:|
| Profiled routing time | 6.913 s | 1.829 s |
| QUBO builds/solves | 54 | 32 |
| Formulation time | 0.815 s, 11.8% | 0.176 s, 9.6% |
| Solve time | 5.623 s, 81.3% | 1.258 s, 68.8% |
| Decode time | 0.016 s, 0.2% | 0.008 s, 0.4% |
| Total BQM variables: min / median / max | 23 / 802 / 2,279 | 33 / 207.5 / 2,279 |

The remainder includes routing decisions, rerouting, scratch boards, and circuit emission. Junction-grid alternative-path enumeration consumes about 0.136 s here. Within the solve stage, dense conversion alone takes about 0.556 s on linear and 0.081 s on junction grid; these are nested timings, not additional costs.

The installed TabuSampler converts the BQM to a **dense square array**. At 2,279 variables a single float64 matrix is about 40 MiB, before additional conversion arrays. Sparse BQM storage does not imply sparse solver memory. Track total variables, auxiliaries, couplings, conversion time, and peak memory.

The README's statement that “the solver is not the bottleneck” should distinguish **solution quality** from **runtime**. A near-optimal local solver can still dominate compile latency. The README's median-size claim also needs a workload and a distinction between primary and auxiliary variables.

### 6.1 Measured shortcuts that did not justify priority

**Native shortest-path memoization:** temporarily cached exactly the paths returned by the original native method. Three alternating-order repetitions per configuration produced unchanged operation counts. Median routing times:

| Architecture/circuit | Original | Cached |
|---|---:|---:|
| Linear chain-18 | 0.0905 s | 0.0898 s |
| Linear random-17 | 6.2282 s | 5.9443 s |
| Junction chain-18 | 0.3069 s | 0.3093 s |
| Junction random-17 | 1.2608 s | 1.2674 s |

The roughly 4.6% improvement on one case is worth revisiting, but most differences are small or negative. This does not support a major caching speedup claim. Future distance/lower-bound caches are a different experiment; include preprocessing cost and topology/cost-model invalidation.

**Singleton fixing:** captured 32 real junction-grid BQMs; fixed the unique variable of each singleton `(ion,time)` group to one and restored it before decoding. This removed 121 of 10,491 aggregate variables, only 1.15%. Ten random assignments per window verified reduced/full energy equality after restoration. Total sampled solve time was 1.278 s before versus 1.345 s after. Among 27 windows with valid plans in both versions, 26 had equal operation counts and one became worse; one additional window gained validity, while four remained invalid in both versions.

This is a useful negative result: preserving the restricted optimization problem does not preserve a finite-budget tabu trajectory. Do not ship singleton fixing as a proven speed improvement. It may become useful alongside deeper propagation, but needs a separate budget/quality study.

### 6.2 Higher-value latency changes

| Change | Implementation decision | Evidence / acceptance criterion |
|---|---|---|
| Better solve admission | Predict expected accepted saving and solve time using slack, ions, horizon, projected variables/couplings, occupancy and family-independent structural features | Existing fast mode often helps; train grouped by circuit family/graph seed, then require held-out quality/latency targets |
| Cheap model-size guard | Estimate reachable-state and crossing-product counts before constructing the BQM; bound total variables and estimated dense memory | Profiles show that 16 positions permit 2,279 variables; compare budgets such as 128/256/512/1024, not just node caps |
| Adaptive solver effort | Retain a validated incumbent; use a small first budget, expand only on high-value or uncertain windows | Uniformly smaller budgets already reportedly failed; test a staged policy against equal wall-time baselines |
| Reuse validated plans | Cache repeated window solutions by canonical topology, full relevant occupancy, goal/partner roles, costs and boundary conditions | Best suited to repeated layers/QEC kernels; always revalidate and reprice on the current board; report cold and warm cache results |
| Avoid unnecessary work | Reuse a single dry-run greedy plan for its cost, warm start and replay; avoid repeated `BoardView` reconstruction; cache static distance tables | Code-level duplication is visible; measure only after the main solve costs are controlled |
| Faster clear rerouting | Search the occupancy-filtered graph for a clear path, with a detour cost cap | Can replace repeated k-simple-path enumeration, but changes behavior because the current search stops after eight candidates; compare quality on both baseline and hybrid |
| UI responsiveness | Cache immutable parse/topology results and retain completed results in session state; use an explicit run action for the single-window solve | Frontend currently executes the single-window solve on reruns; do not share mutable placements across sessions |

Do not treat `--jobs` as a single-circuit latency fix: it improves suite throughput and may increase contention. Avoid cloud Leap as a default latency strategy for tiny local windows; its queue/network/runtime costs require a separate use case.

Tabu's `coefficient_z_first` bounds variable updates with exceptions for greedy descent; the 2,000 ms timeout is per read. The documentation warns that finite timeouts can affect reproducibility. Log timeout/budget settings and test multiple seeds rather than declaring iteration budgets fully deterministic under all loads. [D-Wave TabuSampler documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.TabuSampler.sample.html)

## 7. Technical changes for better routing quality

### A. Stronger, fair heuristic before escalation

Enumerate both mover directions and all free destination slots, price legal greedy plans under the chosen objective, and use future-position evaluation only as a controlled tie-break or explicit weight. Preserve a no-QUBO version of every such improvement.

Why: current mover selection estimates hop length, while acceptance compares actual blocker moves/swaps. A cheap better choice can eliminate a bad window without solving anything. A QUBO gain against the old selector alone can overstate its incremental value.

Test baseline, stronger heuristic, current hybrid, and stronger heuristic plus QUBO on identical initial placements. A selector with better immediate cost is not guaranteed to improve the full circuit. This is a screening hypothesis, not a theorem.

**Probe performed:** `research_policy_probe.py` temporarily changes only mover/slot selection, scoring the current greedy plan cost minus the existing lookahead bonus. It applies the selector equally to baseline and hybrid, for four families at 8/12 qubits and seed 0. Examples, aggregated over the two sizes:

| Case | Original baseline / hybrid | Changed selector baseline / hybrid |
|---|---:|---:|
| Linear QFT | 861 / 770 | 815 / 748 |
| Linear ring-QAOA | 549 / 504 | 613 / 552 |
| Junction QFT | 566 / 551 | 550 / 566 |
| Junction parity proxy | 583 / 535 | 629 / 519 |

This rules out adopting this selector wholesale: a cheaper immediate decision can worsen final outcomes, and QUBO can lose against the changed baseline. These runs still use the current uncorrected greedy movement semantics and were not independently replay-certified; treat them as diagnostics only. Revisit after legality repair, together with downstream scoring.

### B. Candidate-board lookahead and bounded rollout

First correct partner identity and final-board scoring. Then, only for ambiguous candidate layouts, simulate the next `K` ready two-qubit gates using the cheap legal router from both the greedy and QUBO boards. Compare immediate cost plus rollout cost. Start with `K = 2, 4, 8`, candidate count 2–4, and strict per-circuit time budgets.

Clone **both placement and dependency/frontier state**. Rollout must not mutate the live adapter or DAG. Use deterministic tie-breaking and retain the greedy candidate. Prefer Pareto-distinct terminal layouts among valid samples, rather than returning only one minimum-energy layout.

This differs from the previously unsuccessful “more lookahead”: it prices downstream executable plans on actual candidate states rather than increasing stale distance penalties. It can still fail beyond its horizon. Evaluate global regressions explicitly and keep it only if added savings justify its compile cost.

### C. Preserve boundary state for a real global non-regression guarantee

An offline local replacement can provide a stronger guarantee than today's acceptance rule: replace a segment of a valid baseline trace only if it executes the same logical gates, preserves their legal dependencies, and rejoins the **same complete placement and remaining computation state** at the segment boundary with lower additive cost.

Under those conditions, the unchanged suffix remains executable and its additive cost is unchanged. This gives a genuine whole-trace operation-count guarantee. If optimizing makespan, the boundary must also include resource availability/timing state.

Tradeoff: fixed boundary layouts restrict improvements. Start with small trace segments spanning several gates, and compare their benefit per optimization second with compiling both complete policies. This is a principled route to a cheaper safe portfolio, not an assurance that sufficient improvements exist.

### D. Select windows by conflicts and model budget

Current longest-fitting path chunks optimize a single route independently and may settle blockers poorly for the next chunk. Construct a small conflict neighborhood from the mover, blocking ions, alternate parking slots, and immediate upcoming interactions. Expand only if the current neighborhood cannot produce a useful valid candidate.

Consider overlap between chunks and retain a few boundary placements when slack is high. Keep outside ions fixed and enforce boundary occupancy explicitly. Large-neighborhood repair is motivated by MAPF research, but QCCD has gate rendezvous, allowed in-trap exchanges and hardware resources that standard MAPF does not model. [MAPF large-neighborhood benchmarking](https://arxiv.org/abs/2407.09451)

### E. Path-candidate QUBO as the main formulation experiment

Instead of one variable per ion/position/time, generate a small set of **complete, timed candidate trajectories** for each active ion, including a legal baseline combination, then select one per ion:

```text
min sum(c[i,p] * y[i,p])
  + A * sum_i (sum_p y[i,p] - 1)^2
  + B * sum_conflicting_pairs y[i,p] * y[j,q]
```

With `m` ions and `K` candidates each, the primary variable count is `m*K`; interactions may still be quadratic in that count. Conflicts can encode occupancy, forbidden transport exchanges, and timed resource overlap. Candidate generation and waiting-time choices are part of the measured runtime, not free preprocessing.

Crucial qualifications:

- Independently chosen individual paths may omit cooperative blocker movements. Include paths derived from complete legal plans and expand the pool when needed.
- Pairwise collision checks do not automatically guarantee the current sequential scheduler can execute a longer rotation cycle. Use the full validator or a stronger scheduling formulation.
- In-trap swaps require coordinated actions and a consistent joint cost. Do not simply sum two independent hop costs again.
- Restricted-pool optimality is not global optimality. Compare the candidate pool with a tiny exact action-based oracle to quantify missed plans.
- Use exact enumeration/ILP as a same-pool classical comparator. The QUBO encoding itself is not the source of quantum advantage.

Path-based conflict-graph QUBOs and decomposition have direct precedent in hybrid MAPF work. The proposal here is an adaptation to the project's physical semantics, not a new established performance result. [Hybrid Quantum-Classical Multi-Agent Pathfinding](https://arxiv.org/html/2501.14568v2)

### F. Initial placement, gate ordering and role-aware policies

Add an interaction-weighted initial placement and compare it equally for heuristic and hybrid. Permit prioritization among truly ready disjoint gates. Commutation-aware block reordering requires a separate semantics proof; never reorder CX gates merely because their interaction graph looks convenient.

For QAOA, preserve layer semantics and exploit recurring interaction graphs. For QEC, retain ancilla/data roles and measurement boundaries, then compare ancilla-only movement and ancilla reuse. The likely gain may come from a better structured heuristic, with QUBO reserved for residual conflicts.

Recent parallelism-aware QCCD routing research supports evaluating movement and scheduling jointly across QFT, QAOA and arithmetic workloads, but its reported gains use a different compiler and model. [Adaptive Parallelism-Aware Qubit Routing, 2026](https://arxiv.org/html/2603.19969v1)

## 8. Physical execution latency needs its own scheduler

Keep three separate metrics:

1. **Compile latency:** CPU wall time, including separately reported import/setup/model construction and routing.
2. **Movement work:** sum of primitive movement durations under a stated cost model. This is what current `movement_time_us` approximates.
3. **Execution makespan:** end time of a valid hardware schedule with gate, movement, measurement/reset and resource durations.

BQSKit circuit depth with identity markers only reflects qudit dependencies; it cannot represent all trap, junction, laser or channel contention. A lower depth or lower duration sum does not by itself prove a faster execution schedule.

Add a movement/event trace with ion IDs, source/destination, primitive type, duration and required resources. Schedule operations with per-ion, trap, transport-edge and junction exclusivity; then validate dependencies and resource overlap. Begin with a documented simplified resource model and compare against a fully serialized schedule. Optimize critical-path decisions only after measuring this scheduler.

Use the cited split/merge and transport durations as a reproducible **scenario**, not a universal hardware specification. Sweep physical-swap, in-trap repositioning, split/merge and junction durations; include cooling/measurement only when modeled explicitly. Cyclone makes clear that architecture and parallel scheduling can materially affect QCCD execution behavior. [Cyclone operation model](https://arxiv.org/html/2511.15910v1)

The latest hardware/compiler survey also emphasizes platform-specific operation and architectural abstractions. Keep native ordered traps and optimistic adapter graphs separate in all tables. [From QCCD Hardware to Quantum Compilers, October 2026](https://arxiv.org/abs/2610.36965)

## 9. Experimental protocol and implementation gates

### Phase 0 — trustworthy baseline

Files: `walk_planner.py`, both placement implementations, `congestion_handler.py`, `shaw_routing_pass.py`, frontend parser, benchmark runner.

Deliver: shared replay validator; depth fix; unsupported-input rejection; consistent operation semantics; configuration manifests and unique result directories. Log failures separately rather than letting zero failure counts enter quality plots.

Pass gate: existing tests plus targeted adversarial legality checks, all current circuits parsed, sampled end-to-end physical replay, and unitary checks on small one-/two-qubit-gate circuits. Never compare an invalid baseline trace against a valid hybrid as though both were usable outputs.

### Phase 1 — latency instrumentation and quality-aware admission

Log each trigger/skip/build/solve/decode/accept stage; primary and auxiliary sizes; rejected sample reasons; validated warm-start feasibility; local saving; whether the final combined chunk plan was accepted; and final circuit delta.

The current learning labels count windows that locally improved even if the full chunked plan is rejected later. Correct this before training. Separate training/validation/test by graph instance, circuit template and placement seed, not just alternating filenames.

Provisional engineering target: **at least 2x lower total hybrid routing time while retaining at least 95% of aggregate operation savings on held-out cases with positive reference savings**. For nonpositive reference savings, assess absolute operation totals and regression rates separately. This is a target, not a forecast. Reject policies that hide losses behind a better average.

### Phase 2 — quality improvements

Order: stronger legal heuristic; partner-ID scoring; bounded rollout; initial placement; conflict-based windows. Perform one change at a time and combinations only after individual attribution.

Pass gate: paired full-circuit improvement with a family-grouped bootstrap interval above zero, no legality failures, and a reported latency/quality Pareto curve. Set a hard regression-rate target before tuning; use the boundary-preserving repair or full portfolio when an actual non-regression guarantee is required.

### Phase 3 — formulation and physical scheduling research

Compare current occupancy QUBO, consistent action/transition model, path-candidate QUBO, and direct constrained local search/CP-SAT/MILP. Use the same physical validator and same decision scope.

For tiny windows, use exhaustive search over **legal physical actions**, retaining exact endpoint conditions, to certify minimum operation cost independently of BQM energy. Report time-limited oracle cases and invalid optimum samples explicitly. On larger windows, report incumbent cost and valid lower bound without calling a timeout optimal.

### Benchmark matrix

| Dimension | Initial confirmation study | Scale-up after passing |
|---|---|---|
| Qubits | 8, 12, 16, 20 | 32, 64, 128 under resource budgets |
| Circuit families | Existing corpus; QFT; ring/cubic QAOA; VQE/TFIM; arithmetic | QPE, chemistry-derived circuits, real QEC extraction |
| Circuit instances | Multiple graph/depth/template seeds | Held-out family and topology transfer |
| Hardware | Native linear and junction grid; adapter negative controls | Ordered traps with capacity 3/4/8, bottleneck junctions, ring |
| Placement | Home, shuffled, interaction-aware | Several occupancy levels and role-aware placement |
| Solver | At least 5 paired seeds | More seeds where variance affects conclusions |
| Timing | Warm-up plus 5 alternating-order repetitions, single worker | Separate parallel throughput/memory study |
| Budgets | Fixed total compile budget and model memory limits | Quality/latency frontier, not one best setting |

Report weighted operation saving, mean/median per-instance saving, wins/ties/losses, worst regression, success rate, physical makespan when available, and median/p95 compile latency. Keep timeout/invalid cases in denominators. The current runner's +/-5% verdict bin is different from strict better/equal/worse and should be labeled clearly.

Mandatory baselines: current heuristic; stronger heuristic; both with the same placement improvement; default QUBO; fast QUBO; equal-budget constrained local solver; and best-of-both including the total compilation cost of both runs. Do not compare a tuned hybrid to an untuned baseline with different input decompositions or machine capacities.

## 10. Reproduction and research artifacts

```powershell
# Existing tests
.venv/Scripts/python.exe -m pytest -q

# Stored-result audit, six circuit families, normal/fast screen, runtime profiles
.venv/Scripts/python.exe experiments/research_audit.py

# Native path cache and singleton-fixing experiments
.venv/Scripts/python.exe experiments/research_ablations.py

# Independent movement checks and small-circuit unitary checks
.venv/Scripts/python.exe experiments/research_validation.py
.venv/Scripts/python.exe experiments/research_validation.py --existing

# Stronger cheap move-selector experiment, applied equally to both methods
.venv/Scripts/python.exe experiments/research_policy_probe.py
```

Outputs are under [experiments/results/research_audit](experiments/results/research_audit/). Scripts intentionally use separate files and restore temporary monkeypatches; rerunning replaces only their own research output files. Run timing experiments sequentially to avoid cross-experiment CPU contention.

Artifact inventory:

- `audit.json`: environment, historical CSV aggregation and minimal bug probes.
- `circuit_screen.csv`: all 144 hybrid screening rows with corresponding baseline metrics.
- `profiles.json`, `profile_linear.txt`, `profile_junction_grid.txt`: per-window model sizes and runtime attribution.
- `ablations.json`: path-cache repetitions and full/reduced BQM comparisons.
- `validation.json`: 48 generated-circuit routing checks, 24 small-unitary checks, and individual forbidden operations.
- `existing_validation.json`: eight existing-corpus routing checks and individual forbidden operations.
- `policy_probe.csv`: 32 configurations comparing the old and changed move selector, each with baseline and hybrid.

All four new experiment scripts passed Python syntax compilation. No production source file, existing test, or existing result CSV was modified.

The screen demonstrates where further evaluation is justified. It does not establish that every proposed implementation will work. The strongest immediate decision is to fix legality and objective semantics, preserve the successful selective structure, and spend optimization effort only where measured benefit warrants it.
