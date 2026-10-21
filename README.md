# QubRoute: Selective-QUBO Optimization for Trapped-Ion Qubit Shuttling

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: BQSKit](https://img.shields.io/badge/Compiler-BQSKit-purple.svg)](https://bqskit.lbl.gov/)
[![Optimization: dimod](https://img.shields.io/badge/QUBO-dimod%20%2F%20dwave--samplers-green.svg)](https://github.com/dwavesystems/dimod)
[![UI: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Tests: 71 Passed](https://img.shields.io/badge/Tests-71%20Passed-brightgreen.svg)](tests/)

> **Minimal Working Prototype & Compiler Pipeline**: A hybrid quantum compilation framework combining fast heuristic routing (SHAW) with localized **Quadratic Unconstrained Binary Optimization (QUBO)** to eliminate shuttling bottlenecks in Trapped-Ion Quantum Charge-Coupled Device (QCCD) architectures.

---

## 📌 Project Overview

### The Trapped-Ion Shuttling Bottleneck
In trapped-ion quantum computers, physical ions are trapped in micro-fabricated electrode arrays. Multi-qubit entangling gates require physical ions to be **shuttled** into shared interaction zones across the QCCD architecture:
- **Ion Heating & Dephasing**: Every shuttle hop causes kinetic heating and motional dephasing, degrading gate fidelity.
- **Congestion & Deadlocks**: Narrow transfer segments and multi-ion trap capacity constraints cause severe bottlenecks under high multi-qubit gate contention. Purely greedy heuristics (e.g., A* or basic SHAW) struggle with cascading blockages, leading to high routing regret or swap deadlocks.
- **Global QUBO Scaling Barrier**: Formulating the entire shuttling routing problem across an entire chip into a single global QUBO matrix scales exponentially ($O(N_{\text{ions}} \cdot |V| \cdot T)$), making classical simulated annealing or QPU execution impractically slow for realistic quantum circuits.

### The QubRoute Solution: Selective QUBO Optimization
**QubRoute** introduces a selective, hybrid paradigm:
1. **Continuous Heuristic Routing**: Circuits are routed with a SHAW-style greedy walk along the architecture's position graph.
2. **Real-Time Contention Monitoring**: When a path is blocked, the router computes three congestion metrics: Local Contention ($\kappa$), Routing Regret ($\rho$), and Blockage Depth ($d$).
3. **Selective QUBO Escalation**: If the trigger fires and the greedy walk is *not provably optimal* (its cost exceeds the lower bound), the system builds a QUBO that plans the moves of **every ion in the window $W$** and solves it with tabu search, warm-started from the greedy walk. Long paths are planned window by window.
4. **Explicit Validation**: Every candidate sample is decoded and checked against the physical rules (one-hot, start positions, movement legality, capacity, no exchanges over transport, goal reached, schedulable) instead of trusting its energy.
5. **Never Worse Locally**: A plan is executed only if it needs fewer physical operations than the greedy walk on the same board and leaves ions no further from their next gate partners. Otherwise the greedy walk runs.
6. **Best of Both (optional)**: Compiling a circuit both ways and keeping the cheaper result is never worse than the baseline.

The prototype includes both a **complete compilation backend** and an **interactive Streamlit web dashboard** for circuit execution and mathematical visualization.

---

## 💻 Source Code Structure

```text
Selective-QUBO-Optimization-for-Trapped-Ion-Qubit-Shuttling/
├── src/                               # Core compilation and optimization engine
│   ├── position_graph.py              # Position graph, placement, linear & junction-grid machines
│   ├── circuit_dag.py                 # Circuit DAG dependency extraction
│   ├── walk_planner.py                # Greedy walk planner, cost models, scratch board
│   ├── congestion_handler.py          # Severity metrics (κ, ρ, d), window extraction, accept rule
│   ├── qubo_formulator.py             # Time-expanded QUBO with reachability pruning
│   ├── qubo_solver.py                 # Exact / tabu (default) / annealing / optional D-Wave Leap
│   ├── solution_decoder.py            # Per-sample validation and move-plan scheduling
│   ├── shaw_routing_pass.py           # BQSKit routing pass (greedy walks + QUBO plans)
│   ├── qccd_adapter.py                # Adapter for the qccd/ architecture models
│   └── benchmark_runner.py            # Baseline vs selective QUBO: architectures, seeds, parallel
├── qccd/                              # Richer QCCD architectures (linear, grid, junction)
├── frontend/                          # QubRoute interactive web dashboard
│   ├── app.py                         # Streamlit application (Topology & Live Routing tabs)
│   └── README.md                      # Frontend-specific documentation and usage instructions
├── .streamlit/                        # Streamlit configuration
├── tests/                             # 71 unit & integration tests + 200 OpenQASM benchmark circuits
├── experiments/
│   ├── optimality_gap.py              # Tabu vs exact MILP optimum on real windows
│   ├── learn_trigger.py               # Learns which windows are worth solving
│   ├── results/                       # comparison*.csv per architecture
│   └── plots/                         # Per-family operations, per-circuit change, compile time
├── examples/                          # Standalone demonstration scripts
│   ├── demo_selective_qubo.py         # One congestion event through the whole pipeline
│   └── demo_position_graph.py         # QCCD architecture topology demo
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation Instructions

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10.11)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/Chaitanya-G41/Selective-QUBO-Optimization-for-Trapped-Ion-Qubit-Shuttling.git
cd Selective-QUBO-Optimization-for-Trapped-Ion-Qubit-Shuttling
```

### 2. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation with the Test Suite
Run the automated test suite:
```bash
python -m pytest -v
```
All 71 unit and integration tests should pass.

---

## 🖥️ Running the Prototype

### Option A: Interactive Web Application (QubRoute GUI)
Launch the interactive Streamlit dashboard:

```bash
streamlit run frontend/app.py
```
*(Or on Windows: `.venv\Scripts\python.exe -m streamlit run frontend/app.py`)*

The dashboard will open at **`http://localhost:8501`** and features:
- **Tab 1: Hardware Topology**: Live visual map of the QCCD architecture, trap slots, transfer segments, and ion placement tables.
- **Tab 2: Live Routing & Benchmarks**:
  - **QASM Benchmark Execution**: Select a benchmark circuit, inspect its OpenQASM 3.0 source, and compile it with both the baseline and selective QUBO, side by side.
  - **Single Window Deep Dive**: Step through one congestion event: metrics, window, QUBO size, solver, rejected candidates, and the greedy walk vs the QUBO plan.

### Option B: Terminal Prototype Demonstration
Run the standalone end-to-end compiler demonstration script:
```bash
python examples/demo_selective_qubo.py
```

### Option C: Automated Benchmark Suite
Compare the baseline with selective QUBO on all 200 circuits:
```bash
python src/benchmark_runner.py --arch junction_grid --jobs 8
```
- `--arch linear | junction_grid | grid | grid_dense`: machine (default `linear`)
- `--jobs N`: parallel workers
- `--seeds N`: repeat with N solver seeds and report mean ± stdev
- `--fast`: learned trigger, several times faster with a small quality cost
- `--objective time`: the QUBO minimises shuttling time instead of operation count

Results go to `experiments/results/comparison[_arch].csv` and `experiments/plots/`.

---

## ⚙️ Implementation Details

### 1. Architectural Abstraction: The Position Graph ($G_p$)
The QCCD hardware is abstracted as an undirected graph $G_p = (V, E)$ (`src/position_graph.py`):
- **Trap Zones**: Contain discrete slot nodes $\{v_1, \dots, v_k\}$ connected as a fully connected clique (representing within-trap ion rotation/swapping).
- **Transport Segments**: Transition nodes connecting adjacent traps via merge and split operations.
- **Capacity Function $\text{cap}(v)$**: Trap slots hold at most 1 ion; transport channels strictly forbid multi-ion occupancy.
- **Ion Placement ($\phi$)**: Tracks the instantaneous mapping of qudit $q_i \mapsto v \in V$.

Machines used in the benchmarks (one home ion per trap, capacity 2):
- **linear**: traps in a chain, joined by one segment between neighbours.
- **junction_grid**: X-junctions on a square grid with a linear trap on every edge; a trap's two end slots connect to its two junctions, so crossing a trap means passing its ion chain.
- **grid** / **grid_dense**: the `qccd/` grid via an adapter, where any slot of a trap reaches any neighbour (no in-trap ordering).

### 2. Congestion Severity Metrics & Trigger Condition
When a mover's shortest path $P$ (with $C_{LB} = |P| - 1$ hops) contains occupied intermediate positions $B(P)$, the router computes (`src/congestion_handler.py`):
1. **Local Contention ($\kappa$)**: fraction of occupied positions on $P$.
2. **Routing Regret ($\rho$)**: $\rho = 2|B(P)| / C_{LB}$ (each blocker ~ one hop aside and one back).
3. **Blockage Depth ($d$)**: levels of occupied neighbours behind each blocker.

**Trigger**: $(\kappa > \kappa_{th} \land \rho > \rho_{th}) \lor d > d_{th}$, followed by a cheap filter: skip if the greedy walk already meets the lower bound. Defaults are $\kappa_{th} = \rho_{th} = 0$, $d_{th} = 2$: on the benchmark suite, stricter thresholds only gave up savings while saving little time.

**Learned trigger** (`--fast`, `experiments/learn_trigger.py`): a depth-2 decision tree trained on logged solves gives a readable rule: solve only if the greedy walk's slack over the lower bound is at least 2 operations (windows with ≤3 ions) or 4 (larger windows). It skips about half of all solves.

### 3. Local Window ($W$)
$W$ contains the path, every slot of each trap the path touches, the goal trap's slots, and the neighbours of blocked positions. All ions in $W$ are decision variables. If $W$ would exceed 16 positions, the path is cut at trap slots into the longest chunks that fit, and the chunks are planned one after another on a scratch copy of the board (greedy for any chunk the QUBO cannot improve). The combined plan must still beat the greedy walk as a whole.

### 4. QUBO Formulation
Binary variables $x_{i,v,t}$ mean ion $i$ is at position $v$ at time $t \in [0, T]$ (`src/qubo_formulator.py`). Only **reachable** states get a variable: $\text{dist}(\text{start}_i, v) \le t$ and $v$ can still reach the ion's final set within $T - t$ steps. This pruning makes the start positions and goals hard constraints and keeps models small (median ~50 variables). Final sets: the mover must end in the goal trap, ions that start in the goal trap stay there, and every other ion must end on a trap slot.

$$H = H_{\text{cost}} + H_{\text{lookahead}} + H_{\text{settle}} + P\,(H_{\text{one-hot}} + H_{\text{capacity}} + H_{\text{movement}} + H_{\text{crossing}})$$

| Term | Expression | Purpose |
|---|---|---|
| $H_{\text{cost}}$ | $\sum c(u,v)\, x_{i,u,t}\,x_{i,v,t+1}$ over adjacent $u \ne v$ | price per hop of any ion: 1, or its duration under the time model |
| $H_{\text{lookahead}}$ | $w \cdot \text{dist}(v, \text{next partner of } i)\; x_{i,v,T}$ | leave ions well placed for upcoming gates ($w = 0.5$) |
| $H_{\text{settle}}$ | $\varepsilon\, x_{i,v,T}$ for $v \ne \text{start}_i$ (ions without an upcoming gate) | tie-break: don't disturb bystanders ($\varepsilon\,|A| < 1$) |
| $H_{\text{one-hot}}$ | $\sum_{i,t} (\sum_v x_{i,v,t} - 1)^2$ (with its constant) | exactly one position per ion and step |
| $H_{\text{capacity}}$ | $\sum x_{i,v,t}\,x_{j,v,t}$, $i \ne j$ | one ion per position |
| $H_{\text{movement}}$ | $\sum x_{i,u,t}\,x_{i,w,t+1}$, $(u,w) \notin E$ | no teleporting |
| $H_{\text{crossing}}$ | $x_{a,u,t}x_{a,v,t+1}x_{b,v,t}x_{b,u,t+1}$ on transport edges | no exchanges over segments (in-trap exchanges are physical swaps and allowed) |

$P$ exceeds the largest possible objective, so every violation costs more than any valid plan. The quartic crossing terms are reduced with Rosenberg products whose penalty is $2P$, so an auxiliary variable can never cheat.

### 5. Solving, Validation and Acceptance
- **Solver** (`src/qubo_solver.py`): `ExactSolver` for ≤16 variables, otherwise **tabu search** (dwave-samplers) with a fixed iteration budget, so runs are reproducible. It is warm-started from the greedy walk encoded as a trajectory. Simulated annealing is available (`method="sa"`) but does poorly here: moving one ion means leaving a one-hot state, which costs a full penalty.
- **Validation** (`src/solution_decoder.py`): each distinct low-energy sample is decoded and checked for one-hot, start positions, movement legality, capacity, collision avoidance, goal, and schedulability. Valid samples become move plans (`move` = 1 op, in-trap `swap` = 1 op), and the cheapest wins.
- **Acceptance** (`src/congestion_handler.py`): plan cost < greedy walk cost on the same board, and the plan leaves ions no further from their next partners.
- **Cost models** (`src/walk_planner.py`): by default every operation costs 1. `TIME_COSTS_US` prices operations by approximate hardware duration: split/merge 80 µs and transport 10 µs ([Cyclone, arXiv:2511.15910](https://arxiv.org/pdf/2511.15910)), gate-based ion swap 300 µs ([Moveless, arXiv:2508.03914](https://arxiv.org/pdf/2508.03914)), in-trap repositioning 10 µs (assumed). Shuttling time is always reported; `--objective time` also optimises it.
- **Optimality** (`experiments/optimality_gap.py`): windows captured from every 10th circuit were solved exactly as a MILP (SciPy/HiGHS, 20 s limit) and compared with tabu. Linear machine: tabu reached the proven optimum in 92 of 95 windows, and every plan had the optimal operation count; the 3 energy gaps come only from the small layout tie-break terms. Junction grid: 117 of 119 windows optimal, 1 operation missed in total. The solver is not the bottleneck.
- **D-Wave Leap** (`method="leap"`): optional and untested here. It needs `pip install dwave-system` and a Leap API token, and each cloud call takes seconds while tabu takes milliseconds on these windows.

### 6. Correctness of the Routed Circuit
Physical movement is recorded as identity markers: `IdentityGate(1)` per shuttle and `IdentityGate(2)` per physical ion swap. A physical swap moves ions; it does not exchange their quantum states, so the routed circuit's unitary equals the input's (tested).

---

## 📊 Benchmark Results

All 200 OpenQASM circuits in `tests/`, one home ion per trap (capacity 2). Cost = total physical operations (shuttles + ion swaps); negative change = fewer operations. "Best of both" keeps the cheaper of the two compiled circuits. Reproduce with `python src/benchmark_runner.py --arch <arch>`.

**Junction grid** (linear traps between X-junctions):

| Family | Circuits | Baseline ops | Selective-QUBO ops | Change | Best of both | Better / equal / worse |
|---|---|---|---|---|---|---|
| chain | 38 | 3239 | 2967 | −8.4% | −8.4% | 32 / 6 / 0 |
| random | 39 | 12230 | 11384 | −6.9% | −8.3% | 30 / 1 / 8 |
| layered | 39 | 6960 | 6363 | −8.6% | −8.8% | 34 / 2 / 3 |
| star | 38 | 4136 | 3797 | −8.2% | −8.3% | 34 / 2 / 2 |
| long_range | 39 | 4432 | 4184 | −5.6% | −6.0% | 32 / 5 / 2 |
| other | 7 | 225 | 191 | −15.1% | −15.1% | 3 / 4 / 0 |
| **all** | **200** | **31222** | **28886** | **−7.5%** | **−8.1%** | **165 / 20 / 15** |

**Linear chain**:

| Family | Circuits | Baseline ops | Selective-QUBO ops | Change | Best of both | Better / equal / worse |
|---|---|---|---|---|---|---|
| chain | 38 | 2333 | 1607 | −31.1% | −31.1% | 38 / 0 / 0 |
| random | 39 | 23657 | 23134 | −2.2% | −5.9% | 30 / 1 / 8 |
| layered | 39 | 14110 | 13897 | −1.5% | −4.0% | 25 / 2 / 12 |
| star | 38 | 8653 | 8545 | −1.2% | −1.2% | 33 / 5 / 0 |
| long_range | 39 | 10832 | 10744 | −0.8% | −0.8% | 27 / 10 / 2 |
| other | 7 | 360 | 354 | −1.7% | −1.7% | 3 / 4 / 0 |
| **all** | **200** | **59945** | **58281** | **−2.8%** | **−4.8%** | **156 / 22 / 22** |

**Summary across machines and settings**:

| | Linear | Junction grid | Grid (adapter) |
|---|---|---|---|
| Operations (selective QUBO) | −2.8% | −7.5% | 0.0% |
| Operations, mean ± stdev over 5 solver seeds | −2.6% ± 0.6% | −7.6% ± 0.1% | — |
| Operations (best of both) | −4.8% | −8.1% | 0.0% |
| Shuttling time | −5.0% | −8.1% | 0.0% |
| Circuit depth | −2.4% | −7.3% | 0.0% |
| Shuttling time with `--objective time` | −4.7% | −10.3% | — |
| Operations with `--fast` | −2.8% | −6.6% | — |
| Compile time vs baseline | ~30× (~10× with `--fast`) | ~8× (~3× with `--fast`) | ~1× |

What the numbers say:
- The gain comes from congestion inside linear ion chains: the QUBO plans blockers and the mover together and lets the mover stop in any free slot of the target trap.
- On the adapter `grid`, any free slot of a trap connects to every neighbour, so passing through a trap never collides. The greedy walk is optimal in almost every case and there is nothing to improve.
- Every accepted plan is cheaper where it runs, but a different layout can make the greedy router's later choices costlier, so some circuits end up worse overall. "Best of both" removes those losses at the price of compiling twice.
- Tried and dropped because they made results worse: QUBO choice of which ion moves, lookahead over several upcoming gates, pricing in-trap exchanges as one swap in the QUBO, a shorter time horizon, and a smaller tabu budget.

---

## 📜 References & Acknowledgments

- **SHAW Router**: Bach, Safro, Younis – *"Efficient Compilation for Shuttling Trapped-Ion Machines via the Position Graph Architectural Abstraction"*, arXiv:2501.12470 (ACM TQC 2026).
- **QUBO Formulation**: Glover, Kochenberger, Du – *"A Tutorial on Formulating and Using QUBO Models"*, arXiv:1811.11538 (2019).
- **BQSKit**: Quantum Synthesis & Compilation Toolset, Lawrence Berkeley National Laboratory.
- **dimod / dwave-samplers / neal**: D-Wave Binary Quadratic Model SDK, tabu search and simulated annealing samplers.

---

## 📄 License
This project is open-source under the **MIT License**. See `LICENSE` for details.
