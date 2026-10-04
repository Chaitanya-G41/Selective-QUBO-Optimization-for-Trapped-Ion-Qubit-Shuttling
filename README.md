# Selective QUBO Optimization for Trapped-Ion Qubit Shuttling

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: BQSKit](https://img.shields.io/badge/Compiler-BQSKit-purple.svg)](https://bqskit.lbl.gov/)
[![Optimization: dimod](https://img.shields.io/badge/QUBO-dimod%2Fneal-green.svg)](https://github.com/dwavesystems/dimod)

A hybrid quantum compilation framework that combines fast greedy heuristic routing (SHAW) with local **Quadratic Unconstrained Binary Optimization (QUBO)** to resolve severe congestion bottlenecks in Trapped-Ion Quantum Charge-Coupled Device (QCCD) architectures.

---

## 📌 Problem Overview & Motivation

Trapped-ion quantum computers feature exceptionally high gate fidelities and long coherence times. However, multi-qubit entangling gates require physical ions to be shuttled into shared trap zones on a **QCCD (Quantum Charge-Coupled Device)** architecture.

### The Shuttling Bottleneck
- **Ion Heating & Decoupling**: Frequent shuttling operations introduce kinetic heating and transport latency.
- **Local Congestion & Deadlocks**: Simple greedy routing algorithms (e.g. A* or basic SHAW) struggle when multiple ions compete for narrow transport channels and trap slots, leading to routing delays or capacity deadlocks.
- **Global QUBO Unscalability**: Encoding the entire shuttling routing problem across a full chip into a single global QUBO matrix scales exponentially with timesteps and positions ($O(N_{ions} \times N_{pos} \times T)$), making classical simulated annealing or QPU execution impractically slow.

### Our Solution: Selective QUBO Optimization
Rather than using QUBO globally, our framework monitors local routing severity metrics ($\kappa$, $\rho$, $d$). When local congestion crosses critical thresholds, the system extracts a **bounded local congestion window $W$** around the blocked region, formulates a targeted QUBO problem for only the active local ions over a compact time horizon $T$, solves it via Simulated Annealing (or Exact Solvers), validates physical feasibility, and seamlessly merges the optimal local routing trajectory back into the global compilation pass.

---

## 🏗️ System Architecture & Workflow

The framework implements a complete 9-phase compilation and optimization pipeline:

```text
  +-----------------------------------------------------------------------+
  |                     QASM Quantum Circuit                              |
  +-----------------------------------------------------------------------+
                                      |
                                      v
  +-----------------------------------------------------------------------+
  | Phase 1 & 2: Circuit DAG Parsing & QCCD Position Graph Placement      |
  +-----------------------------------------------------------------------+
                                      |
                                      v
  +-----------------------------------------------------------------------+
  | Phase 3: Congestion Severity Analysis                                  |
  | Compute metrics: Local Contention (κ), Routing Regret (ρ), Depth (d) |
  +-----------------------------------------------------------------------+
                                      |
                Is Congestion Severe? ( (κ > κ_th AND ρ > ρ_th) OR d > d_th )
                                    /   \
                             YES   /     \  NO
                                  v       v
  +-----------------------------------+  +--------------------------------+
  | Phase 5: Extract Local Window W   |  | Fallback: Greedy SHAW Routing  |
  +-----------------------------------+  +--------------------------------+
                    |                                     |
                    v                                     |
  +-----------------------------------+                   |
  | Phase 6: Formulate QUBO Matrix    |                   |
  | H = H_cost + λ(H_init + H_onehot  |                   |
  |   + H_cap + H_move + H_cross)     |                   |
  +-----------------------------------+                   |
                    |                                     |
                    v                                     |
  +-----------------------------------+                   |
  | Phase 7: QUBO Solver (Exact/SA)   |                   |
  +-----------------------------------+                   |
                    |                                     |
                    v                                     |
  +-----------------------------------+                   |
  | Phase 8 & 9: Decode & Validate    |                   |
  | Checks: Movement, Occupancy,      |                   |
  | Capacity, Anti-Crossing, Goal     |                   |
  +-----------------------------------+                   |
             /             \                              |
      Valid /               \ Invalid / Worse             |
           v                 v                            |
  +-----------------+  +----------------------------------+
  | Accept QUBO     |  | Safe Fallback: Greedy Clearer    |
  | Trajectories    |  | / Alternate Unblocked Path       |
  +-----------------+  +----------------------------------+
           \                 /
            v               v
  +-----------------------------------------------------------------------+
  |                     Final Routed QASM Circuit                         |
  +-----------------------------------------------------------------------+
```

---

## 🧮 Mathematical QUBO Formulation

For a local congestion window $W$ with active ions $i \in I_{\text{active}}$, positions $v \in V_W$, and timesteps $t \in [0, T]$, we introduce binary decision variables:
$$x_{i, v, t} \in \{0, 1\} \quad \text{where } x_{i,v,t} = 1 \iff \text{ion } i \text{ is at position } v \text{ at timestep } t$$

The combined Hamiltonian minimized by the solver is:
$$H = H_{\text{cost}} + \lambda \left( H_{\text{init}} + H_{\text{one\_hot}} + H_{\text{capacity}} + H_{\text{movement}} + H_{\text{crossing}} + H_{\text{goal}} \right)$$

1. **Hard Initial Position Pinning ($H_{\text{init}}$)**: Forces each ion to start at its known initial position at $t=0$:
   $$H_{\text{init}} = \lambda \sum_{i} (1 - x_{i, s_i, 0})$$
2. **One-Hot Location Constraint ($H_{\text{one\_hot}}$)**: Ensures each ion occupies exactly one position per timestep:
   $$H_{\text{one\_hot}} = \lambda \sum_{i, t} \left( \sum_{v} x_{i, v, t} - 1 \right)^2$$
3. **Capacity Constraint ($H_{\text{capacity}}$)**: Enforces node capacity limits $\text{cap}(v)$:
   $$H_{\text{capacity}} = \lambda \sum_{v, t} \sum_{i \neq j} x_{i,v,t} x_{j,v,t} \quad (\text{for } \text{cap}(v)=1)$$
4. **Movement Legality Constraint ($H_{\text{movement}}$)**: Penalizes illegal hops between non-adjacent hardware nodes:
   $$H_{\text{movement}} = \lambda \sum_{i, t, v} \sum_{w \notin E(v)} x_{i,v,t} x_{i,w,t+1}$$
5. **Anti-Crossing Penalty ($H_{\text{crossing}}$)**: Prevents physical ion crossing (simultaneous opposite movement across an edge):
   $$H_{\text{crossing}} = \lambda \sum_{(u,v) \in E} \sum_{t} \sum_{i < j} \left( x_{i,u,t} x_{i,v,t+1} x_{j,v,t} x_{j,u,t+1} \right)$$
6. **Goal Target Constraint ($H_{\text{goal}}$)**: Rewards moving ion for reaching its target position at $t=T$:
   $$H_{\text{goal}} = \lambda (1 - x_{\text{moving}}, v_{\text{target}}, T)$$

---

## 📁 Repository Structure

```text
Selective-QUBO-Optimization-for-Trapped-Ion-Qubit-Shuttling/
├── src/
│   ├── position_graph.py       # Hardware topology (PositionGraph & Placement)
│   ├── circuit_dag.py          # Circuit DAG dependency extractor
│   ├── congestion_handler.py   # Severity metrics (κ, ρ, d) & local window extraction
│   ├── qubo_formulator.py      # BQM formulation & Rosenberg quadratization
│   ├── qubo_solver.py          # Exact & Simulated Annealing solvers (dimod/neal)
│   ├── solution_decoder.py     # Trajectory decoder & physical validation (5 checks)
│   ├── shaw_routing_pass.py    # Primary BQSKit compilation pass
│   ├── qccd_adapter.py         # Hardware architecture adapter
│   ├── benchmark_runner.py     # Automated benchmark suite runner
│   └── experiment_tracker.py   # Code progression & benchmark tracker
├── tests/                      # Pytest suite (77+ unit & integration tests)
├── experiments/                # Benchmark outputs, CSV results, & plots
│   ├── results/                # comparison.csv & experiment_history.json
│   └── plots/                  # Generated performance comparison graphs
├── examples/                   # Demonstration scripts & visual examples
├── conftest.py                 # Pytest automated logging hook
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Step-by-Step Setup & Installation Guide

### Prerequisites
- Python **3.10+** (Tested on Python 3.10.11)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Chaitanya-G41/Selective-QUBO-Optimization-for-Trapped-Ion-Qubit-Shuttling.git
cd Selective-QUBO-Optimization-for-Trapped-Ion-Qubit-Shuttling
```

### 2. Create & Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Verification & Running Tests

Run the full automated test suite (77 unit tests covering formulator, solver, decoder, and routing passes):

```bash
pytest -v
```

Run individual module smoke tests:
```bash
python src/qubo_formulator.py    # Smoke test for QUBO Matrix construction
python src/qubo_solver.py        # Smoke test for Exact & SA solvers
python src/solution_decoder.py   # Smoke test for Decoder validation
```

---

## 📊 Running Benchmarks & Tracking Experiments

### 1. Run Baseline vs Hybrid Benchmark Suite
Run the full comparative benchmark against OpenQASM benchmark circuits:
```bash
python src/benchmark_runner.py --mode both
```
This automatically generates:
- `experiments/results/comparison.csv`: Full metric log (shuttle hops, SWAP count, compile time).
- `experiments/plots/shuttling_ops.png`: Visual comparison bar charts.
- `experiments/plots/compile_time.png`: Compilation time overhead comparison.

### 2. Track System Iteration History
View recorded experiment snapshots across iterations of your code:
```bash
python src/experiment_tracker.py --history
```

Compare two specific runs:
```bash
python src/experiment_tracker.py --compare run_001 run_002
```

---

## 📜 References & Acknowledgments

- **SHAW Router**: Bach, Safro, Younis – *"Efficient Compilation for Shuttling Trapped-Ion Machines via the Position Graph Architectural Abstraction"*, arXiv:2501.12470 (ACM TQC 2026).
- **QUBO Formulation**: Glover, Kochenberger, Du – *"A Tutorial on Formulating and Using QUBO Models"*, arXiv:1811.11538 (2019).
- **BQSKit**: Quantum Synthesis & Compilation Toolset, Lawrence Berkeley National Laboratory.

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for details.
