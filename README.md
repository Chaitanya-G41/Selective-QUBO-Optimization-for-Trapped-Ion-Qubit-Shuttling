# QubRoute: Selective-QUBO Optimization for Trapped-Ion Qubit Shuttling

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: BQSKit](https://img.shields.io/badge/Compiler-BQSKit-purple.svg)](https://bqskit.lbl.gov/)
[![Optimization: dimod](https://img.shields.io/badge/QUBO-dimod%2Fneal-green.svg)](https://github.com/dwavesystems/dimod)
[![UI: Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Tests: 86 Passed](https://img.shields.io/badge/Tests-86%20Passed-brightgreen.svg)](tests/)

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
1. **Continuous Heuristic Routing**: Circuits are routed using fast heuristic algorithms along the architecture's position graph.
2. **Real-Time Contention Monitoring**: When ions encounter blockages, the router computes three congestion metrics: Local Contention ($\kappa$), Routing Regret ($\rho$), and Blockage Depth ($d$).
3. **Selective QUBO Escalation**: When severity exceeds configurable thresholds, the system extracts a **localized subproblem window $W$** around the bottleneck, formulates a targeted Binary Quadratic Model (BQM) with a bounded variable budget, and solves it via Simulated Annealing (or an exact solver).
4. **Physical Feasibility 5-Check Suite**: The decoded trajectory must pass five strict hardware constraints (movement legality, capacity, one-hot occupancy, anti-crossing collision avoidance, and gate feasibility) and demonstrate a strict cost improvement ($C_{\text{QUBO}} < C_{\text{heuristic}}$).
5. **Zero-Deadlock Fallback**: If the QUBO solution is invalid or fails to beat the heuristic, the system safely falls back to a greedy clearing pass or alternate detour.

The prototype includes both a **complete compilation backend** and an **interactive Streamlit web dashboard** for circuit execution and mathematical visualization.

---

## 💻 Source Code Structure

```text
Selective-QUBO-Optimization-for-Trapped-Ion-Qubit-Shuttling/
├── src/                               # Core compilation and optimization engine
│   ├── position_graph.py              # Hardware topology graph (V, E) & ion placement tracking
│   ├── circuit_dag.py                 # Circuit DAG dependency extraction from OpenQASM
│   ├── congestion_handler.py          # Severity metrics (κ, ρ, d), window extraction & trigger logic
│   ├── qubo_formulator.py             # BQM Hamiltonian formulation & Rosenberg quadratization
│   ├── qubo_solver.py                 # Hybrid Exact / Simulated Annealing solver engine (neal/dimod)
│   ├── solution_decoder.py            # Trajectory decoder & 5-check physical feasibility validator
│   ├── shaw_routing_pass.py           # End-to-end BQSKit compilation pass with QUBO escalation
│   ├── qccd_adapter.py                # Hardware architecture adapter for linear QCCD topologies
│   ├── benchmark_runner.py            # Automated runner for OpenQASM benchmark suites
│   └── experiment_tracker.py          # Benchmark metric tracking and iteration history
├── frontend/                          # QubRoute interactive web dashboard
│   ├── app.py                         # Streamlit application (Topology & Live Routing tabs)
│   └── README.md                      # Frontend-specific documentation and usage instructions
├── .streamlit/                        # Streamlit configuration
│   └── config.toml                    # Light theme, color palette, and server configuration
├── tests/                             # Comprehensive test suite (86 passing unit & integration tests)
│   ├── test_benchmark_runner.py       # Benchmark execution and metric validation tests
│   ├── test_circuit_dag.py            # Quantum circuit DAG dependency tests
│   ├── test_congestion_handler.py     # Congestion metrics (κ, ρ, d) & trigger threshold tests
│   ├── test_position_graph.py         # QCCD graph construction & placement tests
│   ├── test_qasm_suite.py             # OpenQASM circuit parsing and routing preservation tests
│   ├── test_qccd_adapter.py           # Adapter integration tests
│   ├── test_qubo_formulator.py        # QUBO variable budgeting, penalty & constraint tests
│   ├── test_qubo_solver.py            # Exact & SA solver verification tests
│   ├── test_shaw_routing_pass.py      # BQSKit compilation pass integration tests
│   └── test_solution_decoder.py       # 5 physical feasibility validation checks & cost tests
├── experiments/                       # Experimental benchmarks and evaluation data
│   ├── results/                       # comparison.csv & experiment_history.json
│   └── plots/                         # Shuttling operation and compilation time plots
├── examples/                          # Standalone demonstration scripts
│   ├── demo_selective_qubo.py         # End-to-end prototype walkthrough in terminal
│   └── demo_position_graph.py         # QCCD architecture topology demo
├── requirements.txt                   # Production and testing dependencies
└── README.md                          # Project documentation
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
Run the full 86-test automated test suite:
```bash
python -m pytest -v
```
All 86 unit and integration tests should pass.

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
  - **QASM Benchmark Execution**: Select from 10 real benchmark circuits, inspect their raw OpenQASM 3.0 source code, compile in real-time, and view QUBO acceptance rates, shuttling hop counts, and ion swaps.
  - **Single Window Deep Dive**: Step through a single congestion bottleneck to observe variable budgeting, simulated annealing convergence, and the 5-point physical validation suite.

### Option B: Terminal Prototype Demonstration
Run the standalone end-to-end compiler demonstration script:
```bash
python examples/demo_selective_qubo.py
```

### Option C: Automated Benchmark Suite
Run the comparative benchmarking suite (Baseline Heuristic vs. Hybrid Selective-QUBO):
```bash
python src/benchmark_runner.py --mode both
```
Results and plots will be written to `experiments/results/comparison.csv` and `experiments/plots/`.

---

## ⚙️ Implementation Details

### 1. Architectural Abstraction: The Position Graph ($G_p$)
The QCCD hardware is abstracted as an undirected graph $G_p = (V, E)$ (`src/position_graph.py`):
- **Trap Zones**: Contain discrete slot nodes $\{v_1, \dots, v_k\}$ connected as a fully connected clique (representing within-trap ion rotation/swapping).
- **Transport Segments**: Transition nodes connecting adjacent traps via merge and split operations.
- **Capacity Function $\text{cap}(v)$**: Trap slots hold at most 1 ion; transport channels strictly forbid multi-ion occupancy.
- **Ion Placement ($\phi$)**: Tracks the instantaneous mapping of qudit $q_i \mapsto v \in V$.

### 2. Congestion Severity Metrics & Trigger Condition
When an ion's shortest shuttling path $P = (u_0, u_1, \dots, u_m)$ encounters stationary or competing ions, the router computes three localized metrics (`src/congestion_handler.py`):
1. **Local Contention Index ($\kappa$)**:
   $$\kappa = \frac{\sum_{v \in P} \mathbb{I}(\text{occupied}(v))}{|P|}$$
2. **Routing Regret ($\rho$)**:
   $$\rho = \frac{C_{\text{detour}} - |P|}{|P|}$$
3. **Blockage Depth ($d$)**: The cascade chain length of secondary blockers that would also need to be displaced to free the primary path.

**Trigger Decision**:
$$\text{Trigger QUBO} \iff (\kappa > \kappa_{\text{th}} \land \rho > \rho_{\text{th}}) \lor d > d_{\text{max}}$$
*(Default thresholds: $\kappa_{\text{th}} = 0.20, \rho_{\text{th}} = 0.20, d_{\text{max}} = 0$)*.

### 3. Local Window Extraction ($W$) & Variable Budget
Instead of global compilation, the router bounds the problem to an induced subgraph $W \subseteq G_p$ of radius $r=1$ around the contested hops:
- Only ions located inside $W$ are designated as active decision variables.
- Ions outside $W$ are pinned as static hardware obstacles.
- A **soft budget limit of 100 variables** ensures annealing convergence within milliseconds ($< 30\text{ ms}$).

### 4. Mathematical QUBO Formulation
Binary decision variables $x_{i, v, t} \in \{0, 1\}$ represent whether ion $i$ is at node $v$ at timestep $t \in [0, T]$ (`src/qubo_formulator.py`):

$$\min_{\mathbf{x}} H(\mathbf{x}) = H_{\text{cost}} + \lambda \left( H_{\text{init}} + H_{\text{one\_hot}} + H_{\text{capacity}} + H_{\text{movement}} + H_{\text{crossing}} + H_{\text{goal}} \right)$$

| Component | Mathematical Expression | Physical Purpose |
|---|---|---|
| **Objective ($H_{\text{cost}}$)** | $\sum_{t=1}^T \sum_{(u,v) \in E} x_{i,u,t-1} x_{i,v,t}$ | Minimizes total shuttling transport hops |
| **Initial Placement ($H_{\text{init}}$)** | $\sum_{i} (1 - x_{i, \phi(i), 0})^2$ | Pins each ion to its initial position at $t=0$ |
| **One-Hot ($H_{\text{one\_hot}}$)** | $\sum_{i, t} \left(\sum_{v \in W} x_{i,v,t} - 1\right)^2$ | Ensures each ion occupies exactly one node per timestep |
| **Capacity ($H_{\text{capacity}}$)** | $\sum_{v, t} \sum_{i \neq j} x_{i,v,t} x_{j,v,t}$ | Prevents more than one ion in a capacity-1 slot |
| **Movement Legality ($H_{\text{movement}}$)** | $\sum_{i, t} \sum_{(u,w) \notin E} x_{i,u,t} x_{i,w,t+1}$ | Penalizes invalid teleportation across non-adjacent nodes |
| **Anti-Crossing ($H_{\text{crossing}}$)** | $\sum_{(u,v) \in E, t, i < j} x_{i,u,t} x_{i,v,t+1} x_{j,v,t} x_{j,u,t+1}$ | Prevents ions from swapping positions across the same segment |
| **Goal Achievement ($H_{\text{goal}}$)** | $(1 - x_{\text{moving}, v_{\text{target}}, T})^2$ | Incentivizes the target qubit to arrive at the gate zone |

*Higher-order terms in $H_{\text{crossing}}$ are quadratized into 2-body interactions using **Rosenberg auxiliary variables**.*

### 5. Physical Feasibility 5-Check Validation Suite
Before accepting any QUBO trajectory, the solution decoder (`src/solution_decoder.py`) verifies:
1. **Movement Legality**: Every step strictly follows hardware edges in $E(G_p)$.
2. **Trap Capacity**: No node occupancy exceeds $\text{cap}(v)$ at any timestep.
3. **One-Hot Occupancy**: Every active ion has valid spatial positions at every $t$.
4. **Collision Avoidance**: No simultaneous anti-crossing transitions on identical edges.
5. **Gate Feasibility**: The moving ion successfully arrives at its destination zone within horizon $T$.

**Margin Gating**:
$$\text{Accepted} \iff \text{All 5 Checks PASS} \land C_{\text{QUBO}} < C_{\text{heuristic}}$$

If any constraint fails or the QUBO path is not strictly better than the heuristic, the solution is **safely rejected**, and the router executes the greedy fallback with zero deadlocks.

---

## 📊 Benchmark Highlights

Evaluated across OpenQASM benchmark circuits on linear QCCD architectures:

| Benchmark Circuit | Qubits | Gates | Congestion Events | QUBO Triggers | QUBO Acceptance Rate | Cost Improvement |
|---|---|---|---|---|---|---|
| `168_random_7q.qasm` | 7 | 10 | 10 | 10 | **60.0%** (6/10) | **+69.2%** |
| `043_chain_16q.qasm` | 16 | 20 | 18 | 9 | **55.6%** (5/9) | **+75.0%** |
| `031_chain_17q.qasm` | 17 | 20 | 18 | 9 | **55.6%** (5/9) | **+75.0%** |
| `019_chain_18q.qasm` | 18 | 20 | 18 | 9 | **55.6%** (5/9) | **+75.0%** |
| `192_random_5q.qasm` | 5 | 12 | 14 | 12 | **41.7%** (5/12) | **+68.3%** |
| `090_long_range_5q.qasm` | 5 | 8 | 1 | 1 | **100.0%** (1/1) | **+75.0%** |

- **Up to 75% Shuttling Cost Reduction** on highly congested interaction paths compared to greedy clearing.
- **100% Deadlock-Free Guarantee** via safe automated fallback.
- **Bounded Compilation Latency**: Subproblem annealing converges in under 30 ms per trigger.

---

## 📜 References & Acknowledgments

- **SHAW Router**: Bach, Safro, Younis – *"Efficient Compilation for Shuttling Trapped-Ion Machines via the Position Graph Architectural Abstraction"*, arXiv:2501.12470 (ACM TQC 2026).
- **QUBO Formulation**: Glover, Kochenberger, Du – *"A Tutorial on Formulating and Using QUBO Models"*, arXiv:1811.11538 (2019).
- **BQSKit**: Quantum Synthesis & Compilation Toolset, Lawrence Berkeley National Laboratory.
- **dimod / neal**: D-Wave Systems Binary Quadratic Model SDK and Simulated Annealing Sampler.

---

## 📄 License
This project is open-source under the **MIT License**. See `LICENSE` for details.
