# QubRoute Frontend Inventory

Inventory of all pages, sidebar controls, widgets, tables, charts, and text blocks across the current frontend (`frontend/app.py`, `frontend/views/*.py`, `frontend/circuit_sim_ui.py`) and the original `main:frontend/app.py`.

---

## 1. Global Shell & Sidebar (`frontend/app.py`)
- **Page Config**: `st.set_page_config(page_title="QubRoute — Selective-QUBO Trapped-Ion Shuttling", layout="wide", initial_sidebar_state="expanded")`
- **Header**: `.brand-container` with `QubRoute` title, subtitle, and `.theme-pill` showing active Architecture, Policy, Objective.
- **Sidebar (`QubRoute Controls`)**:
  - `st.radio`: Theme Mode (`Dark (High Contrast)`, `Light`) — defaulting to device preference (`prefers-color-scheme`)
  - `st.expander("Hardware Topology", expanded=True)`:
    - `st.selectbox`: Architecture (`junction_grid`, `linear`, `grid`, `grid_dense`)
    - `st.slider`: Traps (`2–12`, default `4`, covers original `2–8`)
    - `st.selectbox`: Ions / Trap (`1, 2, 3`, default `2`, covers original `1, 2`)
  - `st.expander("Congestion & Escalation", expanded=False)`:
    - `st.caption`: Trigger formula `(κ > κ_th ∧ ρ > ρ_th) ∨ d > d_th`
    - `st.slider`: Contention κ threshold (`0.0–1.0`, default `0.0`)
    - `st.slider`: Routing Regret ρ threshold (`0.0–1.0`, default `0.0`)
    - `st.number_input`: Blockage depth d threshold (`0–5`, default `2`)
    - `st.checkbox`: Learned trigger (`--fast`)
  - `st.expander("QUBO Solver & Objective", expanded=False)`:
    - `st.radio`: Routing Policy (`Full QUBO search`, `Fast greedy improvement`)
    - `st.selectbox`: Solver Engine (`Tabu Search (dwave-samplers)`, `Simulated Annealing (dwave-neal)`, `Exact MILP (automatic for ≤16 vars)`, `D-Wave Leap (Cloud QPU — Disabled)`)
    - `st.radio`: Optimization Objective (`Operations (unit count)`, `Time (hardware µs)`)
    - `st.number_input`: Tabu iterations / var (`100–20000`, default `4000`)
    - `st.number_input`: Solver Seed (`0–9999`, default `0`)
    - `st.checkbox`: Whole-circuit safeguard (portfolio) (default `True`)
- **Top-Level Tabs (`st.tabs`)** (must fit on one line at 1440 px):
  1. `Overview`
  2. `Hardware`
  3. `Live Routing`
  4. `Single Window Deep Dive`
  5. `QUBO Formulation`
  6. `Benchmarks`
  7. `Optimality`
  8. `Method & References`

---

## 2. Tab 1: Overview (`frontend/views/overview.py`)
- **Headline Metrics (`st.metric` × 5)**:
  - `Benchmark Circuits` (`len(jg_df)`)
  - `Junction Grid Baseline` (`jg_base_ops ops`)
  - `Selective QUBO` (`jg_hyb_ops ops`, delta `-% operations`)
  - `Win / Tie / Loss` (`better / equal / worse`)
  - `Accepted QUBO Plans` (`accepted / blocked`, delta accept rate)
- **Real-Circuit Selector + `exact_sim` Animation (`circuit_sim_ui.py`)**:
  - `st.selectbox`: Filter Family (`all`, `curated`, `chain`, `random`, `layered`, `star`, `long_range`, `other`)
  - `st.selectbox`: Select Circuit (searchable, all 200 `.qasm` circuits)
  - `st.selectbox`: Congestion Events stepper (`Event i / N`) + `st.metric("Events in Circuit")`
  - `components.html(render_exact_sim(...), height=EXACT_SIM_HEIGHT, scrolling=True)` (auto-plays on open)
  - Verdict box: 4 `st.metric` (Event Greedy, Event QUBO, Circuit Total, Trigger) + `st.caption` + verdict badge
- **Text Blocks**:
  - `Local Implementation Update (23 November 2026)` bordered card
  - `The Trapped-Ion Shuttling Bottleneck` (3 bordered cards: `Ion Heating & Dephasing`, `Congestion & Deadlocks`, `Global QUBO Scaling Barrier` with LaTeX)
  - `The QubRoute Solution: Selective QUBO Optimization` (6 `.pipeline-step` cards)
  - `Framework Capabilities & Core Modules` (3-column bordered card)

---

## 3. Tab 2: Hardware (`frontend/views/hardware.py`)
- **Hardware Metrics (`st.metric` × 5)**:
  - `Architecture`, `Positions |V|`, `Transitions |E|`, `Active Qubits`, `Trap Capacity`
- **Animated Placement Map**:
  - `components.html(render_exact_sim(...), height=EXACT_SIM_HEIGHT, scrolling=True)` with `positions=positions_for_exact_sim(pg, architecture)`
- **Tables (`st.dataframe` × 2)**:
  - `Trap & Slot Configuration` (`Trap ID`, `Slots`, `Capacity`, `Occupied`)
  - `Initial Qubit Placements (Mapping φ)` (`Qubit`, `Physical Node`, `Zone Type`)
- **Text Blocks**:
  - `Architectural Abstraction: The Position Graph (G_p)` (2 bordered cards: `Graph Elements & Edge Labels`, `Machine Models Evaluated in Benchmarks`)

---

## 4. Tab 3: Live Routing (`frontend/views/live_routing.py`)
- **Circuit Selector & Controls**:
  - `st.selectbox`: Filter Family (`all`, `curated`, `chain`, `random`, `layered`, `star`, `long_range`, `other`)
  - `st.selectbox`: Select Circuit (all 200 `.qasm` circuits)
  - `st.expander("View OpenQASM 3.0 Source: ...")` with `st.code(..., language="qasm")`
  - `st.button("Compile Baseline & Selective QUBO", type="primary")`
- **Compilation Results**:
  - `st.metric` × 6: `Qubits`, `Original Gates`, `Baseline Ops`, `Selective-QUBO Ops`, `Shuttling Time`, `QUBO Accepted`
  - `st.caption`: Selected route, blocked congestion events, safeguard status
  - `st.dataframe` (`Routing Performance Log` with Shuttle Hops, Ion Swaps, Total Ops, Shuttling Time µs, Circuit Depth)
  - `Physical Movement Traces Summary` + `Gate Sequence` (`st.dataframe` of routed circuit operations, preserved from `main:frontend/app.py`)
- **Animated Congestion Event Replay + Verdict (`circuit_sim_ui.py`)**:
  - `st.selectbox`: Congestion Events stepper + `st.metric("Events in Circuit")`
  - `components.html(render_exact_sim(...), height=EXACT_SIM_HEIGHT, scrolling=True)`
  - Data-only verdict box (`st.metric` × 4 + `st.caption` + badge)

---

## 5. Tab 4: Single Window Deep Dive (`frontend/views/deep_dive.py`)
- **Scenario Selector**:
  - `st.selectbox`: Select Shuttling Scenario (8 scenarios, including all 4 from `main:frontend/app.py`)
- **1. Real-Time Contention Monitoring**:
  - `st.metric` × 4: `Local Contention (κ)`, `Routing Regret (ρ)`, `Blockage Depth (d)`, `Greedy Walk Cost`
  - Trigger condition evaluation card
- **2. Local Window (W) & Lower Bound Filter**:
  - Window details bullets or skip badge
- **3. Time-Expanded QUBO Formulation & Candidate Validation**:
  - `st.dataframe`: `BQM Model Parameters`
  - `st.dataframe`: `Physical Constraints Check (Per-Sample Validation)`
  - `st.expander("Inspect Time-Expanded Occupancy Grid x(i, v, t)")` with `st.dataframe`
- **4. Greedy Walk vs Selective-QUBO Move Plan**:
  - `st.dataframe` × 2: `Greedy Walk Baseline` and `Selective QUBO Plan`
  - Acceptance verdict badge
- **5. Interactive Animated Simulation**:
  - `components.html(render_exact_sim(...), height=EXACT_SIM_HEIGHT, scrolling=True)`

---

## 6. Tab 5: QUBO Formulation (`frontend/views/qubo_formulation.py`)
- **Complete Shuttling Hamiltonian**: `st.latex` + description
- **Hamiltonian Terms Breakdown**: `st.dataframe` (7 terms)
- **Variable Pruning & Rosenberg Reduction**: 2 bordered cards with LaTeX
- **Live BQM Size & Sparsity Inspection**: `st.metric` × 4 (`Unpruned Variables`, `Pruned Variables x(i,v,t)`, `Rosenberg Aux Variables`, `Quadratic Couplings`)

---

## 7. Tab 6: Benchmarks (`frontend/views/benchmarks.py`)
- **Filters**: `st.selectbox` × 2 (`Select Architecture`, `Select Family Filter`)
- **Tables**:
  - `st.dataframe`: `Per-Family Results` (computed from CSV total ops, ignoring `verdict` column)
  - `st.dataframe`: `Summary Across Machines & Solver Settings`
- **Text Block**: `What the Experimental Numbers Say`
- **Visual Benchmark Analytics (`st.tabs` × 4 + `st.pyplot` × 4)**:
  - `Operations by Family`, `Per-Circuit Change (%)`, `Compile Time Scaling`, `Acceptance vs Contention`
- **Filtered Circuit Run Data**: `st.dataframe` + `st.download_button`

---

## 8. Tab 7: Optimality (`frontend/views/optimality.py`)
- **Headline Metrics (`st.metric` × 3)**: `Linear Chain Optimality`, `Junction Grid Optimality`, `Solver Bottleneck`
- **Text Block**: `Mathematical Gap & Solver Comparison`
- **Interactive Verification Tool**:
  - `st.button("Run Exact MILP vs Tabu on Active Window")`
  - `st.metric` × 6 (Tabu Energy, Latency, Plan Cost; MILP Ground Energy, Latency, Plan Cost) + gap badge

---

## 9. Tab 8: Method & References (`frontend/views/method_references.py`)
- **Text & Tables**:
  - `1. Correctness of the Routed Circuit & Unitary Invariance`
  - `2. Physical Duration Cost Models (TIME_COSTS_US)` + `st.dataframe`
  - `3. Learned Trigger (--fast)`
  - `4. Ablation: What Was Tried and Dropped`
  - `5. Cloud QPU Scaling & D-Wave Leap Note`
  - `6. Academic References & Acknowledgments`
  - `st.expander("View Open-Source License (MIT)")`
