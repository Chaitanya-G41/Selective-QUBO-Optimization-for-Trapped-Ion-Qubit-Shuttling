# Frontend Dashboard: Selective-QUBO Trapped-Ion Shuttling Router

Interactive presentation dashboard built with Streamlit for demonstration, college presentations, and hackathon evaluation.

## How to Run

From the project root:

```powershell
.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Or if your virtual environment is already active in PowerShell:

```powershell
streamlit run frontend/app.py
```

The web dashboard will automatically open in your browser at `http://localhost:8501`.

## Implemented Tabs

### Tab 1: Hardware Topology & Ion Layout
- Configurable QCCD hardware parameters (number of traps, trap capacity).
- Live visualization of the architectural Position Graph $G_p = (V, E)$ (slots, segments, placed ions).
- Trap and slot occupancy breakdown tables.

### Tab 2: Live Routing & Benchmarks
- **QASM Benchmark Execution**: pick a circuit and a machine (linear, junction grid, grid) and compile it with the baseline and with selective QUBO, side by side (operations, triggers, accepted plans, compile time).
- **Single Window Deep Dive**: one congestion event step by step: severity metrics ($\kappa, 
ho, d$), trigger and filter decisions, the window $W$, QUBO size, solver and rejected candidates, and the greedy walk vs the QUBO plan.
- Sidebar: trigger thresholds and solver settings (tabu search or simulated annealing).
