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

### Tab 2: Live Routing Pipeline & Congestion Analysis
- Pre-configured routing scenarios (high contention, uncongested path, cascaded blockers).
- Dynamic calculation of contention severity metrics ($\kappa, \rho, d$).
- Automated decision trigger: Fast Heuristic vs Selective QUBO Escalation.
- Sub-problem window extraction ($W$) and QUBO problem size budgeting ($\le 100$ variables).
- Real-time solver execution with `SimulatedAnnealingSampler` (neal) or `ExactSolver` (dimod).
- 5-point physical feasibility validation checklist and decoded shuttling trajectories.
