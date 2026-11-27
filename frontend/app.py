"""
QubRoute — Selective-QUBO Trapped-Ion Quantum Shuttling Compiler
Frontend Dashboard (4 tabs: Simulator, Hardware, Benchmarks, Method)
"""

import os
import sys
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
VIEWS_DIR = os.path.join(FRONTEND_DIR, "views")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")

for d in (FRONTEND_DIR, VIEWS_DIR, SRC_DIR, TESTS_DIR, EXPERIMENTS_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from theme import inject_theme_css
from views.simulator import render_simulator_tab
from views.hardware import render_hardware_tab
from views.benchmarks import render_benchmarks_tab
from views.method import render_method_tab

st.set_page_config(
    page_title="QubRoute — Selective-QUBO Trapped-Ion Shuttling",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "qubroute_theme" not in st.session_state:
    try:
        ctx_theme = getattr(getattr(st, "context", None), "theme", None)
        if ctx_theme and getattr(ctx_theme, "type", None) in ("dark", "light"):
            st.session_state["qubroute_theme"] = ctx_theme.type
        else:
            st.session_state["qubroute_theme"] = "dark"
    except Exception:
        st.session_state["qubroute_theme"] = "dark"

inject_theme_css()

with st.sidebar:
    st.markdown("### QubRoute Controls")

    theme_choice = st.radio(
        "Theme Mode",
        options=["Dark (High Contrast)", "Light"],
        index=0 if st.session_state["qubroute_theme"] == "dark" else 1,
        horizontal=True,
    )
    new_theme = "dark" if "Dark" in theme_choice else "light"
    if new_theme != st.session_state["qubroute_theme"]:
        st.session_state["qubroute_theme"] = new_theme
        st.rerun()

    st.markdown("---")

    with st.expander("Hardware Topology", expanded=True):
        architecture = st.selectbox(
            "Architecture",
            options=["junction_grid", "linear", "grid", "grid_dense"],
            index=0,
        )
        st.session_state["architecture"] = architecture
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            num_traps = st.slider("Traps", min_value=2, max_value=12, value=4, step=1)
        with col_t2:
            trap_capacity = st.selectbox("Ions / Trap", options=[1, 2, 3], index=1)

    with st.expander("Congestion & Escalation", expanded=False):
        kappa_th = st.slider("Contention κ threshold", 0.0, 1.0, 0.0, 0.05)
        rho_th = st.slider("Routing Regret ρ threshold", 0.0, 1.0, 0.0, 0.05)
        depth_th = st.number_input("Blockage depth d threshold", min_value=0, max_value=5, value=2, step=1)
        fast_trigger = st.checkbox("Learned trigger (--fast)", value=False)

    with st.expander("QUBO Solver & Objective", expanded=False):
        routing_mode = st.radio("Routing Policy", ["Full QUBO search", "Fast greedy improvement"], index=0)
        solver_engine = st.selectbox(
            "Solver Engine",
            options=[
                "Tabu Search (dwave-samplers)",
                "Simulated Annealing (dwave-neal)",
                "Exact MILP (automatic for ≤16 vars)",
                "D-Wave Leap (Cloud QPU — Disabled)",
            ],
            index=0,
        )
        objective = st.radio("Optimization Objective", ["Operations (unit count)", "Time (hardware µs)"], index=0)
        tabu_effort = st.number_input("Tabu iterations / var", min_value=100, max_value=20000, value=4000, step=500)
        solver_seed = st.number_input("Solver Seed", min_value=0, max_value=9999, value=0, step=1)
        baseline_guard = st.checkbox("Whole-circuit safeguard (portfolio)", value=True)

st.markdown(
    f"""
    <div class="brand-container">
        <div>
            <div class="brand-title">QubRoute</div>
        </div>
        <div class="theme-pill">
            Architecture: <strong>{architecture}</strong> &nbsp;|&nbsp;
            Policy: <strong>{'QUBO' if routing_mode.startswith('Full') else 'Incumbent'}</strong> &nbsp;|&nbsp;
            Objective: <strong>{'Operations' if 'Operations' in objective else 'Time (µs)'}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_sim, tab_hw, tab_bench, tab_method = st.tabs([
    "Simulator",
    "Hardware",
    "Benchmarks",
    "Method",
])

with tab_sim:
    render_simulator_tab(
        architecture=architecture,
        kappa_th=kappa_th,
        rho_th=rho_th,
        depth_th=int(depth_th),
        routing_mode=routing_mode,
        solver_engine=solver_engine,
        tabu_effort=int(tabu_effort),
        solver_seed=int(solver_seed),
        objective=objective,
        baseline_guard=baseline_guard,
        fast_trigger=fast_trigger,
    )

with tab_hw:
    render_hardware_tab(architecture, num_traps, trap_capacity)

with tab_bench:
    render_benchmarks_tab()

with tab_method:
    render_method_tab()
