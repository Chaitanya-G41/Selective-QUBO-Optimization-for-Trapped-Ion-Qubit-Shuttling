"""
QubRoute — Selective-QUBO Trapped-Ion Quantum Shuttling Compiler
Frontend Dashboard
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
VIEWS_DIR = os.path.join(FRONTEND_DIR, "views")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")

for d in (FRONTEND_DIR, VIEWS_DIR, SRC_DIR, TESTS_DIR, EXPERIMENTS_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import streamlit as st
from theme import inject_theme_css, get_current_theme

# Import Views
from views.overview import render_overview_tab
from views.hardware import render_hardware_tab
from views.live_routing import render_live_routing_tab
from views.deep_dive import render_deep_dive_tab
from views.qubo_formulation import render_qubo_tab
from views.benchmarks import render_benchmarks_tab
from views.optimality import render_optimality_tab
from views.method_references import render_method_tab

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QubRoute — Selective-QUBO Trapped-Ion Shuttling",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for theme
if "qubroute_theme" not in st.session_state:
    st.session_state["qubroute_theme"] = "dark"

inject_theme_css()
theme = get_current_theme()
is_dark = st.session_state["qubroute_theme"] == "dark"

# ---------------------------------------------------------------------------
# Sidebar Control Panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### QubRoute Controls")

    # Theme switcher
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

    # Hardware Architecture
    with st.expander("Hardware Topology", expanded=True):
        architecture = st.selectbox(
            "Architecture",
            options=["junction_grid", "linear", "grid", "grid_dense"],
            index=0,
            help=(
                "linear: traps in a 1D chain joined by segments;\n"
                "junction_grid: linear traps on edges of 2D X-junction grid;\n"
                "grid: 2D adapter grid with full slot connectivity;\n"
                "grid_dense: 2D dense grid with 2 ions per capacity-3 trap."
            ),
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            num_traps = st.slider("Traps", min_value=2, max_value=12, value=4, step=1)
        with col_t2:
            trap_capacity = st.selectbox("Ions / Trap", options=[1, 2, 3], index=1)

    # Congestion Triggers
    with st.expander("Congestion & Escalation", expanded=False):
        st.caption("Trigger condition: (κ > κ_th ∧ ρ > ρ_th) ∨ d > d_th")
        kappa_th = st.slider(
            "Contention κ threshold",
            0.0, 1.0, 0.0, 0.05,
            help="Fraction of occupied intermediate positions on the shortest path.",
        )
        rho_th = st.slider(
            "Routing Regret ρ threshold",
            0.0, 1.0, 0.0, 0.05,
            help="Estimated extra detours relative to lower bound: ρ = 2|B(P)| / C_LB.",
        )
        depth_th = st.number_input(
            "Blockage depth d threshold",
            min_value=0, max_value=5, value=2, step=1,
            help="Levels of occupied positions clustered behind blockers.",
        )
        fast_trigger = st.checkbox(
            "Learned trigger (--fast)",
            value=False,
            help="Depth-2 decision tree filter: skip QUBO if greedy slack over lower bound is < 2 (small) or < 4 (large).",
        )

    # QUBO Solver Configuration
    with st.expander("QUBO Solver & Objective", expanded=False):
        routing_mode = st.radio(
            "Routing Policy",
            ["Full QUBO search", "Fast greedy improvement"],
            index=0,
            help="Full QUBO solves localized BQMs. Fast greedy improvement uses deterministic vacancy chain clearing.",
        )
        solver_engine = st.selectbox(
            "Solver Engine",
            options=[
                "Tabu Search (dwave-samplers)",
                "Simulated Annealing (dwave-neal)",
                "Exact MILP (automatic for ≤16 vars)",
                "D-Wave Leap (Cloud QPU — Disabled)",
            ],
            index=0,
            help="Tabu search is the default benchmark solver with warm-start support.",
        )
        if "Leap" in solver_engine:
            st.warning("D-Wave Leap requires cloud authentication and the dwave-system package. Classical tabu runs in milliseconds.")

        objective = st.radio(
            "Optimization Objective",
            ["Operations (unit count)", "Time (hardware µs)"],
            index=0,
            help="Operations: 1 op per shuttle/swap. Time: Cyclone & Moveless model (split/merge 80 µs, transport 10 µs, swap 300 µs, in-trap 10 µs).",
        )

        tabu_effort = st.number_input(
            "Tabu iterations / var",
            min_value=100, max_value=20000, value=4000, step=500,
        )
        solver_seed = st.number_input("Solver Seed", min_value=0, max_value=9999, value=0, step=1)
        baseline_guard = st.checkbox(
            "Whole-circuit safeguard (portfolio)",
            value=True,
            help="Keeps the cheaper of the independent baseline and selective QUBO routes under the configured objective.",
        )

# ---------------------------------------------------------------------------
# Header (Brand presentation)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="brand-container">
        <div>
            <div class="brand-title">QubRoute</div>
            <div class="brand-sub">Selective-QUBO Routing for Trapped-Ion Quantum Architectures</div>
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

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
(
    tab_overview,
    tab_hardware,
    tab_live_routing,
    tab_deep_dive,
    tab_qubo,
    tab_benchmarks,
    tab_optimality,
    tab_methods,
) = st.tabs([
    "Overview",
    "Hardware",
    "Live Routing",
    "Single Window Deep Dive",
    "QUBO Formulation",
    "Benchmarks",
    "Optimality",
    "Method & References",
])

with tab_overview:
    render_overview_tab(architecture)

with tab_hardware:
    render_hardware_tab(architecture, num_traps, trap_capacity)

with tab_live_routing:
    render_live_routing_tab(
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

with tab_deep_dive:
    render_deep_dive_tab(
        kappa_th=kappa_th,
        rho_th=rho_th,
        depth_th=int(depth_th),
        tabu_effort=int(tabu_effort),
        solver_seed=int(solver_seed),
    )

with tab_qubo:
    render_qubo_tab()

with tab_benchmarks:
    render_benchmarks_tab()

with tab_optimality:
    render_optimality_tab()

with tab_methods:
    render_method_tab()
