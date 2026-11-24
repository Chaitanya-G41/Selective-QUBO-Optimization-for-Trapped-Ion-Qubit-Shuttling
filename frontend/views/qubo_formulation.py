"""
frontend/views/qubo_formulation.py
QUBO Formulation Tab: Complete mathematical Hamiltonian, term breakdown,
variable reachability pruning, Rosenberg reduction, and live BQM analytics.
"""

import streamlit as st
from theme import get_current_theme
from position_graph import build_linear_qccd, Placement
from congestion_handler import CongestionHandler, walk_trajectories
from qubo_formulator import QUBOFormulator


def render_qubo_tab() -> None:
    theme = get_current_theme()
    is_dark = st.session_state.get("qubroute_theme", "dark") == "dark"

    st.markdown("### QUBO Formulation & BQM Encoding")
    st.caption("How localized trapped-ion shuttling windows are transformed into quadratic binary optimization models")

    # -----------------------------------------------------------------------
    # Full Hamiltonian Mathematical Display
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### Complete Shuttling Hamiltonian ($H$)")
        st.latex(
            r"H = H_{\text{cost}} + H_{\text{lookahead}} + H_{\text{settle}} + "
            r"P \cdot \left( H_{\text{one-hot}} + H_{\text{capacity}} + H_{\text{movement}} + H_{\text{crossing}} \right)"
        )
        st.markdown(
            f"""
            <p style="font-size: 0.92rem; color: {theme['muted']}; margin-top: 8px;">
            Binary decision variables <code>x(i, v, t) ∈ {{0, 1}}</code> indicate whether ion <code>i</code> occupies physical position <code>v ∈ V</code> at discrete time step <code>t ∈ [0, T]</code>.<br>
            The penalty multiplier <code>P</code> strictly exceeds the maximum possible routing objective cost, ensuring that any physical rule violation is penalized higher than any valid shuttling plan.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Table of Hamiltonian Terms (from README)
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown("#### Hamiltonian Terms Breakdown")

    terms_data = [
        {
            "Term": "H_cost",
            "Mathematical Expression": "∑ c(u,v) · x(i,u,t) · x(i,v,t+1)",
            "Physical Purpose": "Price per physical hop of any ion (unit count = 1.0, or duration under TIME_COSTS_US)",
            "Weight / Penalty": "1.0 (or hardware µs)",
        },
        {
            "Term": "H_lookahead",
            "Mathematical Expression": "w · dist(v, next_partner(i)) · x(i,v,T)",
            "Physical Purpose": "Leave ions in favorable positions for upcoming two-qubit entangling gates",
            "Weight / Penalty": "w = 0.5",
        },
        {
            "Term": "H_settle",
            "Mathematical Expression": "ε · x(i,v,T) for v ≠ start_i",
            "Physical Purpose": "Tie-break penalty: prevent disturbing bystander ions that have no immediate gate",
            "Weight / Penalty": "ε · |A| < 1 (ε = 0.05)",
        },
        {
            "Term": "H_one-hot",
            "Mathematical Expression": "∑_{i,t} (∑_v x(i,v,t) - 1)²",
            "Physical Purpose": "Strict physical constraint: each ion must occupy exactly one position at every time step",
            "Weight / Penalty": "P (Hard constraint)",
        },
        {
            "Term": "H_capacity",
            "Mathematical Expression": "∑ x(i,v,t) · x(j,v,t) for i ≠ j",
            "Physical Purpose": "Strict hardware constraint: at most one ion per discrete trap slot or transport segment",
            "Weight / Penalty": "P (Hard constraint)",
        },
        {
            "Term": "H_movement",
            "Mathematical Expression": "∑ x(i,u,t) · x(i,w,t+1) for (u,w) ∉ E ∧ u ≠ w",
            "Physical Purpose": "Adjacency constraint: ions cannot teleport across non-connected positions",
            "Weight / Penalty": "P (Hard constraint)",
        },
        {
            "Term": "H_crossing",
            "Mathematical Expression": "x(a,u,t) · x(a,v,t+1) · x(b,v,t) · x(b,u,t+1)",
            "Physical Purpose": "Strict collision avoidance: forbid anti-parallel crossing exchanges over transport segments",
            "Weight / Penalty": "2P (Rosenberg reduced)",
        },
    ]
    st.dataframe(terms_data, width="stretch")

    # -----------------------------------------------------------------------
    # Variable Pruning & Rosenberg Reduction Details
    # -----------------------------------------------------------------------
    st.write("")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        with st.container(border=True):
            st.markdown("#### Reachability-Based Variable Pruning")
            st.markdown(
                """
                A naive global formulation would generate $|I| \\cdot |V| \\cdot (T+1)$ binary variables.
                QubRoute prunes unfeasible position-time pairs using forward and backward reachability:

                $$\\text{dist}(\\text{start}_i, v) \\le t \\quad \\land \\quad \\text{dist}(v, \\text{FinalSet}_i) \\le T - t$$

                - **Forward Reachability**: An ion cannot reach position $v$ if the shortest path from its starting slot exceeds $t$ hops.
                - **Backward Reachability**: Position $v$ is pruned if an ion at $v$ cannot reach its target set within the remaining $T - t$ steps.
                - **Final Sets**: The mover must reach the goal trap slots; bystanders must end in legal trap slots.
                - **Pruning Efficiency**: Eliminates **85% to 95%** of candidate variables, keeping model sizes to a median of **~50 variables**!
                """
            )

    with col_p2:
        with st.container(border=True):
            st.markdown("#### Rosenberg Reduction for Crossing Terms")
            st.markdown(
                """
                The collision constraint $H_{\\text{crossing}}$ is quartic (degree 4) because it involves two ions at two steps:
                $$x_{a,u,t} \\cdot x_{a,v,t+1} \\cdot x_{b,v,t} \\cdot x_{b,u,t+1}$$

                Standard QUBO requires quadratic interactions ($x_i x_j$).
                QubRoute reduces quartic terms via **Rosenberg substitution**:
                1. Introduce auxiliary binary variable $y = x_{a,u,t} \\cdot x_{b,v,t}$.
                2. Apply Rosenberg penalty with weight $2P$:
                $$P_{\\text{Rosenberg}}(x_1, x_2, y) = 2P \\cdot (x_1 x_2 - 2 x_1 y - 2 x_2 y + 3y)$$
                3. Substitute $y$ into the remaining product: $y \\cdot x_{a,v,t+1} \\cdot x_{b,u,t+1}$.
                Because the penalty is $2P$, an auxiliary variable can never "cheat" to lower the objective.
                """
            )

    # -----------------------------------------------------------------------
    # Live BQM Statistics Example
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown("#### Live BQM Size & Sparsity Inspection")
    st.caption("Inspect live BQM metrics constructed on a standard 3-trap congestion scenario")

    # Generate live model
    pg_demo = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl_demo = Placement(pg_demo)
    for q in range(3):
        pl_demo.place(q, f"t{q}:0")
    path_demo = pg_demo.shortest_path("t0:0", "t2:1")
    handler_demo = CongestionHandler()
    blocked_demo = handler_demo._detect_blockages(path_demo, pl_demo)
    win_demo = handler_demo.extract_window(path_demo, blocked_demo, pl_demo, pg_demo)
    traj_demo = walk_trajectories(pg_demo, pl_demo, path_demo, win_demo)
    prob_demo = QUBOFormulator().build(win_demo, pg_demo, min_horizon=len(traj_demo[win_demo.mover]) - 1)

    unpruned_count = len(win_demo.active_ions) * len(win_demo.window_nodes) * (prob_demo.time_horizon + 1)
    pruned_count = prob_demo.num_variables
    reduction_pct = (unpruned_count - pruned_count) / unpruned_count * 100

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Unpruned Variables", str(unpruned_count))
    col_m2.metric("Pruned Variables x(i,v,t)", str(pruned_count), delta=f"-{reduction_pct:.1f}% pruned", delta_color="normal")
    col_m3.metric("Rosenberg Aux Variables", str(prob_demo.num_aux_variables))
    col_m4.metric("Quadratic Couplings", f"{len(prob_demo.bqm.quadratic):,}")
