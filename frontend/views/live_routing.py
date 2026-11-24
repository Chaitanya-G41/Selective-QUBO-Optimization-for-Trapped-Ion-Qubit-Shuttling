"""
frontend/views/live_routing.py
Live Routing Tab: 200 circuits selector, OpenQASM viewer, live compile, and side-by-side animated simulation.
"""

from pathlib import Path
import streamlit as st

from live_router import get_available_circuits, run_live_compilation
from theme import get_current_theme
from trap_sim import render_sim_comparison

TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"


def render_live_routing_tab(
    architecture: str,
    kappa_th: float,
    rho_th: float,
    depth_th: int,
    routing_mode: str,
    solver_engine: str,
    tabu_effort: int,
    solver_seed: int,
    objective: str,
    baseline_guard: bool,
    fast_trigger: bool,
) -> None:
    theme = get_current_theme()
    is_dark = st.session_state.get("qubroute_theme", "dark") == "dark"

    # Get all 200 benchmark circuits
    all_circuits = get_available_circuits()
    if not all_circuits:
        st.error("No benchmark circuits found in tests/ directory.")
        return

    # Filter circuits by family or search
    c_col1, c_col2 = st.columns([1, 3])
    with c_col1:
        family_filter = st.selectbox(
            "Filter Family",
            ["all", "chain", "random", "layered", "star", "long_range", "other"],
            index=0,
        )

    filtered_circuits = (
        all_circuits if family_filter == "all"
        else [c for c in all_circuits if c[1] == family_filter]
    )

    circuit_labels = [f"{c[0]} ({c[2]} qubits, {c[3]} gates)" for c in filtered_circuits]
    circuit_files = [c[0] for c in filtered_circuits]

    with c_col2:
        selected_label = st.selectbox("Select Benchmark Circuit (200 available)", circuit_labels, index=0)
        selected_file = circuit_files[circuit_labels.index(selected_label)]

    # OpenQASM Source Code Viewer
    with st.expander(f"View OpenQASM 3.0 Source: {selected_file}", expanded=False):
        qasm_path = TESTS_DIR / selected_file
        if qasm_path.exists():
            code_text = qasm_path.read_text(encoding="utf-8")
            st.code(code_text, language="qasm")
        else:
            st.info("Source code file not accessible.")

    # Compilation Trigger
    st.write("")
    compile_btn = st.button("Compile Baseline & Selective QUBO", type="primary", width="stretch")

    if compile_btn or "last_compiled_res" in st.session_state and st.session_state["last_compiled_res"]["circuit_name"] == selected_file:
        if compile_btn:
            with st.spinner(f"Compiling {selected_file} with baseline and selective QUBO on {architecture}..."):
                res = run_live_compilation(
                    circuit_name=selected_file,
                    architecture=architecture,
                    kappa_th=kappa_th,
                    rho_th=rho_th,
                    depth_th=depth_th,
                    routing_mode=routing_mode,
                    solver_engine=solver_engine,
                    tabu_effort=tabu_effort,
                    solver_seed=solver_seed,
                    objective=objective,
                    baseline_guard=baseline_guard,
                    fast_trigger=fast_trigger,
                )
                st.session_state["last_compiled_res"] = res
        else:
            res = st.session_state["last_compiled_res"]

        # Results Metrics Row
        st.write("")
        st.markdown("#### Compilation Results & Routing Metrics")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Qubits", str(res["num_qudits"]))
        m2.metric("Original Gates", str(res["num_gates"]))
        m3.metric("Baseline Ops", f"{res['base_total_ops']} ops", help="Shuttle hops + ion swaps")
        m4.metric(
            "Selective-QUBO Ops",
            f"{res['hyb_total_ops']} ops",
            delta=f"{res['gain_pct']:+.1f}%",
            delta_color="normal",
            help="Total operations with selective QUBO optimization",
        )
        m5.metric(
            "Shuttling Time",
            f"{res['hyb_time_us']:.0f} µs",
            delta=f"{(res['hyb_time_us'] - res['base_time_us']) / res['base_time_us'] * 100:+.1f}%" if res['base_time_us'] else "0%",
            delta_color="inverse",
            help="Time under Cyclone & Moveless duration model",
        )
        m6.metric(
            "QUBO Accepted",
            f"{res['qubo_accepted']} / {res['qubo_triggers']}",
            help="Accepted local QUBO plans vs triggers fired",
        )

        st.caption(
            f"Selected complete route: **{res['selected_route']}** &nbsp;|&nbsp; "
            f"Blocked congestion events: **{res['blocked_events']}** &nbsp;|&nbsp; "
            f"Safeguard: {'Active (cheaper of baseline & hybrid retained)' if baseline_guard else 'Raw hybrid exposed'}"
        )

        # Status badge
        if res["qubo_accepted"] > 0:
            st.markdown(
                f'<div class="badge-ok">{res["qubo_accepted"]} QUBO plans accepted: '
                f'total physical operations reduced from {res["base_total_ops"]} → {res["hyb_total_ops"]} '
                f'({res["gain_pct"]:+.1f}%). All physical movement constraints validated.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="badge-neutral">No local QUBO plan beat the greedy walk on this circuit; '
                'the router cleanly preserved the greedy baseline route.</div>',
                unsafe_allow_html=True,
            )

        # Performance Log & Gate Breakdown
        st.write("")
        col_log1, col_log2 = st.columns(2)
        with col_log1:
            with st.container(border=True):
                st.markdown("#### Routing Performance Log")
                log_data = [
                    {"Parameter": "Circuit Name", "Baseline": res["circuit_name"], "Selective QUBO": res["circuit_name"]},
                    {"Parameter": "Shuttle Hops (Move)", "Baseline": str(res["base_shuttles"]), "Selective QUBO": str(res["hyb_shuttles"])},
                    {"Parameter": "Physical Ion Swaps", "Baseline": str(res["base_swaps"]), "Selective QUBO": str(res["hyb_swaps"])},
                    {"Parameter": "Total Operations", "Baseline": str(res["base_total_ops"]), "Selective QUBO": str(res["hyb_total_ops"])},
                    {"Parameter": "Shuttling Time (µs)", "Baseline": f"{res['base_time_us']:.1f}", "Selective QUBO": f"{res['hyb_time_us']:.1f}"},
                    {"Parameter": "Circuit Depth", "Baseline": str(res["base_depth"]), "Selective QUBO": str(res["hyb_depth"])},
                ]
                st.dataframe(log_data, width="stretch")

        with col_log2:
            with st.container(border=True):
                st.markdown("#### Physical Movement Traces Summary")
                st.markdown(
                    f"""
                    - **Baseline Movement Steps**: {len(res['base_trace'])} physical operations
                    - **Selective QUBO Movement Steps**: {len(res['hyb_trace'])} physical operations
                    - **Operations Avoided**: {res['base_total_ops'] - res['hyb_total_ops']} ops
                    - **Unitary Invariance**: Both routes preserve circuit unitary identically (IdentityGate markers only).
                    """
                )
                if res["hyb_trace"]:
                    st.caption("First 5 hybrid operations: " + ", ".join(f"{op[0]}(Q{op[1]})" for op in res["hyb_trace"][:5]))

        # -------------------------------------------------------------------
        # ANIMATED SIMULATION OF ROUTED MOVES
        # -------------------------------------------------------------------
        st.write("")
        st.markdown("### Interactive Animated Movement Replay")
        st.caption("Watch ions shuttle across the QCCD architecture step-by-step: Baseline vs Selective QUBO")

        render_sim_comparison(
            pg=res["pg"],
            initial_placement=res["initial_phi"],
            greedy_ops=res["base_trace"],
            qubo_ops=res["hyb_trace"],
            architecture=architecture,
            height=540,
        )
