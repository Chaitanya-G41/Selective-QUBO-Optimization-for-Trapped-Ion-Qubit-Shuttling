"""
frontend/views/simulator.py
Simulator Tab: Real 200-circuit selector, congestion-event stepper,
exact_sim animation, and 3 collapsed detail expanders.
"""

from pathlib import Path
from typing import Any, Dict, List
import streamlit as st
import streamlit.components.v1 as components

from circuit_sim_ui import CURATED_CIRCUITS, build_csv_strip, compile_or_get_cached, costs_for_ops
from exact_sim import EXACT_SIM_HEIGHT, ops_from_plan, render_exact_sim
from live_router import get_available_circuits
from trap_sim import positions_for_exact_sim

TESTS_DIR = Path(__file__).resolve().parents[2] / "tests"


def render_simulator_tab(
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
    all_circuits = get_available_circuits()
    if not all_circuits:
        st.error("No benchmark circuits found in tests/ directory.")
        return

    theme_name = st.session_state.get("qubroute_theme", "dark")

    # Controls in one row: Family filter, Circuit selector, Event stepper
    c_fam, c_cir, c_evt = st.columns([1, 2, 2])

    with c_fam:
        family_filter = st.selectbox(
            "Family",
            ["all", "curated", "chain", "random", "layered", "star", "long_range", "other"],
            index=0,
            key="sim_family_filter",
        )

    if family_filter == "all":
        filtered = all_circuits
    elif family_filter == "curated":
        curated_set = set(CURATED_CIRCUITS)
        filtered = [c for c in all_circuits if c[0] in curated_set]
    else:
        filtered = [c for c in all_circuits if c[1] == family_filter]

    if not filtered:
        st.warning("No circuits match this family.")
        return

    labels = [f"{c[0]} ({c[2]}q, {c[3]}g)" for c in filtered]
    files = [c[0] for c in filtered]
    default_idx = files.index("010_chain_6q.qasm") if "010_chain_6q.qasm" in files else 0

    with c_cir:
        selected_label = st.selectbox(
            f"Circuit ({len(all_circuits)} total)",
            labels,
            index=default_idx,
            key="sim_circuit_select",
        )
    selected_file = files[labels.index(selected_label)]

    with st.spinner(f"Routing {selected_file} on {architecture}..."):
        res = compile_or_get_cached(
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

    events: List[Dict[str, Any]] = res.get("event_records", [])
    ev_idx = 0
    with c_evt:
        if events:
            default_ev = next(
                (i for i, ev in enumerate(events) if ev["accepted"] and ev["qubo_count"] < ev["greedy_count"]),
                0,
            )
            ev_labels = [
                f"Event {i + 1}/{len(events)} — Q{ev['mover']} — "
                + (
                    f"QUBO {ev['qubo_count']} vs {ev['greedy_count']}"
                    if ev["accepted"] and ev["qubo_count"] < ev["greedy_count"]
                    else f"Greedy {ev['greedy_count']}"
                )
                for i, ev in enumerate(events)
            ]
            chosen_ev = st.selectbox(
                f"Congestion event ({len(events)})",
                ev_labels,
                index=default_ev,
                key=f"sim_event_{selected_file}_{architecture}",
            )
            ev_idx = ev_labels.index(chosen_ev)
        else:
            st.selectbox("Congestion event", ["0 events (full circuit)"], index=0, disabled=True)

    # Render ONE exact_sim animation
    pg = res["pg"]
    pos = positions_for_exact_sim(pg, architecture)
    strip = build_csv_strip(architecture)

    if events:
        ev = events[ev_idx]
        start = ev["start"]
        g_ops = ops_from_plan(ev["greedy_ops"])
        q_ops = ops_from_plan(ev["qubo_ops"])
        g_costs = costs_for_ops(pg, start, g_ops)
        q_costs = costs_for_ops(pg, start, q_ops)
        scenarios = [
            {"name": "SHAW Greedy Walk", "ops": g_ops, "costs": g_costs},
            {"name": "Selective QUBO Plan", "ops": q_ops, "costs": q_costs},
        ]
        metrics = {
            "kappa": ev["kappa"],
            "rho": ev["rho"],
            "depth": ev["depth"],
            "kappa_th": ev["kappa_th"],
            "rho_th": ev["rho_th"],
            "depth_th": ev["depth_th"],
            "greedy_cost": ev["greedy_count"],
            "lower_bound": ev["lower_bound"],
            "qubo_vars": ev["qubo_vars"],
            "solver": ev["solver"],
            "solve_ms": ev["solve_ms"],
            "triggered": ev["triggered"],
        }
        html = render_exact_sim(
            pg,
            start,
            scenarios,
            mover=ev["mover"],
            path=ev["path"],
            window=ev["window_nodes"],
            blocked=ev["blocked"],
            positions=pos,
            metrics=metrics,
            strip=strip,
            theme=theme_name,
            autoplay=True,
        )
    else:
        ev = None
        start = res["initial_phi"]
        g_ops = ops_from_plan(res["base_trace"])[:24]
        q_ops = ops_from_plan(res["hyb_trace"])[:24]
        scenarios = [
            {"name": "SHAW Greedy Walk", "ops": g_ops, "costs": costs_for_ops(pg, start, g_ops)},
            {"name": "Selective QUBO Plan", "ops": q_ops, "costs": costs_for_ops(pg, start, q_ops)},
        ]
        html = render_exact_sim(
            pg,
            start,
            scenarios,
            mover=None,
            path=[],
            window=[],
            blocked=[],
            positions=pos,
            metrics=None,
            strip=strip,
            theme=theme_name,
            autoplay=True,
        )

    components.html(html, height=EXACT_SIM_HEIGHT, scrolling=True)

    # Three collapsed expanders under the animation
    with st.expander("Event details", expanded=False):
        if ev is not None:
            st.dataframe(
                [
                    {"Metric": "Contention κ (threshold)", "Value": f"{ev['kappa']:.3f} ({ev['kappa_th']:.2f})"},
                    {"Metric": "Regret ρ (threshold)", "Value": f"{ev['rho']:.3f} ({ev['rho_th']:.2f})"},
                    {"Metric": "Blockage depth d (threshold)", "Value": f"{ev['depth']} ({ev['depth_th']})"},
                    {"Metric": "Trigger fired", "Value": str(ev["triggered"])},
                    {"Metric": "Window size |W|", "Value": str(len(ev["window_nodes"]))},
                    {"Metric": "Lower bound", "Value": str(ev["lower_bound"])},
                    {"Metric": "QUBO variables", "Value": str(ev["qubo_vars"])},
                    {"Metric": "Solver", "Value": str(ev["solver"])},
                    {"Metric": "Solve ms", "Value": f"{ev['solve_ms']:.2f}" if ev["solve_ms"] is not None else "—"},
                    {"Metric": "Greedy / QUBO ops", "Value": f"{ev['greedy_count']} / {ev['qubo_count']}"},
                    {"Metric": "Greedy / QUBO µs", "Value": f"{ev['greedy_us']:.0f} / {ev['qubo_us']:.0f}"},
                    {"Metric": "Accepted", "Value": str(ev["accepted"])},
                ],
                width="stretch",
            )
        else:
            st.dataframe([{"Metric": "Blocked events", "Value": "0"}], width="stretch")

    with st.expander("Circuit summary", expanded=False):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Baseline ops", str(res["base_total_ops"]))
        m2.metric("Selective QUBO ops", str(res["hyb_total_ops"]), delta=f"{res['gain_pct']:+.1f}%")
        m3.metric("QUBO triggers", str(res["qubo_triggers"]))
        m4.metric("QUBO accepted", str(res["qubo_accepted"]))
        m5.metric("Selected route", str(res["selected_route"]))

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.dataframe(
                [
                    {"Parameter": "Shuttle Hops", "Baseline": res["base_shuttles"], "Selective QUBO": res["hyb_shuttles"]},
                    {"Parameter": "Ion Swaps", "Baseline": res["base_swaps"], "Selective QUBO": res["hyb_swaps"]},
                    {"Parameter": "Total Ops", "Baseline": res["base_total_ops"], "Selective QUBO": res["hyb_total_ops"]},
                    {"Parameter": "Time (µs)", "Baseline": f"{res['base_time_us']:.1f}", "Selective QUBO": f"{res['hyb_time_us']:.1f}"},
                    {"Parameter": "Circuit Depth", "Baseline": res["base_depth"], "Selective QUBO": res["hyb_depth"]},
                ],
                width="stretch",
            )
        with col_s2:
            routed_ops = res.get("routed_ops", [])
            if routed_ops:
                st.dataframe(
                    [{"Step": i + 1, "Operation": op_str} for i, op_str in enumerate(routed_ops[:15])],
                    width="stretch",
                )

    with st.expander("OpenQASM source", expanded=False):
        qasm_path = TESTS_DIR / selected_file
        if qasm_path.exists():
            st.code(qasm_path.read_text(encoding="utf-8"), language="qasm")
