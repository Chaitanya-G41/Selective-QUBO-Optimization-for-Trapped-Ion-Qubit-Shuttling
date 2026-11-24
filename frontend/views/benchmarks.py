"""
frontend/views/benchmarks.py
Benchmarks Tab: Real CSV benchmark results across all 200 circuits,
per-family tables, cross-machine summary, interactive visualization charts, and data export.
"""

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import get_cross_machine_summary, get_family_breakdown, load_benchmark_csv
from theme import get_current_theme


def render_benchmarks_tab() -> None:
    theme = get_current_theme()
    is_dark = st.session_state.get("qubroute_theme", "dark") == "dark"

    st.markdown("### Validated Experimental Benchmarks")
    st.caption("Comprehensive results on 200 OpenQASM circuits evaluated across 4 QCCD architectures")

    # -----------------------------------------------------------------------
    # Machine & Family Filters
    # -----------------------------------------------------------------------
    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        selected_arch = st.selectbox(
            "Select Architecture",
            ["junction_grid", "linear", "grid", "grid_dense"],
            index=0,
            help="junction_grid: X-junctions with linear traps on edges; linear: 1D trap chain",
        )
    with f_col2:
        selected_family = st.selectbox(
            "Select Family Filter",
            ["all", "chain", "random", "layered", "star", "long_range", "other"],
            index=0,
        )

    # -----------------------------------------------------------------------
    # Per-Family Performance Table
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown(f"#### Per-Family Results: {selected_arch.replace('_', ' ').title()}")
    st.caption("Total operations = shuttle hops + ion swaps. Negative change indicates fewer physical operations.")

    family_df = get_family_breakdown(selected_arch)
    st.dataframe(family_df, width="stretch")

    # -----------------------------------------------------------------------
    # Cross-Machine Summary Table (from README)
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown("#### Summary Across Machines & Solver Settings")
    st.caption("Comparison across all 4 QCCD architectures and execution modes")

    cross_df = get_cross_machine_summary()
    st.dataframe(cross_df, width="stretch")

    # -----------------------------------------------------------------------
    # Analysis & Insights (from README)
    # -----------------------------------------------------------------------
    st.write("")
    with st.container(border=True):
        st.markdown("#### What the Experimental Numbers Say")
        st.markdown(
            """
            - **Origin of the Gain**: The improvement originates from resolving congestion inside linear ion chains. The QUBO plans blockers and the mover together and lets the mover stop in any free slot of the target trap.
            - **Grid vs Chain Architectures**: On the adapter `grid`, any free slot of a trap connects to every neighbour, so passing through a trap never collides. The greedy walk is optimal in almost every case and there is nothing to improve.
            - **Global Layout Dynamics**: Every accepted plan is cheaper where it runs locally, but a different layout can occasionally make the greedy router's later choices costlier, so some circuits end up worse overall. **Best of both** removes those losses by keeping the cheaper of the two compiled circuits.
            - **What Was Tried and Dropped**: Evaluated approaches that degraded results included: QUBO choice of which ion moves, lookahead over several upcoming gates, pricing in-trap exchanges as one swap in the QUBO, shorter time horizons, and smaller tabu budgets.
            """
        )

    # -----------------------------------------------------------------------
    # Interactive Benchmark Charts
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown("#### Visual Benchmark Analytics")

    df_full = load_benchmark_csv(selected_arch)
    if df_full.empty:
        st.warning("Benchmark CSV data not found.")
        return

    # Filter data
    df_chart = df_full if selected_family == "all" else df_full[df_full["family"] == selected_family]

    tab_c1, tab_c2, tab_c3, tab_c4 = st.tabs([
        "Operations by Family",
        "Per-Circuit Change (%)",
        "Compile Time Scaling",
        "Acceptance vs Contention",
    ])

    with tab_c1:
        fig1 = _plot_ops_by_family(df_full, theme, is_dark)
        st.pyplot(fig1, width="stretch")
        plt.close(fig1)

    with tab_c2:
        fig2 = _plot_per_circuit_change(df_chart, theme, is_dark)
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

    with tab_c3:
        fig3 = _plot_compile_time(df_chart, theme, is_dark)
        st.pyplot(fig3, width="stretch")
        plt.close(fig3)

    with tab_c4:
        fig4 = _plot_acceptance_vs_contention(df_chart, theme, is_dark)
        st.pyplot(fig4, width="stretch")
        plt.close(fig4)

    # -----------------------------------------------------------------------
    # Filtered Data Table & CSV Export
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown("#### Filtered Circuit Run Data")

    display_cols = [
        "circuit_name", "family", "baseline_num_qudits", "baseline_num_gates_original",
        "baseline_total_ops", "hybrid_total_ops", "total_ops_improvement",
        "computed_best_of_both_ops", "computed_verdict", "baseline_compile_time_s", "hybrid_compile_time_s",
    ]
    present_cols = [c for c in display_cols if c in df_chart.columns]
    st.dataframe(df_chart[present_cols], width="stretch")

    # CSV Download Button
    csv_bytes = df_chart.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download Filtered Results as CSV ({len(df_chart)} rows)",
        data=csv_bytes,
        file_name=f"qubroute_{selected_arch}_{selected_family}.csv",
        mime="text/csv",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Plotting Helpers (High contrast, elegant matplotlib styling)
# ---------------------------------------------------------------------------

def _plot_ops_by_family(df: pd.DataFrame, theme: dict, is_dark: bool) -> plt.Figure:
    families = ["chain", "random", "layered", "star", "long_range", "other"]
    base_sums = []
    hyb_sums = []
    bob_sums = []
    labels = []

    for fam in families:
        fdf = df[df["family"] == fam]
        if not fdf.empty:
            labels.append(fam)
            base_sums.append(fdf["baseline_total_ops"].sum())
            hyb_sums.append(fdf["hybrid_total_ops"].sum())
            bob_sums.append(fdf["computed_best_of_both_ops"].sum())

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["panel"])

    x = np.arange(len(labels))
    w = 0.25

    ax.bar(x - w, base_sums, width=w, label="Baseline", color=theme["rails"], edgecolor=theme["card_border"], linewidth=1)
    ax.bar(x, hyb_sums, width=w, label="Selective QUBO", color=theme["gold"], edgecolor="#ffffff" if is_dark else "#000", linewidth=1)
    ax.bar(x + w, bob_sums, width=w, label="Best of Both", color=theme["green"], edgecolor=theme["card_border"], linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=theme["text"], fontweight="bold", fontsize=9)
    ax.set_ylabel("Total Physical Operations", color=theme["text"], fontsize=9.5, fontweight="bold")
    ax.set_title("Total Operations by Circuit Family", color=theme["text"], fontsize=11, fontweight="bold", pad=12)
    ax.legend(facecolor=theme["panel"], edgecolor=theme["card_border"], labelcolor=theme["text"])
    ax.tick_params(colors=theme["muted"])
    ax.grid(axis="y", color=theme["card_border"], linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def _plot_per_circuit_change(df: pd.DataFrame, theme: dict, is_dark: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["panel"])

    sorted_df = df.sort_values(by="total_ops_improvement", ascending=False).reset_index(drop=True)
    changes = sorted_df["total_ops_improvement"] * 100

    colors = [theme["green"] if c > 0 else (theme["red"] if c < 0 else theme["muted"]) for c in changes]

    ax.bar(range(len(changes)), changes, color=colors, width=0.8, edgecolor="none")
    ax.axhline(0, color=theme["text"], linewidth=1, linestyle="-", alpha=0.6)

    ax.set_xlabel("Circuits (Sorted by Operations Reduced)", color=theme["text"], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Operation Improvement (%)", color=theme["text"], fontsize=9.5, fontweight="bold")
    ax.set_title("Per-Circuit Operations Improvement (%) — Positive is Better", color=theme["text"], fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(colors=theme["muted"])
    ax.grid(axis="y", color=theme["card_border"], linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def _plot_compile_time(df: pd.DataFrame, theme: dict, is_dark: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["panel"])

    qubits = df["baseline_num_qudits"]
    b_time = df["baseline_compile_time_s"]
    h_time = df["hybrid_compile_time_s"]

    ax.scatter(qubits, b_time, color=theme["rails"], label="Baseline Compile Time", alpha=0.8, s=40, edgecolors="none")
    ax.scatter(qubits, h_time, color=theme["gold"], label="Selective QUBO Compile Time", alpha=0.8, s=45, edgecolors="#ffffff" if is_dark else "#000")

    ax.set_yscale("log")
    ax.set_xlabel("Number of Qubits", color=theme["text"], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Compile Time (seconds, log scale)", color=theme["text"], fontsize=9.5, fontweight="bold")
    ax.set_title("Compilation Latency vs Qubit Count", color=theme["text"], fontsize=11, fontweight="bold", pad=12)
    ax.legend(facecolor=theme["panel"], edgecolor=theme["card_border"], labelcolor=theme["text"])
    ax.tick_params(colors=theme["muted"])
    ax.grid(color=theme["card_border"], linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def _plot_acceptance_vs_contention(df: pd.DataFrame, theme: dict, is_dark: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["panel"])

    kappa = df["hybrid_kappa_mean"]
    accept_rate = df["hybrid_qubo_acceptance_rate"] * 100

    sc = ax.scatter(
        kappa, accept_rate,
        c=df["hybrid_rho_mean"],
        cmap="viridis",
        s=50,
        alpha=0.85,
        edgecolors=theme["card_border"],
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Routing Regret (ρ)", color=theme["text"], fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=theme["muted"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=theme["muted"])

    ax.set_xlabel("Mean Local Contention (κ)", color=theme["text"], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("QUBO Acceptance Rate (%)", color=theme["text"], fontsize=9.5, fontweight="bold")
    ax.set_title("QUBO Plan Acceptance vs Contention Metrics", color=theme["text"], fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(colors=theme["muted"])
    ax.grid(color=theme["card_border"], linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig
