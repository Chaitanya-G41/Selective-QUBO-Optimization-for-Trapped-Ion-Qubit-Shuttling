"""
frontend/views/benchmarks.py
Benchmarks Tab: driven by sidebar Architecture (st.session_state["architecture"]),
Selected circuit strip from Simulator, Family multiselect filter, 4 big numbers,
themed HTML tables (dark & light), bar chart, CSV download, and Optimality table.
"""

import html as _html
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    ALL_FAMILIES,
    fmt_pct,
    get_cross_machine_summary,
    get_family_breakdown,
    load_benchmark_csv,
)
from theme import get_current_theme


def _render_themed_table(df: pd.DataFrame, theme: Dict[str, str]) -> None:
    """Render a pandas DataFrame as a clean HTML table styled with the active theme."""
    if df.empty:
        return
    headers = "".join(
        f"<th style='text-align:left;padding:8px 12px;background:{theme['table_header_bg']};"
        f"color:{theme['text']};border-bottom:1.5px solid {theme['card_border']};"
        f"font-size:0.82rem;font-weight:700;white-space:nowrap;'>{_html.escape(str(c))}</th>"
        for c in df.columns
    )
    body_rows = []
    for idx, row in enumerate(df.itertuples(index=False)):
        bg = theme["table_row_even"] if idx % 2 == 0 else theme["table_row_odd"]
        cells = "".join(
            f"<td style='padding:7px 12px;color:{theme['text']};"
            f"border-bottom:1px solid {theme['card_border']};font-size:0.85rem;'>"
            f"{_html.escape(str(v))}</td>"
            for v in row
        )
        body_rows.append(f"<tr style='background:{bg};'>{cells}</tr>")

    table_html = (
        f"<div style='overflow-x:auto;border:1.5px solid {theme['card_border']};"
        f"border-radius:8px;background:{theme['panel']};margin-bottom:12px;'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:Plus Jakarta Sans,sans-serif;'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _selected_circuit_name() -> str:
    raw = st.session_state.get("sim_circuit_select", "010_chain_6q.qasm")
    return str(raw).split(" (")[0].strip()


def render_benchmarks_tab() -> None:
    theme = get_current_theme()
    is_dark = st.session_state.get("qubroute_theme", "dark") == "dark"
    selected_arch = st.session_state.get("architecture", "junction_grid")

    df_full = load_benchmark_csv(selected_arch)
    if df_full.empty:
        st.warning(f"Benchmark CSV data not found for `{selected_arch}`.")
        return

    # Selected circuit strip (follows Simulator selection)
    circuit_name = _selected_circuit_name()
    st.markdown(f"#### Selected circuit ({selected_arch})")
    match = df_full[df_full["circuit_name"] == circuit_name] if "circuit_name" in df_full.columns else pd.DataFrame()
    if match.empty:
        st.info(f"Selected circuit `{circuit_name}` is not in the `{selected_arch}` benchmark CSV.")
    else:
        row = match.iloc[0]
        cb = int(row["baseline_total_ops"])
        ch = int(row["hybrid_total_ops"])
        cpct = ((ch - cb) / cb * 100) if cb else 0.0
        trig = int(row.get("hybrid_qubo_triggers", row.get("qubo_triggers", 0)))
        acc = int(row.get("hybrid_qubo_accepted", row.get("qubo_accepted", 0)))
        verdict = "better" if ch < cb else ("worse" if ch > cb else "equal")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Circuit", circuit_name)
        s2.metric("Baseline ops", f"{cb:,}")
        s3.metric("Selective QUBO ops", f"{ch:,}")
        s4.metric("Change", fmt_pct(cpct))
        s5.metric("Triggers / Accepted", f"{trig} / {acc}")
        s6.metric("Result", verdict)

    # Family multiselect filter
    selected_families = st.multiselect(
        "Family",
        list(ALL_FAMILIES),
        default=list(ALL_FAMILIES),
        key="bench_families_multi",
    )
    if not selected_families:
        st.warning("Select at least one family.")
        return

    df_chart = df_full[df_full["family"].isin(selected_families)]

    base_ops = int(df_chart["baseline_total_ops"].sum())
    hyb_ops = int(df_chart["hybrid_total_ops"].sum())
    delta_pct = ((hyb_ops - base_ops) / base_ops * 100) if base_ops else 0.0
    better = int((df_chart["hybrid_total_ops"] < df_chart["baseline_total_ops"]).sum())
    equal = int((df_chart["hybrid_total_ops"] == df_chart["baseline_total_ops"]).sum())
    worse = int((df_chart["hybrid_total_ops"] > df_chart["baseline_total_ops"]).sum())

    # 4 big numbers
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Circuits", str(len(df_chart)))
    m2.metric("Baseline ops", f"{base_ops:,}")
    m3.metric("Selective QUBO ops", f"{hyb_ops:,}", delta=fmt_pct(delta_pct))
    m4.metric("Better / equal / worse", f"{better} / {equal} / {worse}")

    # Per-family table
    st.markdown(f"#### Per-Family Results ({selected_arch})")
    _render_themed_table(get_family_breakdown(selected_arch, tuple(selected_families)), theme)

    # Cross-machine summary table
    st.markdown("#### Cross-Machine Summary")
    _render_themed_table(get_cross_machine_summary(selected_arch), theme)

    # Bar chart
    fig = _plot_ops_by_family(df_chart, selected_families, theme, is_dark)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    # Download CSV
    csv_bytes = df_chart.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download CSV ({len(df_chart)} rows)",
        data=csv_bytes,
        file_name=f"qubroute_{selected_arch}.csv",
        mime="text/csv",
        width="stretch",
    )

    # Optimality section at the bottom (Tabu vs Exact MILP)
    st.markdown("#### Optimality: Tabu Search vs Exact MILP (HiGHS)")
    opt_df = pd.DataFrame([
        {
            "Architecture": "linear",
            "Windows solved to proven optimum": "92 / 95 (96.8%)",
            "Physical ops missed": "0",
            "Tabu latency": "~2–5 ms",
            "Exact MILP latency": "up to 20–30 s",
        },
        {
            "Architecture": "junction_grid",
            "Windows solved to proven optimum": "117 / 119 (98.3%)",
            "Physical ops missed": "1",
            "Tabu latency": "~2–8 ms",
            "Exact MILP latency": "up to 20–30 s",
        },
    ])
    _render_themed_table(opt_df, theme)


def _plot_ops_by_family(df: pd.DataFrame, selected_families: list, theme: dict, is_dark: bool) -> plt.Figure:
    base_sums, hyb_sums, bob_sums, labels = [], [], [], []

    for fam in ALL_FAMILIES:
        if fam not in selected_families:
            continue
        fdf = df[df["family"] == fam]
        if not fdf.empty:
            labels.append(fam)
            base_sums.append(fdf["baseline_total_ops"].sum())
            hyb_sums.append(fdf["hybrid_total_ops"].sum())
            bob_sums.append(fdf["computed_best_of_both_ops"].sum())

    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=140)
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
    ax.legend(facecolor=theme["panel"], edgecolor=theme["card_border"], labelcolor=theme["text"])
    ax.tick_params(colors=theme["muted"])
    ax.grid(axis="y", color=theme["card_border"], linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig
