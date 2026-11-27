"""
frontend/data_loader.py
Loads, standardizes, and computes benchmark statistics from experiments/results/*.csv.
Every CSV column is treated as optional; better/equal/worse is computed from total ops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd
import streamlit as st
from benchmark_runner import circuit_family

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"

CSV_FILES = {
    "junction_grid": "comparison_junction_grid.csv",
    "linear": "comparison.csv",
    "grid": "comparison_grid.csv",
    "grid_dense": "comparison_grid_dense.csv",
}

ALL_FAMILIES = ("chain", "random", "layered", "star", "long_range", "other")


def fmt_pct(pct: float) -> str:
    """Format percentage change: never -0.0% or +0.0%; minus sign only for fewer ops."""
    if abs(pct) < 0.05:
        return "0.0%"
    if pct < 0:
        return f"-{abs(pct):.1f}%"
    return f"+{pct:.1f}%"


@st.cache_data
def load_benchmark_csv(arch: str) -> pd.DataFrame:
    """Load benchmark CSV for an architecture and compute standardized columns."""
    filename = CSV_FILES.get(arch, "comparison_junction_grid.csv")
    csv_path = RESULTS_DIR / filename
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Standardize total ops (total_ops = total_shuttle_ops + swap_count), treating columns as optional
    b_shuttle = df["baseline_total_shuttle_ops"] if "baseline_total_shuttle_ops" in df.columns else 0
    b_swap = df["baseline_swap_count"] if "baseline_swap_count" in df.columns else 0
    h_shuttle = df["hybrid_total_shuttle_ops"] if "hybrid_total_shuttle_ops" in df.columns else 0
    h_swap = df["hybrid_swap_count"] if "hybrid_swap_count" in df.columns else 0
    df["baseline_total_ops"] = b_shuttle + b_swap
    df["hybrid_total_ops"] = h_shuttle + h_swap

    # Recompute verdict from total ops (ignore CSV verdict column)
    def get_verdict(row):
        b = row["baseline_total_ops"]
        h = row["hybrid_total_ops"]
        if h < b:
            return "better"
        elif h > b:
            return "worse"
        return "equal"

    df["computed_verdict"] = df.apply(get_verdict, axis=1)

    # Compute improvement %
    df["total_ops_improvement"] = (
        (df["baseline_total_ops"] - df["hybrid_total_ops"]) / df["baseline_total_ops"].replace(0, pd.NA)
    ).fillna(0.0)

    # Best of both ops
    df["computed_best_of_both_ops"] = df[["baseline_total_ops", "hybrid_total_ops"]].min(axis=1)
    df["best_of_both_improvement"] = (
        (df["baseline_total_ops"] - df["computed_best_of_both_ops"]) / df["baseline_total_ops"].replace(0, pd.NA)
    ).fillna(0.0)

    # Family classification
    df["family"] = df["circuit_name"].apply(circuit_family) if "circuit_name" in df.columns else "other"

    # Ensure optional columns exist
    optional_cols = {
        "circuit_name": "unknown",
        "baseline_num_qudits": 0,
        "baseline_num_gates_original": 0,
        "baseline_compile_time_s": 0.0,
        "hybrid_compile_time_s": 0.0,
        "hybrid_congestion_events": 0,
        "hybrid_qubo_triggers": 0,
        "hybrid_qubo_accepted": 0,
        "hybrid_kappa_mean": 0.0,
        "hybrid_rho_mean": 0.0,
        "hybrid_qubo_acceptance_rate": 0.0,
        "baseline_movement_time_us": 0.0,
        "hybrid_movement_time_us": 0.0,
        "baseline_depth": 0,
        "hybrid_depth": 0,
        "qubo_build_time_s": 0.0,
        "qubo_solve_time_s": 0.0,
        "qubo_decode_time_s": 0.0,
        "qubo_triggers": df["hybrid_qubo_triggers"] if "hybrid_qubo_triggers" in df.columns else 0,
        "qubo_accepted": df["hybrid_qubo_accepted"] if "hybrid_qubo_accepted" in df.columns else 0,
        "qubo_early_optimal": 0,
        "selected_route": "hybrid",
        "planning_mode": "qubo",
    }
    for col, default_val in optional_cols.items():
        if col not in df.columns:
            df[col] = default_val

    return df


@st.cache_data
def get_family_breakdown(arch: str, families: Tuple[str, ...] = ALL_FAMILIES) -> pd.DataFrame:
    """Compute per-family table for `arch` filtered to `families`."""
    df = load_benchmark_csv(arch)
    if df.empty:
        return pd.DataFrame()

    fam_set = set(families) if families else set(ALL_FAMILIES)
    df_filtered = df[df["family"].isin(fam_set)]
    if df_filtered.empty:
        return pd.DataFrame()

    rows = []
    for fam in ALL_FAMILIES:
        if fam not in fam_set:
            continue
        fdf = df_filtered[df_filtered["family"] == fam]
        if fdf.empty:
            continue
        c_count = len(fdf)
        b_ops = int(fdf["baseline_total_ops"].sum())
        h_ops = int(fdf["hybrid_total_ops"].sum())
        bob_ops = int(fdf["computed_best_of_both_ops"].sum())

        change = ((h_ops - b_ops) / b_ops * 100) if b_ops else 0.0
        bob_change = ((bob_ops - b_ops) / b_ops * 100) if b_ops else 0.0

        better = int((fdf["hybrid_total_ops"] < fdf["baseline_total_ops"]).sum())
        equal = int((fdf["hybrid_total_ops"] == fdf["baseline_total_ops"]).sum())
        worse = int((fdf["hybrid_total_ops"] > fdf["baseline_total_ops"]).sum())

        rows.append({
            "Family": fam,
            "Circuits": c_count,
            "Baseline ops": b_ops,
            "Selective-QUBO ops": h_ops,
            "Change": fmt_pct(change),
            "Best of both": fmt_pct(bob_change),
            "Better / equal / worse": f"{better} / {equal} / {worse}",
        })

    tot_circuits = len(df_filtered)
    tot_b_ops = int(df_filtered["baseline_total_ops"].sum())
    tot_h_ops = int(df_filtered["hybrid_total_ops"].sum())
    tot_bob_ops = int(df_filtered["computed_best_of_both_ops"].sum())
    tot_change = ((tot_h_ops - tot_b_ops) / tot_b_ops * 100) if tot_b_ops else 0.0
    tot_bob_change = ((tot_bob_ops - tot_b_ops) / tot_b_ops * 100) if tot_b_ops else 0.0
    tot_better = int((df_filtered["hybrid_total_ops"] < df_filtered["baseline_total_ops"]).sum())
    tot_equal = int((df_filtered["hybrid_total_ops"] == df_filtered["baseline_total_ops"]).sum())
    tot_worse = int((df_filtered["hybrid_total_ops"] > df_filtered["baseline_total_ops"]).sum())

    rows.append({
        "Family": "all",
        "Circuits": tot_circuits,
        "Baseline ops": tot_b_ops,
        "Selective-QUBO ops": tot_h_ops,
        "Change": fmt_pct(tot_change),
        "Best of both": fmt_pct(tot_bob_change),
        "Better / equal / worse": f"{tot_better} / {tot_equal} / {tot_worse}",
    })

    return pd.DataFrame(rows)


@st.cache_data
def get_cross_machine_summary(arch: Optional[str] = None) -> pd.DataFrame:
    """Summary table across all machines (Linear, Junction grid, Grid, Grid Dense)."""
    results = {}
    for machine in ["linear", "junction_grid", "grid", "grid_dense"]:
        df = load_benchmark_csv(machine)
        if df.empty:
            continue
        b_ops = df["baseline_total_ops"].sum()
        h_ops = df["hybrid_total_ops"].sum()
        bob_ops = df["computed_best_of_both_ops"].sum()
        ops_change = ((h_ops - b_ops) / b_ops * 100) if b_ops else 0.0
        bob_change = ((bob_ops - b_ops) / b_ops * 100) if b_ops else 0.0

        b_time = df["baseline_movement_time_us"].sum()
        h_time = df["hybrid_movement_time_us"].sum()
        time_change = ((h_time - b_time) / b_time * 100) if b_time else 0.0

        b_depth = df["baseline_depth"].sum()
        h_depth = df["hybrid_depth"].sum()
        depth_change = ((h_depth - b_depth) / b_depth * 100) if b_depth else 0.0

        b_comp = df["baseline_compile_time_s"].mean()
        h_comp = df["hybrid_compile_time_s"].mean()
        comp_ratio = (h_comp / b_comp) if b_comp else 1.0

        results[machine] = {
            "ops_change": fmt_pct(ops_change),
            "bob_change": fmt_pct(bob_change),
            "time_change": fmt_pct(time_change) if b_time > 0 else "—",
            "depth_change": fmt_pct(depth_change) if b_depth > 0 else "—",
            "comp_ratio": f"~{comp_ratio:.0f}×",
        }

    rows = [
        {
            "Metric": "Operations (selective QUBO)",
            "Linear": results.get("linear", {}).get("ops_change", "—"),
            "Junction grid": results.get("junction_grid", {}).get("ops_change", "—"),
            "Grid (adapter)": results.get("grid", {}).get("ops_change", "—"),
            "Grid dense": results.get("grid_dense", {}).get("ops_change", "—"),
        },
        {
            "Metric": "Operations (best of both)",
            "Linear": results.get("linear", {}).get("bob_change", "—"),
            "Junction grid": results.get("junction_grid", {}).get("bob_change", "—"),
            "Grid (adapter)": results.get("grid", {}).get("bob_change", "—"),
            "Grid dense": results.get("grid_dense", {}).get("bob_change", "—"),
        },
        {
            "Metric": "Shuttling time",
            "Linear": results.get("linear", {}).get("time_change", "—"),
            "Junction grid": results.get("junction_grid", {}).get("time_change", "—"),
            "Grid (adapter)": results.get("grid", {}).get("time_change", "—"),
            "Grid dense": results.get("grid_dense", {}).get("time_change", "—"),
        },
        {
            "Metric": "Circuit depth",
            "Linear": results.get("linear", {}).get("depth_change", "—"),
            "Junction grid": results.get("junction_grid", {}).get("depth_change", "—"),
            "Grid (adapter)": results.get("grid", {}).get("depth_change", "—"),
            "Grid dense": results.get("grid_dense", {}).get("depth_change", "—"),
        },
        {
            "Metric": "Compile time vs baseline",
            "Linear": results.get("linear", {}).get("comp_ratio", "—"),
            "Junction grid": results.get("junction_grid", {}).get("comp_ratio", "—"),
            "Grid (adapter)": results.get("grid", {}).get("comp_ratio", "—"),
            "Grid dense": results.get("grid_dense", {}).get("comp_ratio", "—"),
        },
    ]

    return pd.DataFrame(rows)
