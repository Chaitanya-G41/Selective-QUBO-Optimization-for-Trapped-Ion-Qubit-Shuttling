"""
frontend/views/hardware.py
Hardware Tab: Architecture visualizer, position graph abstraction, and machine definitions.
"""

from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import streamlit as st

from position_graph import Placement, PositionGraph, build_junction_grid_qccd, build_linear_qccd
from benchmark_runner import make_machine
from theme import get_current_theme
from trap_sim import compute_machine_layout


def render_hardware_tab(architecture: str, num_traps: int, trap_capacity: int) -> None:
    theme = get_current_theme()
    is_dark = st.session_state.get("qubroute_theme", "dark") == "dark"

    # -----------------------------------------------------------------------
    # Build Hardware Machine
    # -----------------------------------------------------------------------
    num_qudits = max(num_traps, 4)
    machine = make_machine(architecture, num_qudits)
    if machine is None:
        pg = build_linear_qccd(num_traps=num_traps, trap_capacity=trap_capacity)
        pl = Placement(pg)
        for q in range(min(num_qudits, num_traps)):
            pl.place(q, pg.slots_of(f"t{q}")[0])
    else:
        pg, pl = machine

    num_nodes = pg.graph.number_of_nodes()
    num_edges = pg.graph.number_of_edges()
    active_ions = len(pl._phi) if hasattr(pl, "_phi") else num_qudits

    # -----------------------------------------------------------------------
    # Hardware Metrics
    # -----------------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Architecture", architecture)
    m2.metric("Positions |V|", str(num_nodes), help="Total discrete physical locations where an ion can sit")
    m3.metric("Transitions |E|", str(num_edges), help="Legal single-hop physical transitions in the hardware")
    m4.metric("Active Qubits", str(active_ions), help="Number of physical ions currently placed in traps")
    m5.metric("Trap Capacity", f"{trap_capacity} ions / trap", help="Maximum ions permitted in each trap zone")

    st.write("")

    # -----------------------------------------------------------------------
    # Interactive Visual Topology Card
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.markdown(f"#### {architecture.replace('_', ' ').title()} Position Graph ($G_p$)")
        fig = _draw_architecture_matplotlib(pg, pl, architecture, theme, is_dark)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Trap and Slot Configuration Tables
    # -----------------------------------------------------------------------
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        with st.container(border=True):
            st.markdown("#### Trap & Slot Configuration")
            trap_nodes = [n for n in pg.graph.nodes if pg.graph.nodes[n].get("trap_id") is not None]
            trap_ids = list(dict.fromkeys(pg.graph.nodes[n]["trap_id"] for n in trap_nodes))
            trap_data = []
            for tid in trap_ids[:14]:  # Show first 14 traps
                slots = [n for n in trap_nodes if pg.graph.nodes[n]["trap_id"] == tid]
                occupied_count = sum(1 for s in slots if pl.is_occupied(s))
                trap_data.append({
                    "Trap ID": str(tid),
                    "Slots": ", ".join(str(s) for s in slots),
                    "Capacity": trap_capacity,
                    "Occupied": occupied_count,
                })
            st.dataframe(trap_data, width="stretch")
            if len(trap_ids) > 14:
                st.caption(f"Showing 14 of {len(trap_ids)} traps.")

    with col_t2:
        with st.container(border=True):
            st.markdown("#### Initial Qubit Placements (Mapping $\\phi$)")
            ion_data = []
            if hasattr(pl, "_phi"):
                for q, pos in list(pl._phi.items())[:14]:
                    region = "Trap Slot" if pg.graph.nodes[pos].get("trap_id") is not None else "Transport Segment"
                    ion_data.append({
                        "Qubit": f"Q{q}",
                        "Physical Node": str(pos),
                        "Zone Type": region,
                    })
            st.dataframe(ion_data, width="stretch")
            if hasattr(pl, "_phi") and len(pl._phi) > 14:
                st.caption(f"Showing 14 of {len(pl._phi)} qubits.")

    # -----------------------------------------------------------------------
    # Architectural Abstraction Reference (from README)
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown("### Architectural Abstraction: The Position Graph ($G_p$)")
    st.caption("Detailed hardware abstraction definitions from Bach, Safro, Younis (ACM TQC 2026)")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        with st.container(border=True):
            st.markdown("#### Graph Elements & Edge Labels")
            st.markdown(
                """
                The QCCD hardware is abstracted as an undirected graph $G_p = (V, E, \\psi)$:
                - **Trap Zones**: Contain discrete slot nodes $\\{v_1, \\dots, v_k\\}$ connected as a fully connected clique.
                - **Transport Segments**: Transition nodes connecting adjacent traps via merge and split operations.
                - **Capacity Function $\\text{cap}(v)$**: Trap slots hold at most 1 ion; transport channels strictly forbid multi-ion occupancy.
                - **Ion Placement ($\\phi$)**: Tracks the instantaneous dynamic mapping of logical qudits $q_i \\mapsto v \\in V$.

                **Transition Edge Types ($\\psi$):**
                1. `swap`: Transition between two slots inside the same trap (physical ion rotation/re-ordering).
                2. `merge_split`: Transition between a trap-boundary slot and an adjacent transport segment.
                3. `move`: Transition between transport positions (along corridors or crossing X-junctions).
                """
            )

    with col_d2:
        with st.container(border=True):
            st.markdown("#### Machine Models Evaluated in Benchmarks")
            st.markdown(
                """
                All benchmarks evaluate one home ion per trap (capacity 2):
                - **linear**: Traps arranged in a 1D chain, each pair joined by a single transport segment. Mover must pass through intermediate traps or swap with blockers.
                - **junction_grid**: X-junctions on a square grid with a linear trap on every edge. A trap's two end slots connect to its two junctions, so crossing a trap means passing its ion chain. High congestion under multi-directional routing!
                - **grid**: The `qccd/` grid via an adapter, where any slot of a trap reaches any neighbour (no in-trap ordering constraint).
                - **grid_dense**: Packs two home ions into each capacity-3 trap, so ions passing through a trap genuinely collide and require multi-step evacuation.
                """
            )


def _draw_architecture_matplotlib(
    pg: Any,
    pl: Any,
    architecture: str,
    theme: Dict[str, str],
    is_dark: bool,
) -> plt.Figure:
    """Render a crisp, high-contrast matplotlib topology visualizer."""
    pos_dict, canvas_w, canvas_h = compute_machine_layout(pg, architecture)

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=140)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])

    G = pg.graph

    # Convert coordinates to matplotlib scaling
    pos = {n: (pos_dict[str(n)][0], -pos_dict[str(n)][1]) for n in G.nodes if str(n) in pos_dict}

    # Edges
    edge_colors = []
    edge_widths = []
    edge_styles = []
    for u, v in G.edges:
        label = G.edges[u, v].get("label", "move")
        if label == "swap":
            edge_colors.append(theme["gold"] if is_dark else theme["gold_border"])
            edge_widths.append(2.0)
            edge_styles.append("dashed")
        elif label == "merge_split":
            edge_colors.append(theme["rails"])
            edge_widths.append(1.8)
            edge_styles.append("solid")
        else:
            edge_colors.append(theme["rails"])
            edge_widths.append(1.5)
            edge_styles.append("solid")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.85,
    )

    # Node coloring
    node_colors = []
    edge_borders = []
    node_sizes = []

    for n in G.nodes:
        attrs = G.nodes[n]
        is_occupied = pl.is_occupied(n) if hasattr(pl, "is_occupied") else False
        is_trap = attrs.get("trap_id") is not None or ("t" in str(n) and ":" in str(n))

        if is_occupied:
            node_colors.append(theme["gold"] if is_dark else "#d97706")
            edge_borders.append("#ffffff" if is_dark else "#78350f")
            node_sizes.append(560)
        elif is_trap:
            node_colors.append(theme["panel"])
            edge_borders.append(theme["rails"])
            node_sizes.append(420)
        else:
            node_colors.append(theme["bg"])
            edge_borders.append(theme["muted"])
            node_sizes.append(340)

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        edgecolors=edge_borders,
        linewidths=1.8,
        node_size=node_sizes,
    )

    # Labels
    labels = {}
    for n in G.nodes:
        if hasattr(pl, "occupant_of") and pl.occupant_of(n) is not None:
            labels[n] = f"Q{pl.occupant_of(n)}"
        else:
            sn = str(n)
            labels[n] = sn.split(":")[-1] if ":" in sn else sn[-3:]

    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7.5,
        font_family="sans-serif",
        font_weight="bold",
        font_color=theme["bg"] if is_dark else "#ffffff",
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=theme["gold"] if is_dark else "#d97706", edgecolor="#ffffff", label="Placed Qubit Ion (Occupied)"),
        mpatches.Patch(facecolor=theme["panel"], edgecolor=theme["rails"], label="Trap Slot (Empty)"),
        mpatches.Patch(facecolor=theme["bg"], edgecolor=theme["muted"], label="Transport Segment / Junction"),
        mpatches.Patch(facecolor="none", edgecolor=theme["gold"], label="In-Trap Swap Edge (Dashed)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        facecolor=theme["panel"],
        edgecolor=theme["card_border"],
        fontsize=8.5,
        labelcolor=theme["text"],
    )

    ax.axis("off")
    plt.tight_layout()
    return fig
