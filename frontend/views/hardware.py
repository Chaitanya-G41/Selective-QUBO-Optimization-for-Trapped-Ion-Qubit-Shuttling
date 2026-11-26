"""
frontend/views/hardware.py
Hardware Tab: 2x2 grid of all 4 machine types with compact glowing exact_sim maps,
plus collapsed trap/slot and initial placement tables.
"""

from typing import Any, Dict, List, Tuple
import streamlit as st
import streamlit.components.v1 as components

from benchmark_runner import make_machine
from exact_sim import EXACT_SIM_COMPACT_HEIGHT, render_exact_sim
from position_graph import Placement, build_junction_grid_qccd, build_linear_qccd
from trap_sim import positions_for_exact_sim

MACHINE_SPECS: List[Tuple[str, str]] = [
    ("linear", "Traps in a 1D chain joined by single transport segments."),
    ("junction_grid", "X-junctions on a square grid with a linear trap on every edge."),
    ("grid", "qccd/ grid adapter where any slot of a trap reaches any neighbour."),
    ("grid_dense", "Two home ions per capacity-3 trap, causing multi-step collisions."),
]


def _build_machine(arch: str, num_traps: int, trap_capacity: int):
    num_qudits = max(num_traps, 4)
    if arch == "linear":
        pg = build_linear_qccd(num_traps=num_traps, trap_capacity=trap_capacity)
        pl = Placement(pg)
        for q in range(min(num_qudits, num_traps)):
            pl.place(q, pg.slots_of(f"t{q}")[0])
        return pg, pl
    machine = make_machine(arch, num_qudits)
    if machine is not None:
        return machine
    pg = build_linear_qccd(num_traps=num_traps, trap_capacity=trap_capacity)
    pl = Placement(pg)
    for q in range(min(num_qudits, num_traps)):
        pl.place(q, pg.slots_of(f"t{q}")[0])
    return pg, pl


def _demo_shuttle(pg: Any, start: Dict[Any, Any]) -> Tuple[List[Dict[str, str]], List[str]]:
    if not start:
        return [], []
    ion0 = 0 if 0 in start else next(iter(start))
    src = start[ion0]
    occupied = set(start.values())
    free = [n for n in pg.graph.neighbors(src) if n not in occupied]
    if not free:
        return [], []
    dest = free[0]
    return [{"t": "move", "i": str(ion0), "to": str(dest)}], [str(src), str(dest)]


def render_hardware_tab(architecture: str, num_traps: int, trap_capacity: int) -> None:
    theme_name = st.session_state.get("qubroute_theme", "dark")

    machines: Dict[str, Tuple[Any, Any]] = {}
    for arch, _ in MACHINE_SPECS:
        machines[arch] = _build_machine(arch, num_traps, trap_capacity)

    # 2x2 grid of compact cards
    for row_idx in range(2):
        cols = st.columns(2)
        for col_idx in range(2):
            arch, desc = MACHINE_SPECS[row_idx * 2 + col_idx]
            pg, pl = machines[arch]
            num_t = len(getattr(pg, "_trap_slots", {}))
            num_v = pg.graph.number_of_nodes()
            num_e = pg.graph.number_of_edges()

            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"**{arch}** — {desc}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Traps", str(num_t))
                    m2.metric("Positions |V|", str(num_v))
                    m3.metric("Transitions |E|", str(num_e))

                    start = {ion: pl.position_of(ion) for ion in pl._phi}
                    pos = positions_for_exact_sim(pg, arch)
                    demo_ops, demo_path = _demo_shuttle(pg, start)
                    html = render_exact_sim(
                        pg,
                        start,
                        [{"name": arch, "ops": demo_ops}],
                        mover=0 if 0 in start else None,
                        path=demo_path,
                        positions=pos,
                        theme=theme_name,
                        autoplay=len(demo_ops) > 0,
                        compact=True,
                    )
                    components.html(html, height=EXACT_SIM_COMPACT_HEIGHT, scrolling=False)

    # Collapsed original tables with small machine selector
    with st.expander("Trap & slot configuration and initial qubit placements", expanded=False):
        sel_arch = st.selectbox(
            "Machine",
            [a for a, _ in MACHINE_SPECS],
            index=[a for a, _ in MACHINE_SPECS].index(architecture) if architecture in dict(MACHINE_SPECS) else 0,
            key="hw_table_arch",
        )
        pg, pl = machines[sel_arch]
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Trap & slot configuration**")
            trap_nodes = [n for n in pg.graph.nodes if pg.graph.nodes[n].get("trap_id") is not None]
            trap_ids = list(dict.fromkeys(pg.graph.nodes[n]["trap_id"] for n in trap_nodes))
            trap_data = []
            for tid in trap_ids[:20]:
                slots = [n for n in trap_nodes if pg.graph.nodes[n]["trap_id"] == tid]
                trap_data.append({
                    "Trap ID": str(tid),
                    "Slots": ", ".join(str(s) for s in slots),
                    "Capacity": len(slots),
                    "Occupied": sum(1 for s in slots if pl.is_occupied(s)),
                })
            st.dataframe(trap_data, width="stretch")

        with col_t2:
            st.markdown("**Initial qubit placements**")
            ion_data = []
            if hasattr(pl, "_phi"):
                for q, pos_node in list(pl._phi.items())[:20]:
                    region = "Trap Slot" if pg.graph.nodes[pos_node].get("trap_id") is not None else "Transport"
                    ion_data.append({
                        "Qubit": f"Q{q}",
                        "Physical Node": str(pos_node),
                        "Zone Type": region,
                    })
            st.dataframe(ion_data, width="stretch")
