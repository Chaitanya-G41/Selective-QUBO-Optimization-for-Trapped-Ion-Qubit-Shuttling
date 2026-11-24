"""
frontend/trap_sim.py
High-performance glowing animation engine for Trapped-Ion QCCD Shuttling.
Supports all four architectures: linear, junction_grid, grid, grid_dense.
Provides render_sim, render_sim_comparison, ops_from_plan, and ops_from_greedy.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
import streamlit.components.v1 as components
try:
    from theme import get_current_theme
except ImportError:
    from frontend.theme import get_current_theme
from walk_planner import walk_ops


def ops_from_plan(plan: Any) -> List[Tuple[Any, ...]]:
    """Extract standard physical operations from a QUBO MovePlan."""
    if plan is None or not hasattr(plan, "ops"):
        return []
    return list(plan.ops)


def ops_from_greedy(
    actions: List[Tuple[Any, ...]],
    pg: Any = None,
    placement: Any = None,
    path: Optional[List[Any]] = None,
    mover: Optional[int] = None,
) -> List[Tuple[Any, ...]]:
    """Convert greedy plan_walk actions into standard physical operations."""
    if pg is not None and placement is not None and path is not None and mover is not None:
        try:
            return walk_ops(pg, placement, path, mover)
        except Exception:
            pass

    ops: List[Tuple[Any, ...]] = []
    current_mover = mover if mover is not None else 0
    for action in actions:
        kind = action[0]
        if kind == "move":
            # ("move", dest)
            dest = action[1]
            ops.append(("move", current_mover, dest))
        elif kind == "vacate":
            # ("vacate", occupant, hop, free_slot)
            occupant, hop, free_slot = action[1], action[2], action[3]
            ops.append(("move", occupant, free_slot))
            ops.append(("move", current_mover, hop))
        elif kind == "displace":
            # ("displace", ion, dest)
            ion, dest = action[1], action[2]
            ops.append(("move", ion, dest))
        elif kind == "swap":
            # ("swap", occupant, hop)
            occupant = action[1]
            ops.append(("swap", current_mover, occupant))
    return ops


def compute_machine_layout(pg: Any, architecture: str = "linear") -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    """
    Compute clean, aesthetically aligned 2D layout coordinates for all positions
    in any of the 4 QCCD architectures.
    Returns (pos_dict, canvas_width, canvas_height).
    """
    G = pg.graph
    pos: Dict[str, Tuple[float, float]] = {}

    # Check for junction_grid first
    has_junctions = any(str(n).startswith("J") for n in G.nodes)
    is_adapter_grid = any(isinstance(n, tuple) for n in G.nodes)

    if architecture == "junction_grid" or has_junctions:
        # X-junction grid with linear traps on edges
        junctions = [n for n in G.nodes if str(n).startswith("J")]
        max_r, max_c = 0, 0
        for j in junctions:
            m = re.match(r"J(\d+)_(\d+)", str(j))
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                max_r = max(max_r, r)
                max_c = max(max_c, c)
                pos[str(j)] = (c * 240.0 + 100.0, r * 240.0 + 100.0)

        for n in G.nodes:
            s_name = str(n)
            if s_name in pos:
                continue
            trap = G.nodes[n].get("trap_id")
            if trap:
                mh = re.match(r"h(\d+)_(\d+)", str(trap))
                mv = re.match(r"v(\d+)_(\d+)", str(trap))
                slot_idx = int(s_name.split(":")[1]) if ":" in s_name else 0
                if mh:
                    r, c = int(mh.group(1)), int(mh.group(2))
                    pos[s_name] = (c * 240.0 + 100.0 + 70.0 + slot_idx * 45.0, r * 240.0 + 100.0)
                elif mv:
                    r, c = int(mv.group(1)), int(mv.group(2))
                    pos[s_name] = (c * 240.0 + 100.0, r * 240.0 + 100.0 + 70.0 + slot_idx * 45.0)

        width = max(max_c * 240.0 + 200.0, 500.0)
        height = max(max_r * 240.0 + 200.0, 360.0)

    elif is_adapter_grid or architecture in ("grid", "grid_dense"):
        # Adapter grid from qccd/
        max_r, max_c = 0, 0
        for n in G.nodes:
            if isinstance(n, tuple):
                real_id, slot_idx = n[0], n[1]
                if isinstance(real_id, tuple) and real_id[0] in ("H", "V"):
                    kind, r, c = real_id
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)
                    if kind == "H":
                        pos[str(n)] = (c * 220.0 + 175.0, r * 220.0 + 100.0)
                    else:
                        pos[str(n)] = (c * 220.0 + 100.0, r * 220.0 + 175.0)
                elif isinstance(real_id, tuple):
                    r, c = real_id
                    max_r = max(max_r, r)
                    max_c = max(max_c, c)
                    # For capacity 2 or 3
                    offset = (slot_idx - 0.75) * 36.0 if architecture == "grid_dense" else (slot_idx - 0.5) * 40.0
                    pos[str(n)] = (c * 220.0 + 100.0 + offset, r * 220.0 + 100.0)

        width = max(max_c * 220.0 + 260.0, 500.0)
        height = max(max_r * 220.0 + 260.0, 360.0)

    else:
        # Linear chain
        traps = sorted(
            list(set(G.nodes[n].get("trap_id") for n in G.nodes if G.nodes[n].get("trap_id") is not None)),
            key=lambda x: int(re.search(r"\d+", str(x)).group()) if re.search(r"\d+", str(x)) else 0,
        )
        for i, trap in enumerate(traps):
            slots = [n for n in G.nodes if G.nodes[n].get("trap_id") == trap]
            for s_idx, s in enumerate(slots):
                pos[str(s)] = (i * 180.0 + 90.0 + (s_idx - len(slots) / 2.0 + 0.5) * 44.0, 110.0)
            seg = f"seg{i}"
            if seg in G.nodes:
                pos[str(seg)] = (i * 180.0 + 180.0, 110.0)

        width = max(len(traps) * 180.0 + 60.0, 540.0)
        height = 220.0

    # Fallback for any unmapped node
    for n in G.nodes:
        if str(n) not in pos:
            pos[str(n)] = (100.0, 100.0)

    return pos, width, height


def simulate_step_history(
    initial_placement: Any,
    ops: List[Tuple[Any, ...]],
) -> Tuple[List[Dict[int, str]], List[Dict[str, Any]]]:
    """
    Replay a sequence of operations starting from initial_placement.
    Returns (states_by_step, log_events).
    """
    # Extract initial ion -> position mapping
    curr_pos: Dict[int, str] = {}
    if hasattr(initial_placement, "_phi"):
        for ion, p in initial_placement._phi.items():
            curr_pos[int(ion)] = str(p)
    elif isinstance(initial_placement, dict):
        for ion, p in initial_placement.items():
            curr_pos[int(ion)] = str(p)

    states: List[Dict[int, str]] = [dict(curr_pos)]
    events: List[Dict[str, Any]] = []

    for step_idx, op in enumerate(ops):
        kind = op[0]
        if kind == "move":
            ion, dest = int(op[1]), str(op[2])
            old = curr_pos.get(ion, "unknown")
            curr_pos[ion] = dest
            events.append({
                "step": step_idx + 1,
                "type": "move",
                "ion": ion,
                "from": old,
                "to": dest,
                "desc": f"Ion Q{ion} hopped {old} → {dest}",
            })
        elif kind == "swap":
            ion_a, ion_b = int(op[1]), int(op[2])
            pos_a, pos_b = curr_pos[ion_a], curr_pos[ion_b]
            curr_pos[ion_a], curr_pos[ion_b] = pos_b, pos_a
            events.append({
                "step": step_idx + 1,
                "type": "swap",
                "ion": f"{ion_a} ↔ {ion_b}",
                "from": f"{pos_a}, {pos_b}",
                "to": f"{pos_b}, {pos_a}",
                "desc": f"Swapped Q{ion_a} and Q{ion_b} in-trap",
            })
        states.append(dict(curr_pos))

    return states, events


def render_sim_comparison(
    pg: Any,
    initial_placement: Any,
    greedy_ops: List[Tuple[Any, ...]],
    qubo_ops: List[Tuple[Any, ...]],
    window_nodes: Optional[List[Any]] = None,
    blocked_nodes: Optional[List[Any]] = None,
    architecture: str = "linear",
    title: str = "Physical Trapped-Ion Shuttling Replay",
    height: int = 540,
) -> None:
    """
    Render side-by-side interactive glowing simulation: Greedy Walk vs QUBO Plan.
    """
    theme = get_current_theme()
    pos_dict, canvas_w, canvas_h = compute_machine_layout(pg, architecture)

    # Format nodes & edges
    nodes_data = []
    for n in pg.graph.nodes:
        sn = str(n)
        attrs = pg.graph.nodes[n]
        is_trap = attrs.get("trap_id") is not None or "t" in sn and ":" in sn
        is_window = window_nodes is not None and (n in window_nodes or sn in [str(w) for w in window_nodes])
        is_blocker = blocked_nodes is not None and (n in blocked_nodes or sn in [str(b) for b in blocked_nodes])
        x, y = pos_dict.get(sn, (100.0, 100.0))
        nodes_data.append({
            "id": sn,
            "x": x,
            "y": y,
            "is_trap": bool(is_trap),
            "is_window": bool(is_window),
            "is_blocker": bool(is_blocker),
        })

    edges_data = []
    for u, v in pg.graph.edges:
        label = pg.graph.edges[u, v].get("label", "move")
        su, sv = str(u), str(v)
        if su in pos_dict and sv in pos_dict:
            edges_data.append({
                "u": su,
                "v": sv,
                "x1": pos_dict[su][0],
                "y1": pos_dict[su][1],
                "x2": pos_dict[sv][0],
                "y2": pos_dict[sv][1],
                "label": label,
            })

    greedy_states, greedy_events = simulate_step_history(initial_placement, greedy_ops)
    qubo_states, qubo_events = simulate_step_history(initial_placement, qubo_ops)

    sim_payload = {
        "theme": theme,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "nodes": nodes_data,
        "edges": edges_data,
        "greedy": {
            "states": greedy_states,
            "events": greedy_events,
            "total_ops": len(greedy_ops),
        },
        "qubo": {
            "states": qubo_states,
            "events": qubo_events,
            "total_ops": len(qubo_ops),
        },
    }

    html_content = _build_sim_html(sim_payload, comparison_mode=True)
    components.html(html_content, height=height, scrolling=False)


def render_sim(
    pg: Any,
    initial_placement: Any,
    ops: List[Tuple[Any, ...]],
    window_nodes: Optional[List[Any]] = None,
    blocked_nodes: Optional[List[Any]] = None,
    architecture: str = "linear",
    title: str = "Trapped-Ion Movement Replay",
    height: int = 500,
) -> None:
    """Render a single-panel interactive glowing simulation."""
    theme = get_current_theme()
    pos_dict, canvas_w, canvas_h = compute_machine_layout(pg, architecture)

    nodes_data = []
    for n in pg.graph.nodes:
        sn = str(n)
        attrs = pg.graph.nodes[n]
        is_trap = attrs.get("trap_id") is not None or ("t" in sn and ":" in sn)
        is_window = window_nodes is not None and (n in window_nodes or sn in [str(w) for w in window_nodes])
        is_blocker = blocked_nodes is not None and (n in blocked_nodes or sn in [str(b) for b in blocked_nodes])
        x, y = pos_dict.get(sn, (100.0, 100.0))
        nodes_data.append({
            "id": sn,
            "x": x,
            "y": y,
            "is_trap": bool(is_trap),
            "is_window": bool(is_window),
            "is_blocker": bool(is_blocker),
        })

    edges_data = []
    for u, v in pg.graph.edges:
        label = pg.graph.edges[u, v].get("label", "move")
        su, sv = str(u), str(v)
        if su in pos_dict and sv in pos_dict:
            edges_data.append({
                "u": su,
                "v": sv,
                "x1": pos_dict[su][0],
                "y1": pos_dict[su][1],
                "x2": pos_dict[sv][0],
                "y2": pos_dict[sv][1],
                "label": label,
            })

    states, events = simulate_step_history(initial_placement, ops)

    sim_payload = {
        "theme": theme,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "nodes": nodes_data,
        "edges": edges_data,
        "single": {
            "states": states,
            "events": events,
            "total_ops": len(ops),
        },
    }

    html_content = _build_sim_html(sim_payload, comparison_mode=False)
    components.html(html_content, height=height, scrolling=False)


def _build_sim_html(payload: Dict[str, Any], comparison_mode: bool = True) -> str:
    """Generate self-contained glowing SVG animation HTML."""
    payload_json = json.dumps(payload)
    theme = payload["theme"]

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
        body {{
            background: {theme["bg"]};
            color: {theme["text"]};
            font-family: 'Plus Jakarta Sans', sans-serif;
            padding: 8px;
            overflow: hidden;
        }}

        .sim-container {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            width: 100%;
            height: 100%;
        }}

        .sim-grid {{
            display: grid;
            grid-template-columns: { '1fr 1fr' if comparison_mode else '1fr' };
            gap: 12px;
        }}

        .panel-card {{
            background: {theme["panel"]};
            border: 1.5px solid {theme["card_border"]};
            border-radius: 10px;
            padding: 10px 14px;
            position: relative;
            box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        }}

        .panel-card.qubo-highlight {{
            border-color: {theme["gold"]};
            box-shadow: 0 0 16px rgba(255, 215, 98, 0.2);
        }}

        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.88rem;
            font-weight: 700;
        }}

        .panel-title {{
            color: {theme["text"]};
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .panel-badge {{
            font-size: 0.75rem;
            font-weight: 800;
            padding: 2px 8px;
            border-radius: 4px;
            background: {theme["card_bg"]};
            border: 1px solid {theme["card_border"]};
            color: {theme["gold"]};
        }}

        svg.board-canvas {{
            width: 100%;
            height: 200px;
            display: block;
            background: {theme["bg"]};
            border-radius: 8px;
            border: 1px solid {theme["card_border"]};
        }}

        /* Glow & animation styles */
        .edge-line {{
            stroke: {theme["rails"]};
            stroke-width: 2.2;
            stroke-linecap: round;
            opacity: 0.75;
        }}

        .edge-flow {{
            stroke: {theme["gold"]};
            stroke-width: 2.5;
            stroke-dasharray: 4, 8;
            animation: flowDash 1.2s linear infinite;
        }}

        @keyframes flowDash {{
            from {{ stroke-dashoffset: 12; }}
            to {{ stroke-dashoffset: 0; }}
        }}

        .node-slot {{
            fill: {theme["panel"]};
            stroke: {theme["rails"]};
            stroke-width: 1.8;
            transition: all 0.2s ease;
        }}

        .node-window {{
            stroke: {theme["gold"]} !important;
            stroke-width: 2.2 !important;
            stroke-dasharray: 5, 4;
            animation: windowDash 2s linear infinite;
        }}

        @keyframes windowDash {{
            to {{ stroke-dashoffset: -18; }}
        }}

        .blocker-ring {{
            fill: none;
            stroke: {theme["red"]};
            stroke-width: 2;
            animation: pulseRing 1.4s ease-out infinite;
        }}

        @keyframes pulseRing {{
            0% {{ r: 12; opacity: 1; }}
            100% {{ r: 24; opacity: 0; }}
        }}

        .ion-group {{
            transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
            cursor: pointer;
        }}

        .ion-halo {{
            fill: {theme["gold"]};
            opacity: 0.28;
            filter: blur(4px);
        }}

        .ion-body {{
            fill: {theme["gold"]};
            stroke: #ffffff;
            stroke-width: 1.5;
            filter: drop-shadow(0 0 6px rgba(255, 215, 98, 0.8));
        }}

        .ion-text {{
            fill: #040914;
            font-size: 10px;
            font-weight: 800;
            text-anchor: middle;
            dominant-baseline: central;
        }}

        .arrival-burst {{
            fill: none;
            stroke: {theme["green"]};
            stroke-width: 2.5;
            animation: burst 0.5s ease-out forwards;
        }}

        @keyframes burst {{
            0% {{ r: 6; opacity: 1; stroke-width: 3; }}
            100% {{ r: 26; opacity: 0; stroke-width: 0.5; }}
        }}

        /* Controls bar */
        .controls-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: {theme["panel"]};
            border: 1.5px solid {theme["card_border"]};
            border-radius: 8px;
            padding: 8px 16px;
        }}

        .btn-group {{
            display: flex;
            gap: 6px;
            align-items: center;
        }}

        button.btn {{
            background: {theme["card_bg"]};
            color: {theme["text"]};
            border: 1.2px solid {theme["card_border"]};
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 0.82rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
        }}

        button.btn:hover {{
            border-color: {theme["gold"]};
            color: {theme["gold"]};
        }}

        button.btn-primary {{
            background: {theme["gold"]};
            color: #040914;
            border-color: {theme["gold"]};
        }}

        .step-indicator {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            color: {theme["text"]};
        }}

        .scrub-slider {{
            flex: 1;
            margin: 0 16px;
            accent-color: {theme["gold"]};
            cursor: pointer;
        }}

        .log-display {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: {theme["muted"]};
            height: 28px;
            line-height: 28px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div class="sim-container">
        <div class="sim-grid">
            <!-- Left / Greedy or Single panel -->
            <div class="panel-card" id="left-card">
                <div class="panel-header">
                    <span class="panel-title">{ 'Greedy Walk Baseline' if comparison_mode else 'Circuit Shuttling Execution' }</span>
                    <span class="panel-badge" id="left-ops-badge">0 / 0 ops</span>
                </div>
                <svg id="left-canvas" class="board-canvas"></svg>
                <div class="log-display" id="left-log">Ready to simulate.</div>
            </div>

            {f'''
            <!-- Right / QUBO panel -->
            <div class="panel-card qubo-highlight" id="right-card">
                <div class="panel-header">
                    <span class="panel-title" style="color: {theme["gold"]};">Selective QUBO Plan</span>
                    <span class="panel-badge" id="right-ops-badge">0 / 0 ops</span>
                </div>
                <svg id="right-canvas" class="board-canvas"></svg>
                <div class="log-display" id="right-log">Ready to simulate.</div>
            </div>
            ''' if comparison_mode else ''}
        </div>

        <!-- Controls Toolbar -->
        <div class="controls-bar">
            <div class="btn-group">
                <button class="btn" id="btn-reset">Reset</button>
                <button class="btn" id="btn-prev">Prev</button>
                <button class="btn btn-primary" id="btn-play">Play</button>
                <button class="btn" id="btn-next">Next</button>
            </div>

            <input type="range" id="scrubber" class="scrub-slider" min="0" max="10" value="0">

            <div class="step-indicator" id="step-label">Step 0 / 0</div>

            <div class="btn-group" style="margin-left: 12px;">
                <button class="btn" id="btn-speed">1x</button>
            </div>
        </div>
    </div>

    <script>
        const data = {payload_json};
        const comparisonMode = {str(comparison_mode).lower()};

        let currentStep = 0;
        let isPlaying = false;
        let timer = null;
        let speed = 1.0;
        const speedOptions = [0.5, 1.0, 2.0, 4.0];
        let speedIdx = 1;

        const maxSteps = comparisonMode
            ? Math.max(data.greedy.states.length - 1, data.qubo.states.length - 1)
            : data.single.states.length - 1;

        const scrubber = document.getElementById('scrubber');
        scrubber.max = maxSteps;

        function initCanvas(svgId) {{
            const svg = document.getElementById(svgId);
            if (!svg) return;
            svg.setAttribute('viewBox', `0 0 ${{data.canvas_w}} ${{data.canvas_h}}`);
            svg.innerHTML = `
                <defs>
                    <filter id="halo-glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>
                <g class="edges-layer"></g>
                <g class="nodes-layer"></g>
                <g class="ions-layer"></g>
                <g class="fx-layer"></g>
            `;

            // Draw edges
            const edgesLayer = svg.querySelector('.edges-layer');
            data.edges.forEach(e => {{
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', e.x1);
                line.setAttribute('y1', e.y1);
                line.setAttribute('x2', e.x2);
                line.setAttribute('y2', e.y2);
                line.setAttribute('class', 'edge-line');
                edgesLayer.appendChild(line);

                if (e.label === 'move' || e.label === 'merge_split') {{
                    const flow = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    flow.setAttribute('x1', e.x1);
                    flow.setAttribute('y1', e.y1);
                    flow.setAttribute('x2', e.x2);
                    flow.setAttribute('y2', e.y2);
                    flow.setAttribute('class', 'edge-flow');
                    edgesLayer.appendChild(flow);
                }}
            }});

            // Draw nodes
            const nodesLayer = svg.querySelector('.nodes-layer');
            data.nodes.forEach(n => {{
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('transform', `translate(${{n.x}}, ${{n.y}})`);

                if (n.is_blocker) {{
                    const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    ring.setAttribute('r', '14');
                    ring.setAttribute('class', 'blocker-ring');
                    g.appendChild(ring);
                }}

                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('r', n.is_trap ? '12' : '9');
                circle.setAttribute('class', 'node-slot' + (n.is_window ? ' node-window' : ''));
                g.appendChild(circle);

                const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                label.setAttribute('y', n.is_trap ? '19' : '15');
                label.setAttribute('fill', data.theme.muted);
                label.setAttribute('font-size', '8');
                label.setAttribute('text-anchor', 'middle');
                label.textContent = n.id.length > 8 ? n.id.slice(0, 7) + '..' : n.id;
                g.appendChild(label);

                nodesLayer.appendChild(g);
            }});
        }}

        function updateBoard(svgId, stateDict, eventList, step) {{
            const svg = document.getElementById(svgId);
            if (!svg) return;
            const ionsLayer = svg.querySelector('.ions-layer');
            const fxLayer = svg.querySelector('.fx-layer');
            ionsLayer.innerHTML = '';

            const nodePosMap = {{}};
            data.nodes.forEach(n => {{ nodePosMap[n.id] = {{ x: n.x, y: n.y }}; }});

            Object.entries(stateDict).forEach(([ionId, posName]) => {{
                const target = nodePosMap[posName] || {{ x: 100, y: 100 }};
                const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                g.setAttribute('class', 'ion-group');
                g.setAttribute('transform', `translate(${{target.x}}, ${{target.y}})`);

                // Halo
                const halo = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                halo.setAttribute('r', '15');
                halo.setAttribute('class', 'ion-halo');
                g.appendChild(halo);

                // Body
                const body = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                body.setAttribute('r', '9');
                body.setAttribute('class', 'ion-body');
                g.appendChild(body);

                // Text
                const txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                txt.setAttribute('class', 'ion-text');
                txt.textContent = 'Q' + ionId;
                g.appendChild(txt);

                ionsLayer.appendChild(g);
            }});

            // Check if arrival burst should fire
            if (step > 0 && eventList && eventList[step - 1]) {{
                const ev = eventList[step - 1];
                const destPos = nodePosMap[ev.to];
                if (destPos) {{
                    const burst = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    burst.setAttribute('cx', destPos.x);
                    burst.setAttribute('cy', destPos.y);
                    burst.setAttribute('class', 'arrival-burst');
                    fxLayer.appendChild(burst);
                    setTimeout(() => burst.remove(), 500);
                }}
            }}
        }}

        function renderFrame(step) {{
            currentStep = step;
            scrubber.value = step;
            document.getElementById('step-label').textContent = `Step ${{step}} / ${{maxSteps}}`;

            if (comparisonMode) {{
                const gStep = Math.min(step, data.greedy.states.length - 1);
                const qStep = Math.min(step, data.qubo.states.length - 1);

                updateBoard('left-canvas', data.greedy.states[gStep], data.greedy.events, gStep);
                updateBoard('right-canvas', data.qubo.states[qStep], data.qubo.events, qStep);

                document.getElementById('left-ops-badge').textContent = `${{gStep}} / ${{data.greedy.total_ops}} ops`;
                document.getElementById('right-ops-badge').textContent = `${{qStep}} / ${{data.qubo.total_ops}} ops`;

                document.getElementById('left-log').textContent = gStep > 0
                    ? `[Step ${{gStep}}] ` + (data.greedy.events[gStep - 1]?.desc || '')
                    : 'Initial Placement';
                document.getElementById('right-log').textContent = qStep > 0
                    ? `[Step ${{qStep}}] ` + (data.qubo.events[qStep - 1]?.desc || '')
                    : 'Initial Placement';
            }} else {{
                const sStep = Math.min(step, data.single.states.length - 1);
                updateBoard('left-canvas', data.single.states[sStep], data.single.events, sStep);
                document.getElementById('left-ops-badge').textContent = `${{sStep}} / ${{data.single.total_ops}} ops`;
                document.getElementById('left-log').textContent = sStep > 0
                    ? `[Step ${{sStep}}] ` + (data.single.events[sStep - 1]?.desc || '')
                    : 'Initial Placement';
            }}
        }}

        // Controls
        const playBtn = document.getElementById('btn-play');
        playBtn.onclick = () => {{
            if (isPlaying) {{
                pause();
            }} else {{
                play();
            }}
        }};

        function play() {{
            if (currentStep >= maxSteps) currentStep = 0;
            isPlaying = true;
            playBtn.textContent = 'Pause';
            playBtn.classList.remove('btn-primary');
            timer = setInterval(() => {{
                if (currentStep < maxSteps) {{
                    renderFrame(currentStep + 1);
                }} else {{
                    pause();
                }}
            }}, 700 / speed);
        }}

        function pause() {{
            isPlaying = false;
            playBtn.textContent = 'Play';
            playBtn.classList.add('btn-primary');
            clearInterval(timer);
        }}

        document.getElementById('btn-prev').onclick = () => {{
            pause();
            if (currentStep > 0) renderFrame(currentStep - 1);
        }};

        document.getElementById('btn-next').onclick = () => {{
            pause();
            if (currentStep < maxSteps) renderFrame(currentStep + 1);
        }};

        document.getElementById('btn-reset').onclick = () => {{
            pause();
            renderFrame(0);
        }};

        scrubber.oninput = (e) => {{
            pause();
            renderFrame(parseInt(e.target.value));
        }};

        document.getElementById('btn-speed').onclick = (e) => {{
            speedIdx = (speedIdx + 1) % speedOptions.length;
            speed = speedOptions[speedIdx];
            e.target.textContent = speed + 'x';
            if (isPlaying) {{
                pause();
                play();
            }}
        }};

        // Initialize
        initCanvas('left-canvas');
        if (comparisonMode) initCanvas('right-canvas');
        renderFrame(0);
    </script>
</body>
</html>
    """
