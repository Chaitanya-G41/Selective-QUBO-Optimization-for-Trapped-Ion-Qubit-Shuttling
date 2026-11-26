"""
frontend/trap_sim.py
Layout helper for all 4 QCCD architectures: linear (snake, 6 traps/row),
junction_grid, grid, grid_dense.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple


def compute_machine_layout(
    pg: Any, architecture: str = "linear"
) -> Tuple[Dict[str, Tuple[float, float]], float, float]:
    """
    Compute 2D layout coordinates for all positions in any of the 4 QCCD architectures.
    Long linear chains are laid out as a snake, 6 traps per row.
    """
    G = pg.graph
    pos: Dict[str, Tuple[float, float]] = {}

    has_junctions = any(str(n).startswith("J") for n in G.nodes)
    is_adapter_grid = any(isinstance(n, tuple) for n in G.nodes)

    if architecture == "junction_grid" or has_junctions:
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
                    offset = (slot_idx - 0.75) * 36.0 if architecture == "grid_dense" else (slot_idx - 0.5) * 40.0
                    pos[str(n)] = (c * 220.0 + 100.0 + offset, r * 220.0 + 100.0)

        width = max(max_c * 220.0 + 260.0, 500.0)
        height = max(max_r * 220.0 + 260.0, 360.0)

    else:
        # Linear chain: snake layout, 6 traps per row
        traps = sorted(
            list(set(G.nodes[n].get("trap_id") for n in G.nodes if G.nodes[n].get("trap_id") is not None)),
            key=lambda x: int(re.search(r"\d+", str(x)).group()) if re.search(r"\d+", str(x)) else 0,
        )
        per_row = 6
        centres = []
        for i, trap in enumerate(traps):
            row, col = divmod(i, per_row)
            if row % 2 == 1:
                col = per_row - 1 - col
            cx = col * 180.0 + 90.0
            cy = row * 130.0 + 90.0
            centres.append((cx, cy))
            slots = [n for n in G.nodes if G.nodes[n].get("trap_id") == trap]
            for s_idx, s in enumerate(slots):
                pos[str(s)] = (cx + (s_idx - len(slots) / 2.0 + 0.5) * 44.0, cy)

        for i in range(len(traps) - 1):
            seg = f"seg{i}"
            if seg in G.nodes:
                (x1, y1), (x2, y2) = centres[i], centres[i + 1]
                pos[str(seg)] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        num_rows = max(1, (len(traps) + per_row - 1) // per_row)
        cols_used = min(len(traps), per_row)
        width = max(cols_used * 180.0 + 60.0, 540.0)
        height = max(num_rows * 130.0 + 60.0, 220.0)

    for n in G.nodes:
        if str(n) not in pos:
            pos[str(n)] = (100.0, 100.0)

    return pos, width, height


def positions_for_exact_sim(pg: Any, architecture: str = "linear") -> Dict[Any, Tuple[float, float]]:
    """Return {node: (x, y)} keyed by original graph node objects for exact_sim."""
    pos_str, _, _ = compute_machine_layout(pg, architecture)
    return {n: pos_str[str(n)] for n in pg.graph.nodes if str(n) in pos_str}
