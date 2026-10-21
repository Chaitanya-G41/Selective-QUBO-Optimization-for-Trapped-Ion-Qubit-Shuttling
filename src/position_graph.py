"""
position_graph.py

Implements the Position Graph abstraction from:
    Bach, Safro, Younis - "Efficient Compilation for Shuttling Trapped-Ion
    Machines via the Position Graph Architectural Abstraction" (arXiv:2501.12470)

Definition 1 (paraphrased): a labeled graph G_p = (V, E, psi) where
  - each vertex is a physical *position* an ion can occupy
    (either a slot inside a trap, or an intermediate transport position)
  - each edge is a legal transition between two positions
  - psi labels each edge as one of: "swap", "merge_split", "move"
        swap        -> transition between two slots *inside the same trap*
        merge_split  -> transition between a trap-boundary slot and a
                        transport ("segment") position
        move        -> transition *between* transport positions
                        (e.g. across a junction, or along a segment)

This module gives you:
  - PositionGraph: build traps + segments/junctions, wire them together
    with correctly-labeled edges
  - Placement: the Phi function -- tracks which logical qudit currently
    sits at which physical position (this changes during routing; the
    PositionGraph itself is static)
  - build_linear_qccd(): a small factory for a simple "chain of traps"
    architecture, good enough to sanity-check routing logic on toy circuits

NOTE / SCOPE: this is a deliberately simplified first version. The full
Definition 1 vertex/edge-count formulas handle junctions of arbitrary
degree (multiple traps meeting at one junction). This version supports
linear chains and star-shaped junctions, which is enough to get SHAW's
core routing loop working end to end. Extending add_junction() to
arbitrary degree is a clean, isolated follow-up -- it does not require
touching Placement, congestion detection, or anything built on top of
this module.
"""

from __future__ import annotations

import itertools
import networkx as nx

VALID_LABELS = {"swap", "merge_split", "move"}


class PositionGraph:
    """Static hardware structure: which positions exist, and which
    transitions between them are legal. Does NOT track where ions
    currently are -- see Placement for that."""

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self._trap_slots: dict[str, list[str]] = {}   # trap_id -> [node ids]
        self._segments: list[str] = []                # segment/junction node ids
        self._node_counter = itertools.count()

    # ---- construction -----------------------------------------------

    def add_trap(self, trap_id: str, capacity: int) -> list[str]:
        """Add a trap with `capacity` internal ion slots, fully wired
        with 'swap' edges (any slot can exchange with any other slot
        in the same trap -- matches the paper's within-trap semantics)."""
        if trap_id in self._trap_slots:
            raise ValueError(f"trap '{trap_id}' already exists")

        slots = [f"{trap_id}:{i}" for i in range(capacity)]
        for slot in slots:
            self.graph.add_node(slot, kind="trap_slot", trap_id=trap_id)

        for u, v in itertools.combinations(slots, 2):
            self.graph.add_edge(u, v, label="swap")

        self._trap_slots[trap_id] = slots
        return slots

    def add_segment(self, segment_id: str) -> str:
        """Add a transport/junction position (an intermediate node an
        ion passes through while moving between traps)."""
        if self.graph.has_node(segment_id):
            raise ValueError(f"segment '{segment_id}' already exists")
        self.graph.add_node(segment_id, kind="segment")
        self._segments.append(segment_id)
        return segment_id

    def connect(self, u: str, v: str, label: str) -> None:
        """Add a labeled transition edge between two existing positions."""
        if label not in VALID_LABELS:
            raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")
        if not self.graph.has_node(u) or not self.graph.has_node(v):
            raise ValueError(f"both '{u}' and '{v}' must already exist")
        self.graph.add_edge(u, v, label=label)

    def connect_trap_to_segment(self, trap_slot: str, segment_id: str) -> None:
        """Convenience wrapper: a trap-boundary slot <-> transport
        position transition is always labeled 'merge_split'."""
        self.connect(trap_slot, segment_id, label="merge_split")

    def connect_segments(self, seg_a: str, seg_b: str) -> None:
        """Convenience wrapper: transport-to-transport is always 'move'."""
        self.connect(seg_a, seg_b, label="move")

    # ---- queries -------------------------------------------------------

    @property
    def num_positions(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()
    def slots_of(self, trap_id: str) -> list[str]:
        return list(self._trap_slots[trap_id])

    def neighbors_by_label(self, position: str, label: str) -> list[str]:
        """All positions directly reachable from `position` via an edge
        with the given label. This is the primitive congestion-resolution
        and target-trap-selection logic will build on top of."""
        return [
            nbr for nbr in self.graph.neighbors(position)
            if self.graph.edges[position, nbr]["label"] == label
        ]

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Cheapest way to get an ion from `source` to `target`, ignoring
        current occupancy. Used later as the congestion-ignoring lower
        bound (C_LB) for the routing-regret metric."""
        return nx.shortest_path(self.graph, source, target)


class Placement:
    """The Phi function: logical qudit -> physical position, plus its
    inverse (which qudit, if any, occupies a given position). This is
    the *dynamic* state that changes every time an ion moves."""

    def __init__(self, position_graph: PositionGraph) -> None:
        self._pg = position_graph
        self._phi: dict[int, str] = {}          # qudit -> position
        self._occupant: dict[str, int] = {}      # position -> qudit

    def place(self, qudit: int, position: str) -> None:
        if position in self._occupant:
            raise ValueError(f"position '{position}' is already occupied")
        self._phi[qudit] = position
        self._occupant[position] = qudit

    def move(self, qudit: int, new_position: str) -> None:
        if new_position in self._occupant:
            raise ValueError(
                f"cannot move qudit {qudit} to '{new_position}': occupied "
                f"by qudit {self._occupant[new_position]}"
            )
        old_position = self._phi[qudit]
        del self._occupant[old_position]
        self._phi[qudit] = new_position
        self._occupant[new_position] = qudit

    def swap(self, qudit_a: int, qudit_b: int) -> None:
        """Exchange the positions of two placed qudits."""
        pos_a, pos_b = self._phi[qudit_a], self._phi[qudit_b]
        self._phi[qudit_a], self._phi[qudit_b] = pos_b, pos_a
        self._occupant[pos_a], self._occupant[pos_b] = qudit_b, qudit_a

    def position_of(self, qudit: int) -> str:
        return self._phi[qudit]

    def occupant_of(self, position: str) -> int | None:
        return self._occupant.get(position)

    def is_occupied(self, position: str) -> bool:
        return position in self._occupant


# ---- example architecture factory --------------------------------------

def build_linear_qccd(num_traps: int, trap_capacity: int) -> PositionGraph:
    """A minimal QCCD-like architecture: traps in a line, each pair of
    neighboring traps connected via one junction/segment position.

        [trap0 slots] --merge_split-- (seg0) --merge_split-- [trap1 slots] -- ...

    Good enough to run a toy circuit through and watch shuttling happen.
    """
    pg = PositionGraph()
    trap_ids = [f"t{i}" for i in range(num_traps)]
    for tid in trap_ids:
        pg.add_trap(tid, trap_capacity)

    for i in range(num_traps - 1):
        seg_id = f"seg{i}"
        pg.add_segment(seg_id)
        # connect the *last* slot of trap i and the *first* slot of trap i+1
        # to this segment -- these are the "boundary" slots
        pg.connect_trap_to_segment(pg.slots_of(trap_ids[i])[-1], seg_id)
        pg.connect_trap_to_segment(pg.slots_of(trap_ids[i + 1])[0], seg_id)

    return pg

def build_junction_grid_qccd(rows: int, cols: int, trap_capacity: int) -> PositionGraph:
    """A 2-D QCCD: X-junctions on a rows x cols grid, with one linear trap
    on every edge between neighbouring junctions. A trap's first slot
    connects to one junction and its last slot to the other, so an ion
    crossing a trap passes through its whole ion chain.

        J(0,0) --[h0_0]-- J(0,1) --[h0_1]-- J(0,2)
          |                 |                 |
        [v0_0]            [v0_1]            [v0_2]
          |                 |                 |
        J(1,0) --[h1_0]-- J(1,1) --[h1_1]-- J(1,2)

    Traps are named "h{r}_{c}" (horizontal) and "v{r}_{c}" (vertical).
    """
    pg = PositionGraph()
    for r in range(rows):
        for c in range(cols):
            pg.add_segment(f"J{r}_{c}")

    def add_trap_between(trap_id: str, end_a: str, end_b: str) -> None:
        slots = pg.add_trap(trap_id, trap_capacity)
        pg.connect_trap_to_segment(slots[0], end_a)
        pg.connect_trap_to_segment(slots[-1], end_b)

    for r in range(rows):
        for c in range(cols - 1):
            add_trap_between(f"h{r}_{c}", f"J{r}_{c}", f"J{r}_{c + 1}")
    for r in range(rows - 1):
        for c in range(cols):
            add_trap_between(f"v{r}_{c}", f"J{r}_{c}", f"J{r + 1}_{c}")
    return pg
