"""
qccd_adapter.py

Bridges two Position Graph models that describe the same physical idea
(traps, transport, capacity) in two different shapes:

  qccd/position_graph.py (newer, richer module)
      A trap is ONE node that can directly hold up to `capacity` ions.
      "Ion A and ion B are co-located" = they're both in the same
      node's occupant set.

  src/position_graph.py (the model ShawRoutingPass currently expects)
      A trap with capacity C is C separate "slot" nodes, one ion each.
      "Ion A and ion B are co-located" = they're at two slots that
      share the same trap_id.

ShawRoutingPass (shaw_routing_pass.py) is written against the second
shape and we don't want to touch it. So instead of reworking the
routing pass, this module wraps a qccd PositionGraph and exposes it
through the *slot-based* interface ShawRoutingPass already calls:

    pg.graph.nodes[pos]["trap_id"]   (via ShawRoutingPass._trap_of)
    pg.slots_of(trap_id)
    pg.shortest_path(a, b)
    Placement.place / .move / .swap / .position_of / .occupant_of / .is_occupied

How: every real qccd node with capacity C becomes C "virtual slots"
here, named (real_node_id, slot_index). A qccd TRAP node's virtual
slots report their real node_id as trap_id; a CHANNEL/JUNCTION node's
virtual slots report trap_id=None (matching "transport nodes report
None" -- ShawRoutingPass only treats trap_id-bearing positions as
"the same trap").

Known simplifications (documented, not hidden):
  - When choosing which slot represents a real node the path merely
    passes through, we prefer a free one but fall back to slot 0 if
    none exists -- see QccdPositionGraphAdapter.shortest_path.
  - A cross-node edge in qccd connects two real nodes; here we connect
    every virtual slot of one to every virtual slot of the other so an
    ion in any slot of a trap can path out through that edge. Real
    hardware would need an internal shuffle to the boundary slot first;
    we simplify that to a single hop, consistent with this being a
    "thin adapter" rather than a full re-model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from qccd.position_graph import NodeType, PositionGraph as QccdPositionGraph

# A virtual slot is (real_node_id, slot_index_within_that_node).
Slot = Tuple[Any, int]


class QccdPositionGraphAdapter:
    """Presents a qccd PositionGraph as a slot-based PositionGraph."""

    def __init__(self, qccd_graph: QccdPositionGraph) -> None:
        self.qccd_graph = qccd_graph
        # Set after construction by build_from_qccd(), so shortest_path
        # can check real occupancy when choosing an intermediate slot
        # for a trap the path merely passes through. Optional by design
        # -- shortest_path still works (falling back to slot 0) if this
        # is never attached, it just can't avoid an avoidable collision.
        self._placement: Optional["QccdPlacementAdapter"] = None

        # real_node_id -> [virtual slots for that node], in slot order.
        self._slots_by_real: Dict[Any, List[Slot]] = {}
        # virtual slot -> which real node it belongs to (inverse lookup).
        self._real_of_slot: Dict[Slot, Any] = {}

        # The slot-level graph ShawRoutingPass._trap_of reads directly
        # (pg.graph.nodes[pos]["trap_id"]).
        self.graph: nx.Graph = nx.Graph()

        self._build()

    # ------------------------------------------------------------
    # Construction: expand real nodes into virtual slots + edges
    # ------------------------------------------------------------
    def _build(self) -> None:
        for node in self.qccd_graph.nodes():
            trap_id = node.node_id if node.node_type is NodeType.TRAP else None
            slots = [(node.node_id, i) for i in range(node.capacity)]
            self._slots_by_real[node.node_id] = slots
            for slot in slots:
                self._real_of_slot[slot] = node.node_id
                self.graph.add_node(slot, trap_id=trap_id)

            # Within a single trap, every slot can trade with every
            # other slot (an in-trap ion swap) -> label "swap".
            for i, slot_a in enumerate(slots):
                for slot_b in slots[i + 1:]:
                    self.graph.add_edge(slot_a, slot_b, label="swap")

        # Cross-node edges: connect every slot of u to every slot of v
        # for each real qccd edge (u, v). Label reflects what's on
        # each end, mirroring src/position_graph.py's convention.
        seen_pairs = set()
        for u, v, _attrs in self.qccd_graph.edges():
            pair = frozenset((u, v))
            if pair in seen_pairs:
                continue  # qccd stores both directions; only need one
            seen_pairs.add(pair)

            u_type = self.qccd_graph.get_node(u).node_type
            v_type = self.qccd_graph.get_node(v).node_type
            if NodeType.TRAP in (u_type, v_type) and u_type != v_type:
                label = "merge_split"
            elif u_type == NodeType.TRAP and v_type == NodeType.TRAP:
                label = "merge_split"  # not expected from the factories, but handle it
            else:
                label = "move"

            for slot_u in self._slots_by_real[u]:
                for slot_v in self._slots_by_real[v]:
                    self.graph.add_edge(slot_u, slot_v, label=label)

    # ------------------------------------------------------------
    # Interface ShawRoutingPass expects
    # ------------------------------------------------------------
    def slots_of(self, real_node_id: Any) -> List[Slot]:
        return list(self._slots_by_real[real_node_id])

    def shortest_path(self, source: Slot, target: Slot) -> List[Slot]:
        """
        Virtual-slot shortest path, built from the real qccd shortest
        path. Endpoints are kept exactly as given (so a target chosen
        for its specific free slot is honored).

        For any real node the path merely passes *through* -- which on
        topologies like LinearChain/Grid can include a TRAP node with
        no bypass around it, not just channels/junctions -- we need a
        specific slot to route the walk through. We prefer a slot that
        is actually free right now (checked via the attached
        Placement, if any) so a pass-through doesn't manufacture an
        avoidable collision with whoever already happens to be parked
        there. If every slot at that node is genuinely occupied, we
        fall back to slot 0 -- ShawRoutingPass's existing swap-on-
        collision logic in _walk_path then does a real (valid) swap
        through it rather than crashing. That fallback case is exactly
        the kind of thing Chaitanya's real congestion handler should
        eventually do better than "just pick one and swap".
        """
        real_source, real_target = self._real_of_slot[source], self._real_of_slot[target]
        real_path = self.qccd_graph.shortest_path(real_source, real_target)

        virtual_path = [source]
        for real_node in real_path[1:-1]:
            virtual_path.append(self._pick_pass_through_slot(real_node))
        virtual_path.append(target)
        return virtual_path

    def _pick_pass_through_slot(self, real_node: Any) -> Slot:
        candidate_slots = self._slots_by_real[real_node]
        if self._placement is not None:
            for slot in candidate_slots:
                if not self._placement.is_occupied(slot):
                    return slot
        return candidate_slots[0]  # no free slot (or no placement attached yet)


class QccdPlacementAdapter:
    """
    Placement backed by a QccdPositionGraphAdapter. Keeps its own
    slot-level bookkeeping (phi / occupant) and mirrors real moves into
    the underlying qccd graph via place_ion / remove_ion / move_ion.
    """

    def __init__(self, adapter: QccdPositionGraphAdapter) -> None:
        self._adapter = adapter
        self._qccd = adapter.qccd_graph
        self._phi: Dict[Any, Slot] = {}
        self._occupant: Dict[Slot, Any] = {}

    def _real_of(self, slot: Slot) -> Any:
        return self._adapter._real_of_slot[slot]

    def place(self, qudit: Any, position: Slot) -> None:
        """Initial placement of a qudit into a free slot."""
        if position in self._occupant:
            raise ValueError(f"position {position!r} is already occupied")
        self._qccd.place_ion(qudit, self._real_of(position))
        self._phi[qudit] = position
        self._occupant[position] = qudit

    def swap(self, qudit_a: Any, qudit_b: Any) -> None:
        """Exchange the positions of two placed qudits."""
        pos_a, pos_b = self._phi[qudit_a], self._phi[qudit_b]
        real_a, real_b = self._real_of(pos_a), self._real_of(pos_b)
        if real_a != real_b:
            self._qccd.remove_ion(qudit_a, real_a)
            self._qccd.remove_ion(qudit_b, real_b)
            self._qccd.place_ion(qudit_a, real_b)
            self._qccd.place_ion(qudit_b, real_a)
        self._phi[qudit_a], self._phi[qudit_b] = pos_b, pos_a
        self._occupant[pos_a], self._occupant[pos_b] = qudit_b, qudit_a

    def move(self, qudit: Any, new_position: Slot) -> None:
        """Pure shuttle to a free slot -- mirrors via qccd's move_ion,
        which validates the real edge + destination capacity for us."""
        if new_position in self._occupant:
            raise ValueError(f"position {new_position!r} is already occupied")

        old_position = self._phi[qudit]
        old_real, new_real = self._real_of(old_position), self._real_of(new_position)
        if old_real != new_real:
            self._qccd.move_ion(qudit, old_real, new_real)
        # else: same real node -- e.g. hopping between two slots that
        # both belong to one trap qudit already occupies conceptually;
        # nothing to change at the real-graph level.

        del self._occupant[old_position]
        self._phi[qudit] = new_position
        self._occupant[new_position] = qudit

    def position_of(self, qudit: Any) -> Slot:
        return self._phi[qudit]

    def occupant_of(self, position: Slot) -> Optional[Any]:
        return self._occupant.get(position)

    def is_occupied(self, position: Slot) -> bool:
        return position in self._occupant


def build_from_qccd(
    qccd_graph: QccdPositionGraph,
) -> Tuple[QccdPositionGraphAdapter, QccdPlacementAdapter]:
    """Convenience constructor: wrap a qccd graph and get back a
    (position_graph, placement) pair ready to hand to ShawRoutingPass."""
    adapter = QccdPositionGraphAdapter(qccd_graph)
    placement = QccdPlacementAdapter(adapter)
    adapter._placement = placement  # see shortest_path's pass-through note
    return adapter, placement


def seed_default_placement(
    adapter: QccdPositionGraphAdapter,
    placement: QccdPlacementAdapter,
    num_qudits: int,
) -> None:
    """
    Convenience seeding for tests/demos: place logical qudits 0..N-1
    into slot 0 of the first N TRAP nodes found in the qccd graph
    (in insertion order). Real usage would come from an actual initial
    layout/mapping pass instead.
    """
    trap_node_ids = [
        node.node_id
        for node in adapter.qccd_graph.nodes()
        if node.node_type is NodeType.TRAP
    ]
    if len(trap_node_ids) < num_qudits:
        raise ValueError(
            f"qccd graph only has {len(trap_node_ids)} TRAP nodes, "
            f"need at least {num_qudits} to seed one qudit each."
        )
    for q in range(num_qudits):
        placement.place(q, adapter.slots_of(trap_node_ids[q])[0])