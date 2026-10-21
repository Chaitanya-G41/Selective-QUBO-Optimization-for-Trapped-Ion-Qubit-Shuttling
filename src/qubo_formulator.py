"""
qubo_formulator.py
==================
Builds the QUBO for one local congestion window.

Variables
  x[i, v, t] = 1  iff ion i sits at position v at time t   (t = 0..T)

  Only *reachable* states get a variable: ion i can be at v at time t
  only if dist(start_i, v) <= t and v can still reach the ion's final
  set F_i in the remaining T - t steps. This pruning keeps the model
  small and makes the start position and the goal hard by construction:
    - the mover must end on a slot of the goal trap,
    - ions that start in the goal trap (e.g. the gate partner) must end
      there too,
    - every other ion must end on a trap slot (never parked on transport).

Energy  E = H_cost + P * (H_one_hot + H_capacity + H_movement + H_crossing)
  H_cost      price per hop: x[i,v,t] * x[i,w,t+1] for adjacent v != w,
              1 by default or by edge type under a cost model
  H_one_hot   (sum_v x[i,v,t] - 1)^2 per ion and time step
  H_capacity  x[i,v,t] * x[j,v,t] for i != j        (one ion per position)
  H_movement  x[i,v,t] * x[i,w,t+1] for non-adjacent v, w
  H_crossing  two ions exchanging over a transport edge in one step
              (quartic, reduced with Rosenberg products). Exchanges over
              in-trap "swap" edges are legal: they are physical ion swaps.
  H_lookahead w * dist(final position, position of the ion's next gate
              partner), for ions with an upcoming gate. Steers plans
              toward layouts that keep later gates cheap.
  H_settle    tie-break for the other ions: eps per ion that does not
              end where it started (eps * #ions < 1, never worth a hop).

P exceeds the largest possible objective, so any constraint violation
costs more than every feasible plan; a feasible sample's energy is its
hop cost plus the terminal terms. Rosenberg penalties use 2P so an auxiliary variable
can never "cheat" a product term.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

import dimod
import networkx as nx
import numpy as np

from walk_planner import UNIT_COSTS

Key = Tuple[Any, Any, int]  # (ion, position, t)


@dataclass
class WindowInfo:
    """The bounded local window the QUBO is solved on."""
    window_nodes: FrozenSet[Any]
    active_ions: Dict[Any, Any]      # ion -> start position (all ions inside W)
    mover: Any                       # ion SHAW is trying to route
    goal_nodes: FrozenSet[Any]       # mover must end on one of these
    source: Any
    target: Any
    blocked_path: list = field(default_factory=list)
    next_partner_pos: Dict[Any, Any] = field(default_factory=dict)  # ion -> partner position


@dataclass
class QUBOProblem:
    bqm: dimod.BinaryQuadraticModel
    var_map: Dict[Key, str]
    inv_var_map: Dict[str, Key]
    time_horizon: int
    penalty: float
    window: WindowInfo
    graph: nx.Graph                  # window subgraph used for the model
    aux_products: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # z -> (a, b)

    @property
    def num_variables(self) -> int:
        return len(self.var_map)

    @property
    def num_aux_variables(self) -> int:
        return len(self.aux_products)

    def complete(self, sample: Dict[str, int]) -> Dict[str, int]:
        """Copy of `sample` with every aux variable set to its product
        (aux labels are numbered so each one's inputs come first)."""
        full = {v: sample.get(v, 0) for v in self.bqm.variables}
        for z, (a, b) in self.aux_products.items():
            full[z] = full[a] * full[b]
        return full


class _Terms:
    """Collects BQM coefficients in plain arrays and builds the BQM in one
    call: adding terms one by one through dimod is ~5x slower."""

    def __init__(self, labels) -> None:
        self.labels: List[str] = []
        self.index: Dict[str, int] = {}
        self.linear: List[float] = []
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.values: List[float] = []
        self.offset = 0.0
        for label in labels:
            self.add_variable(label)

    def add_variable(self, label: str) -> None:
        self.index[label] = len(self.labels)
        self.labels.append(label)
        self.linear.append(0.0)

    def add_linear(self, label: str, bias: float) -> None:
        self.linear[self.index[label]] += bias

    def add_quadratic(self, a: str, b: str, bias: float) -> None:
        self.rows.append(self.index[a])
        self.cols.append(self.index[b])
        self.values.append(bias)

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        return dimod.BinaryQuadraticModel.from_numpy_vectors(
            np.asarray(self.linear),
            (np.asarray(self.rows, dtype=int), np.asarray(self.cols, dtype=int),
             np.asarray(self.values)),
            self.offset,
            dimod.BINARY,
            variable_order=self.labels,
        )


class InfeasibleWindow(ValueError):
    """The window admits no plan within the time horizon."""


class QUBOFormulator:
    """
    Parameters
    ----------
    time_horizon : fixed T, or None for dist(source, goal) + slack
    slack        : extra steps beyond the mover's shortest route, giving
                   blockers time to step aside
    penalty      : constraint weight P, or None for the provable bound
    lookahead_weight : weight w of H_lookahead (ops per hop of distance)
    costs            : cost model pricing hops by edge label (default: one
                       unit per operation; see walk_planner.TIME_COSTS_US)
    """

    def __init__(
        self,
        time_horizon: Optional[int] = None,
        slack: int = 2,
        penalty: Optional[float] = None,
        lookahead_weight: float = 0.5,
        costs: Optional[Dict[str, float]] = None,
    ) -> None:
        self.time_horizon = time_horizon
        self.slack = slack
        self.penalty = penalty
        self.lookahead_weight = lookahead_weight
        self.costs = costs or UNIT_COSTS

    def build(self, window: WindowInfo, pg: Any, min_horizon: int = 0) -> QUBOProblem:
        # A concrete copy: lookups on subgraph views are slow.
        G = nx.Graph(pg.graph.subgraph(window.window_nodes))
        adjacent = {v: set(G[v]) for v in G}
        goal = frozenset(window.goal_nodes) & G.nodes
        slots = frozenset(v for v in G if G.nodes[v].get("trap_id") is not None)
        if window.mover not in window.active_ions or not goal:
            raise InfeasibleWindow("mover or goal outside the window")

        to_goal = self._distances_to(G, goal)
        to_slot = self._distances_to(G, slots)
        d0 = to_goal.get(window.active_ions[window.mover])
        if d0 is None:
            raise InfeasibleWindow("goal unreachable inside the window")
        T = self.time_horizon if self.time_horizon is not None else max(d0 + self.slack, min_horizon, 1)

        # ---- reachable-state variables --------------------------------
        var_map: Dict[Key, str] = {}
        states: Dict[Tuple[Any, int], list] = {}
        for ion, start in window.active_ions.items():
            ends_in_goal = ion == window.mover or start in goal
            to_final = to_goal if ends_in_goal else to_slot
            from_start = nx.single_source_shortest_path_length(G, start, cutoff=T)
            for t in range(T + 1):
                here = [
                    v for v, d in from_start.items()
                    if d <= t and to_final.get(v, T + 1) <= T - t
                ]
                if not here:
                    raise InfeasibleWindow(f"ion {ion} has no feasible state at t={t}")
                states[ion, t] = here
                for v in here:
                    var_map[ion, v, t] = f"x[{ion}|{v}|{t}]"

        ions = list(window.active_ions)
        hop = {label: self.costs[label] for label in ("swap", "merge_split", "move")}
        unit = sum(hop.values()) / len(hop)  # typical hop price: scales terminal terms
        eps = min(hop.values()) / (len(ions) + 1)
        terminal: Dict[Key, float] = {}
        for ion, start in window.active_ions.items():
            partner_pos = window.next_partner_pos.get(ion)
            if partner_pos is not None:
                dist = nx.single_source_shortest_path_length(pg.graph, partner_pos)
                for v in states[ion, T]:
                    terminal[ion, v, T] = self.lookahead_weight * unit * dist[v]
            elif ion != window.mover:
                for v in states[ion, T]:
                    terminal[ion, v, T] = eps if v != start else 0.0
        # P exceeds the largest possible objective (all hops + terminal).
        max_terminal = sum(
            max(terminal.get((ion, v, T), 0.0) for v in states[ion, T]) for ion in ions
        )
        P = (self.penalty if self.penalty is not None
             else max(hop.values()) * (1.0 + len(ions) * T) + max_terminal)
        bqm = _Terms(var_map.values())
        x = var_map

        # ---- one-hot: (sum - 1)^2 = 1 - sum + 2 * pairs -----------------
        for (ion, t), here in states.items():
            bqm.offset += P
            for v in here:
                bqm.add_linear(x[ion, v, t], -P)
            for v, w in itertools.combinations(here, 2):
                bqm.add_quadratic(x[ion, v, t], x[ion, w, t], 2 * P)

        # ---- capacity: at most one ion per position ---------------------
        for t in range(T + 1):
            for i, j in itertools.combinations(ions, 2):
                shared = set(states[j, t])
                for v in states[i, t]:
                    if v in shared:
                        bqm.add_quadratic(x[i, v, t], x[j, v, t], P)

        # ---- hop cost + illegal moves; index moves for crossings --------
        moves: Dict[Tuple[int, Any, Any], List[Any]] = {}  # (t, u, v) -> ions
        for ion in ions:
            for t in range(T):
                for v in states[ion, t]:
                    for w in states[ion, t + 1]:
                        if v == w:
                            continue
                        legal = w in adjacent[v]
                        price = hop[G.edges[v, w]["label"]] if legal else P
                        bqm.add_quadratic(x[ion, v, t], x[ion, w, t + 1], price)
                        if legal:
                            moves.setdefault((t, v, w), []).append(ion)

        # ---- terminal layout: lookahead, else settle back ---------------
        for key, cost in terminal.items():
            bqm.add_linear(x[key], cost)

        # ---- no exchanges over transport edges --------------------------
        products: Dict[FrozenSet[str], str] = {}
        aux_products: Dict[str, Tuple[str, str]] = {}

        def product(a: str, b: str) -> str:
            """Aux z with z == a*b enforced by a 2P Rosenberg penalty."""
            key = frozenset((a, b))
            if key not in products:
                z = f"aux{len(products)}"
                products[key] = z
                aux_products[z] = (a, b)
                bqm.add_variable(z)
                bqm.add_linear(z, 3 * 2 * P)
                bqm.add_quadratic(a, b, 2 * P)
                bqm.add_quadratic(a, z, -2 * 2 * P)
                bqm.add_quadratic(b, z, -2 * 2 * P)
            return products[key]

        for u, v, label in G.edges(data="label"):
            if label == "swap":
                continue
            for t in range(T):
                for a in moves.get((t, u, v), ()):
                    for b in moves.get((t, v, u), ()):
                        if a == b:
                            continue
                        za = product(x[a, u, t], x[a, v, t + 1])
                        zb = product(x[b, v, t], x[b, u, t + 1])
                        bqm.add_quadratic(za, zb, P)

        return QUBOProblem(
            bqm=bqm.to_bqm(),
            var_map=var_map,
            inv_var_map={label: key for key, label in var_map.items()},
            time_horizon=T,
            penalty=P,
            window=window,
            graph=G,
            aux_products=aux_products,
        )

    @staticmethod
    def _distances_to(G: nx.Graph, targets: Iterable[Any]) -> Dict[Any, int]:
        """Hop distance from every node to the nearest node in `targets`."""
        targets = list(targets)
        if not targets:
            return {}
        return nx.multi_source_dijkstra_path_length(G, targets, weight=None)
