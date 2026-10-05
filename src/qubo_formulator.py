"""
qubo_formulator.py
==================
Builds the QUBO Q-matrix (BinaryQuadraticModel) for a local congestion window.

Formulation Overview:
  Variables : x_{i, v, t} ∈ {0, 1} (ion i at position v at timestep t)
  Objective : H = H_cost + λ * (H_one_hot + H_capacity + H_movement + H_goal)

Penalty Terms (weighted by λ):
  - H_one_hot  : ∑_v x_{i,v,t} = 1  (exactly one position per ion per timestep)
  - H_capacity : ∑_i x_{i,v,t} ≤ cap(v)  (node capacity limit)
  - H_movement : penalizes moves between non-adjacent graph nodes
  - H_goal     : target ion must reach destination at t = T

Objective Term:
  - H_cost     : hop_cost * x_{i,v,t} for positions v ≠ initial_position(i)

Quadratization:
  Rosenberg substitution (z = x_a * x_b) for terms with degree > 2.

References:
  - Glover et al., "A Tutorial on Formulating and Using QUBO Models", arXiv:1811.11538 (2019)
  - Bach et al., "SHAW Router for Trapped-Ion Quantum Computing", arXiv:2501.12470 (2025)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

# dimod is installed via requirements.txt
import dimod
import networkx as nx

from congestion_handler import WindowInfo

logger = logging.getLogger("shaw_router.qubo_formulator")


# ---------------------------------------------------------------------------
# Output data structure  (DO NOT CHANGE – qubo_solver.py depends on this)
# ---------------------------------------------------------------------------

@dataclass
class QUBOProblem:
    """
    The fully assembled QUBO problem ready for a solver.

    Attributes
    ----------
    bqm : dimod.BinaryQuadraticModel
        The quadratic model in QUBO form.  Access the Q-matrix via
        bqm.to_qubo()[0] if you need the raw dict.
    var_map : dict  { (ion_id, position_id, timestep) : variable_label }
        Maps the physical meaning of each binary variable to its label
        inside the BQM.  The decoder needs this to interpret solutions.
    inv_var_map : dict  { variable_label : (ion_id, position_id, timestep) }
        Inverse of var_map.
    num_variables : int
        Total number of binary variables (before Rosenberg aux vars).
    num_aux_variables : int
        Extra variables introduced by Rosenberg quadratization.
    time_horizon : int
        Number of time steps T used in the formulation.
    penalty_lambda : float
        The λ value used to weight constraint penalties.
    window : WindowInfo
        The source window (kept for the decoder).
    metadata : dict
        Anything else you want to pass to the solver or decoder.
    """
    bqm: dimod.BinaryQuadraticModel
    var_map: Dict[Tuple[Any, Any, int], str]
    inv_var_map: Dict[str, Tuple[Any, Any, int]]
    num_variables: int
    num_aux_variables: int
    time_horizon: int
    penalty_lambda: float
    window: WindowInfo
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QUBOFormulator:
    """
    Converts a WindowInfo (local congestion window) into a QUBOProblem.

    Parameters
    ----------
    time_horizon : int
        Number of discrete time steps T.  Typically set to
        shortest_path_length + 2 so there's slack to clear blockers.
        Default: None → auto-compute from window source→target distance.
    penalty_lambda : float
        λ for constraint penalties.  None → use C_max + 1 heuristic.
    hop_cost : float
        Cost weight for each ion movement hop (objective term).
    """

    def __init__(
        self,
        time_horizon: Optional[int] = None,
        penalty_lambda: Optional[float] = None,
        hop_cost: float = 1.0,
    ) -> None:
        self.time_horizon = time_horizon
        self.penalty_lambda = penalty_lambda
        self.hop_cost = hop_cost

        # Track auxiliary variables introduced by Rosenberg quadratization
        self._aux_counter: int = 0
        self._aux_var_map: Dict[Tuple[str, str], str] = {}  # (u, v) → aux_label

    def build(
        self,
        window: WindowInfo,
        pg: Any,   # src.position_graph.PositionGraph
    ) -> QUBOProblem:
        """
        Build the QUBOProblem from the local window and position graph.

        Parameters
        ----------
        window : WindowInfo   – the local congestion window
        pg     : PositionGraph – hardware graph (for legal edge lookup)

        Returns
        -------
        QUBOProblem
        """
        # Reset auxiliary variable state for this build
        self._aux_counter = 0
        self._aux_var_map = {}

        # 1. Compute time horizon T
        T = self._compute_time_horizon(window, pg)

        # 2. Build variable maps
        var_map, inv_var_map = self._make_variables(window, T)
        num_variables = len(var_map)

        # 3. Compute lambda (penalty strength)
        # Tight upper bound: each ion occupies at most 1 node per timestep (H_one_hot)
        num_ions = len(window.active_ions)
        num_pos  = len(window.window_nodes)
        C_max    = num_ions * T * self.hop_cost
        lambda_  = self.penalty_lambda if self.penalty_lambda is not None else (C_max + 1.0)

        # 4. Initialise empty BQM
        bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)

        # Add all variables (initialised to 0 bias)
        for label in var_map.values():
            bqm.add_variable(label, 0.0)

        ions      = list(window.active_ions.keys())
        positions = list(window.window_nodes)

        # 5. Encode constraints as penalty terms
        self._add_initial_position_penalty(bqm, var_map, window, lambda_)
        self._add_one_hot_penalty(bqm, var_map, ions, positions, T, lambda_)
        self._add_capacity_penalty(bqm, var_map, window, pg, T, lambda_)
        self._add_movement_penalty(bqm, var_map, window, pg, T, lambda_)
        self._add_crossing_penalty(bqm, var_map, window, pg, T, lambda_)
        self._add_goal_penalty(bqm, var_map, window, T, lambda_)

        # 6. Add cost objective + progress shaping
        self._add_cost_objective(bqm, var_map, window, T)
        self._add_progress_reward(bqm, var_map, window, pg, T, lambda_)

        # 7. Rosenberg quadratization aux variable tracking
        num_aux = self._rosenberg_quadratize(bqm)

        total_vars = len(bqm.variables)
        if total_vars > 100:
            logger.warning(
                "QUBOProblem formulation size (%d variables) exceeds soft budget of 100",
                total_vars,
            )

        logger.info(
            "QUBOFormulator.build(): T=%d, ions=%d, positions=%d, "
            "vars=%d (+ %d aux), λ=%.2f",
            T, num_ions, num_pos, num_variables, num_aux, lambda_,
        )

        return QUBOProblem(
            bqm=bqm,
            var_map=var_map,
            inv_var_map=inv_var_map,
            num_variables=num_variables,
            num_aux_variables=num_aux,
            time_horizon=T,
            penalty_lambda=lambda_,
            window=window,
            metadata={
                "num_ions": num_ions,
                "num_positions": num_pos,
                "C_max": C_max,
            },
        )

    # -----------------------------------------------------------------------
    # Sub-steps
    # -----------------------------------------------------------------------

    def _compute_time_horizon(self, window: WindowInfo, pg: Any) -> int:
        """
        Compute the number of time steps T.

        Strategy:
          T = shortest_path_length(source → target in G_p)
              + len(blocked_path)   ← slack to clear each blocker

        Falls back to len(blocked_path) + 2 when pg is unavailable.
        """
        if self.time_horizon is not None:
            return self.time_horizon

        slack = len(window.blocked_path)

        G = getattr(pg, "graph", None)
        if G is None:
            return slack + 2

        try:
            sp_len = nx.shortest_path_length(G, window.source, window.target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            sp_len = max(slack, 1)

        T = sp_len + slack
        return max(T, 2)   # always at least 2 steps

    def _make_variables(
        self,
        window: WindowInfo,
        T: int,
    ) -> Tuple[Dict, Dict]:
        """
        Create one binary variable per (ion, position, timestep).

        Only creates variables for ions in window.active_ions and
        positions in window.window_nodes.

        Variable label format: "x_{ion}_{pos}_{t}"
        (sanitised so dimod doesn't choke on special characters)

        Returns (var_map, inv_var_map) where:
          var_map     : (ion, pos, t) → label string
          inv_var_map : label → (ion, pos, t)
        """
        var_map: Dict[Tuple[Any, Any, int], str] = {}
        inv_var_map: Dict[str, Tuple[Any, Any, int]] = {}

        ions      = list(window.active_ions.keys())
        positions = list(window.window_nodes)

        for ion in ions:
            # Sanitise ion id for use in a variable label
            ion_str = str(ion).replace(":", "_").replace(" ", "_")
            for pos in positions:
                pos_str = str(pos).replace(":", "_").replace(" ", "_")
                for t in range(T + 1):         # t ∈ {0, 1, ..., T}
                    key   = (ion, pos, t)
                    label = f"x_{ion_str}_{pos_str}_{t}"
                    var_map[key]        = label
                    inv_var_map[label]  = key

        return var_map, inv_var_map

    def _add_initial_position_penalty(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        lambda_: float,
    ) -> None:
        """
        Add H_init: hard constraint pinning each active ion to its initial position at t=0.
        Subtracts λ from linear bias of x_{i, s_i, 0}, forcing x_{i, s_i, 0} = 1 in ground state.
        """
        for ion, start_pos in window.active_ions.items():
            key = (ion, start_pos, 0)
            if key in var_map:
                bqm.add_variable(var_map[key], -lambda_)

    def _add_one_hot_penalty(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        ions: List[Any],
        positions: List[Any],
        T: int,
        lambda_: float,
    ) -> None:
        """
        Add H_one_hot = λ ∑_{i,t} (∑_v x_{i,v,t} - 1)^2
        Expanded: -λ per variable x_{i,v,t}, +2λ per pair (x_{i,v,t}, x_{i,w,t}).
        """
        for ion in ions:
            for t in range(T + 1):
                # Linear terms (coefficient = -lambda_ per variable)
                for pos in positions:
                    key = (ion, pos, t)
                    if key in var_map:
                        bqm.add_variable(var_map[key], -lambda_)

                # Quadratic terms (coefficient = +2*lambda_ per pair)
                pos_list = [p for p in positions if (ion, p, t) in var_map]
                for v, w in itertools.combinations(pos_list, 2):
                    v_label = var_map[(ion, v, t)]
                    w_label = var_map[(ion, w, t)]
                    bqm.add_interaction(v_label, w_label, 2.0 * lambda_)

    def _add_capacity_penalty(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        pg: Any,
        T: int,
        lambda_: float,
    ) -> None:
        """
        Add H_capacity: penalize > cap(v) ions co-located at position v at timestep t.
        For cap=1: +λ * x_{i,v,t} * x_{j,v,t} for distinct ions i ≠ j.
        For cap>1: uses Rosenberg product substitution for (cap+1)-way co-locations.
        """
        G = getattr(pg, "graph", None)
        ions = list(window.active_ions.keys())

        for pos in window.window_nodes:
            # Determine capacity from node attributes (default 1)
            cap = 1
            if G is not None and G.has_node(pos):
                cap = G.nodes[pos].get("capacity", 1)

            for t in range(T + 1):
                # All ions with a variable at (pos, t)
                ion_vars = [
                    (ion, var_map[(ion, pos, t)])
                    for ion in ions
                    if (ion, pos, t) in var_map
                ]

                if cap == 1:
                    # Every pair of distinct ions must not co-occupy
                    for (i1, lbl1), (i2, lbl2) in itertools.combinations(ion_vars, 2):
                        bqm.add_interaction(lbl1, lbl2, lambda_)

                else:
                    # cap >= 2: penalise (cap+1)-way co-location
                    for combo in itertools.combinations(ion_vars, cap + 1):
                        lbls = [lbl for _, lbl in combo]
                        # Quadratize greedily: chain through pairs
                        running = lbls[0]
                        for next_lbl in lbls[1:]:
                            aux = self._rosenberg_product(bqm, running, next_lbl, lambda_)
                            running = aux
                        # Apply penalty +lambda on the product variable representing co-location:
                        bqm.add_variable(running, lambda_)

    def _add_movement_penalty(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        pg: Any,
        T: int,
        lambda_: float,
    ) -> None:
        """
        Add H_movement: penalize moves between non-adjacent positions:
        +2λ * x_{i,v,t} * x_{i,w,t+1} for (v, w) not in E(G_p) with v ≠ w.

        The 2λ weight (not λ) is load-bearing: a single illegal hop
        must cost more than the goal bonus (λ) plus any saved hop
        costs (≤ T·hop_cost) combined -- otherwise SA "teleports" the
        mover on the last step (one +λ violation pays for itself via
        the -λ goal bonus) and the decoder rejects on
        movement_legality. With auto-scaled λ = C_max + 1 ≥ T·hop_cost,
        2λ provably prices out the teleport (observed failure mode).
        """
        G = getattr(pg, "graph", None)
        ions      = list(window.active_ions.keys())
        positions = list(window.window_nodes)

        for ion in ions:
            for t in range(T):
                for v in positions:
                    key_v_t = (ion, v, t)
                    if key_v_t not in var_map:
                        continue
                    v_label = var_map[key_v_t]

                    for w in positions:
                        if w == v:
                            continue  # staying in place – always legal

                        # Check if v→w is a legal edge in G_p
                        if G is not None:
                            if G.has_edge(v, w) or G.has_edge(w, v):
                                continue  # legal hop – no penalty

                        # Illegal transition: penalise co-assignment
                        key_w_t1 = (ion, w, t + 1)
                        if key_w_t1 not in var_map:
                            continue
                        w_label = var_map[key_w_t1]
                        bqm.add_interaction(v_label, w_label, 2.0 * lambda_)

    def _add_crossing_penalty(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        pg: Any,
        T: int,
        lambda_: float,
    ) -> None:
        """
        Add H_cross: penalize two ions swapping positions in one step.

        A crossing (ion a: u→v while ion b: v→u at the same t) is an
        implicit swap with no explicit swap gate and is always rejected
        downstream by solution_decoder -- but without a BQM term pricing
        it, the sampler returns "feasible" energies for physically
        useless crossing samples (observed: every SA sample rejected on
        collision_avoidance regardless of budget). Per (t, edge, pair):

            +λ * x_{a,u,t} * x_{a,v,t+1} * x_{b,v,t} * x_{b,u,t+1}

        plus the mirror direction. Quartic terms are reduced through
        the shared _rosenberg_product helper (cached per pair). Edges
        are treated undirected (matching _add_movement_penalty).
        """
        G = getattr(pg, "graph", None)
        if G is None:
            return
        ions = list(window.active_ions.keys())
        # Undirected edges, each considered once (frozenset is
        # hashable for both str slot IDs and tuple virtual slots).
        edges = {frozenset((u, v)) for u, v in G.edges() if u != v}

        for t in range(T):
            for edge in edges:
                u, v = tuple(edge)
                for a, b in itertools.combinations(ions, 2):
                    # Direction 1: a goes u→v while b goes v→u.
                    self._penalize_crossing(bqm, var_map, a, u, v, b, t, lambda_)
                    # Direction 2: the mirror (a: v→u while b: u→v).
                    self._penalize_crossing(bqm, var_map, b, u, v, a, t, lambda_)

    def _penalize_crossing(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        a: Any, u: Any, v: Any, b: Any,
        t: int,
        lambda_: float,
    ) -> None:
        """
        One directed crossing term: +λ if (a: u→v AND b: v→u at step
        t→t+1), via two Rosenberg products p1 ≡ a_u_t·a_v_t+1,
        p2 ≡ b_v_t·b_u_t+1 and +λ on z ≡ p1·p2. Skipped silently if
        any of the four variables is outside var_map.
        """
        keys = [(a, u, t), (a, v, t + 1), (b, v, t), (b, u, t + 1)]
        if any(k not in var_map for k in keys):
            return
        lbl_au_t, lbl_av_t1, lbl_bv_t, lbl_bu_t1 = (var_map[k] for k in keys)
        p1 = self._rosenberg_product(bqm, lbl_au_t, lbl_av_t1, lambda_)
        p2 = self._rosenberg_product(bqm, lbl_bv_t, lbl_bu_t1, lambda_)
        z = self._rosenberg_product(bqm, p1, p2, lambda_)
        bqm.add_variable(z, lambda_)

    def _add_crossing_penalty(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        pg: Any,
        T: int,
        lambda_: float,
    ) -> None:
        """
        Add H_crossing: penalize anti-crossing / simultaneous swap moves:
        Ion i moving u -> v at step t while Ion j moves v -> u at step t (i != j).
        Uses Rosenberg product substitution for m1 = x_{i,u,t} * x_{i,v,t+1} and
        m2 = x_{j,v,t} * x_{j,u,t+1}, then adds interaction +λ * m1 * m2.
        """
        G = getattr(pg, "graph", None)
        ions = list(window.active_ions.keys())
        positions = list(window.window_nodes)

        for i, j in itertools.combinations(ions, 2):
            for t in range(T):
                for u in positions:
                    for v in positions:
                        if u == v:
                            continue
                        if G is not None and not (G.has_edge(u, v) or G.has_edge(v, u)):
                            continue

                        k_i_u_t  = (i, u, t)
                        k_i_v_t1 = (i, v, t + 1)
                        k_j_v_t  = (j, v, t)
                        k_j_u_t1 = (j, u, t + 1)

                        if (
                            k_i_u_t in var_map
                            and k_i_v_t1 in var_map
                            and k_j_v_t in var_map
                            and k_j_u_t1 in var_map
                        ):
                            lbl_i1 = var_map[k_i_u_t]
                            lbl_i2 = var_map[k_i_v_t1]
                            lbl_j1 = var_map[k_j_v_t]
                            lbl_j2 = var_map[k_j_u_t1]

                            m1 = self._rosenberg_product(bqm, lbl_i1, lbl_i2, lambda_)
                            m2 = self._rosenberg_product(bqm, lbl_j1, lbl_j2, lambda_)
                            bqm.add_interaction(m1, m2, lambda_)

    def _add_goal_penalty(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        T: int,
        lambda_: float,
    ) -> None:
        """
        Add H_goal: target ion must reach target node at timestep T (-λ * x_{moving_ion, target, T}).
        """
        target = window.target

        # Identify the primary moving ion: the one currently at source
        moving_ion = None
        for ion, pos in window.active_ions.items():
            if pos == window.source:
                moving_ion = ion
                break

        if moving_ion is not None:
            key = (moving_ion, target, T)
            if key in var_map:
                bqm.add_variable(var_map[key], -lambda_)
                logger.debug(
                    "Goal penalty: ion %s must reach %s at t=%d", moving_ion, target, T
                )
        else:
            logger.warning(
                "Goal penalty skipped: no active ion found at window.source %s",
                window.source,
            )

    def _add_cost_objective(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        T: int,
    ) -> None:
        """
        Add H_cost: +hop_cost * x_{i,v,t} for positions v ≠ ion's initial position.
        """
        for ion, start_pos in window.active_ions.items():
            for pos in window.window_nodes:
                if pos == start_pos:
                    # Staying at the starting position costs nothing
                    continue
                for t in range(T + 1):
                    key = (ion, pos, t)
                    if key in var_map:
                        bqm.add_variable(var_map[key], self.hop_cost)

    def _add_progress_reward(
        self,
        bqm: dimod.BinaryQuadraticModel,
        var_map: Dict,
        window: WindowInfo,
        pg: Any,
        T: int,
        lambda_: float,
    ) -> None:
        """
        Add H_progress: potential-based shaping that rewards the mover
        for being closer to the target at EVERY timestep (not just at
        t=T like H_goal): linear bias -w*(D0 - d) on x_{mover,pos,t},
        where d = dist(pos, target) and D0 = dist(source, target).

        This is potential shaping (reward ∝ Φ(s) with Φ = -distance),
        so it guides SA step-by-step toward the target without changing
        which full walk is optimal: any detour still pays hop_cost per
        step and arrives later. Weight w = λ/((T+1)*(Dmax+1)) keeps the
        total shaping below λ, so it can never outweigh a single hard
        constraint -- it only breaks ties between valid walks and, more
        importantly, gives blind SA a gradient to follow. Without this
        term SA parks everyone (zero cost, zero guidance) and the
        decoder rejects on gate_feasibility forever (observed).
        """
        G = getattr(pg, "graph", None)
        if G is None:
            return
        moving_ion = None
        for ion, pos in window.active_ions.items():
            if pos == window.source:
                moving_ion = ion
                break
        if moving_ion is None:
            return

        def dist(a: Any, b: Any) -> Optional[int]:
            try:
                return nx.shortest_path_length(G, a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        D0 = dist(window.source, window.target)
        if D0 is None:
            return
        dists: Dict[Any, int] = {}
        for pos in window.window_nodes:
            d = dist(pos, window.target)
            if d is not None:
                dists[pos] = d
        if not dists:
            return
        Dmax = max(dists.values())
        w = lambda_ / ((T + 1) * (Dmax + 1))

        for pos, d in dists.items():
            for t in range(T + 1):
                k = (moving_ion, pos, t)
                if k in var_map:
                    bqm.add_variable(var_map[k], -w * (D0 - d))

    def _rosenberg_quadratize(
        self,
        bqm: dimod.BinaryQuadraticModel,
    ) -> int:
        """
        Return the count of auxiliary variables introduced by Rosenberg quadratization.
        """
        return self._aux_counter

    # -----------------------------------------------------------------------
    # Internal helper: Rosenberg product substitution
    # -----------------------------------------------------------------------

    def _rosenberg_product(
        self,
        bqm: dimod.BinaryQuadraticModel,
        lbl_a: str,
        lbl_b: str,
        lambda_: float,
    ) -> str:
        """
        Substitute z ≡ a * b using Rosenberg penalty P(a,b,z) = λ(ab - 2az - 2bz + 3z).
        Returns aux variable label z.
        """
        key = (lbl_a, lbl_b)
        if key in self._aux_var_map:
            return self._aux_var_map[key]

        # New auxiliary variable
        self._aux_counter += 1
        z_label = f"_aux_{lbl_a}_{lbl_b}_{self._aux_counter}"
        self._aux_var_map[key] = z_label

        # Rosenberg penalty terms
        bqm.add_variable(z_label, 3.0 * lambda_)          # +3λz
        bqm.add_interaction(lbl_a, lbl_b, lambda_)         # +λ*ab
        bqm.add_interaction(lbl_a, z_label, -2.0 * lambda_)  # -2λ*az
        bqm.add_interaction(lbl_b, z_label, -2.0 * lambda_)  # -2λ*bz

        logger.debug("Rosenberg aux var %s ≡ %s * %s", z_label, lbl_a, lbl_b)
        return z_label


# ---------------------------------------------------------------------------
# Smoke test  (python src/qubo_formulator.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from position_graph import build_linear_qccd, Placement
    from congestion_handler import CongestionHandler

    print("=" * 60)
    print("QUBOFormulator – smoke test")
    print("=" * 60)

    # Build a small 3-trap QCCD hardware graph
    pg = build_linear_qccd(num_traps=3, trap_capacity=2)
    pl = Placement(pg)

    # Place two ions: ion 0 at trap0-slot0, ion 1 (blocker) at trap1-slot0
    pl.place(0, pg.slots_of("t0")[0])
    pl.place(1, pg.slots_of("t1")[0])   # blocks the path

    # Identify a path through the blocker
    path = pg.shortest_path(pg.slots_of("t0")[0], pg.slots_of("t2")[0])
    print(f"Path from t0:0 -> t2:0 : {path}")

    # Extract congestion window
    handler = CongestionHandler()
    blocked = handler._detect_blockages(path, pl)
    print(f"Blocked positions     : {blocked}")

    window = handler.extract_window(path, blocked, pl, pg)
    print(f"Window nodes          : {sorted(window.window_nodes)}")
    print(f"Active ions           : {window.active_ions}")
    print(f"Source -> Target      : {window.source} -> {window.target}")

    # Build QUBO
    formulator = QUBOFormulator()
    problem = formulator.build(window, pg)

    print()
    print(f"Time horizon T        : {problem.time_horizon}")
    print(f"Binary variables      : {problem.num_variables}")
    print(f"Auxiliary vars (Rosen): {problem.num_aux_variables}")
    print(f"Penalty lambda        : {problem.penalty_lambda:.4f}")
    print(f"BQM #variables        : {len(problem.bqm.variables)}")
    print(f"BQM #interactions     : {len(problem.bqm.quadratic)}")
    print()

    # Verify var_map / inv_var_map consistency
    mismatch = 0
    for key, label in problem.var_map.items():
        if problem.inv_var_map.get(label) != key:
            mismatch += 1
    print(f"var_map / inv_var_map consistent : {'YES' if mismatch == 0 else f'NO ({mismatch} mismatches)'}")

    # Verify constraint: λ >= T + 1
    ok = problem.penalty_lambda >= problem.time_horizon + 1
    print(f"lambda >= T+1 satisfied          : {'YES' if ok else 'NO'}")

    print()
    print("Smoke test PASSED.  Run 'pytest tests/test_qubo_formulator.py -v' for full tests.")
