"""
frontend/views/method.py
Method Tab: concise pipeline, trigger rule, Hamiltonian table,
acceptance rule, cost model, and references.
"""

import streamlit as st


def render_method_tab() -> None:
    st.markdown("#### 6-Step Selective QUBO Pipeline")
    st.markdown(
        """
1. **Heuristic routing**: route uncongested gates with the SHAW greedy walk on $G_p$.
2. **Contention monitoring**: on a blocked path, compute $\\kappa$, $\\rho$, and blockage depth $d$.
3. **Selective escalation**: if triggered and greedy exceeds lower bound $C_{LB}$, build a pruned window QUBO warm-started from greedy.
4. **Explicit validation**: decode and verify every sample against all 7 physical legality rules.
5. **Never worse locally**: accept only if `plan_cost < greedy_cost` and `layout_cost <= greedy_layout_cost`.
6. **Whole-circuit safeguard**: compile both routes and keep the cheaper complete circuit.
        """
    )

    st.markdown("#### Trigger & Acceptance Rules")
    st.markdown(
        "- **Trigger**: $(\\kappa > \\kappa_{\\text{th}} \\land \\rho > \\rho_{\\text{th}}) \\lor d > d_{\\text{th}}$\n"
        "- **Acceptance**: `plan_cost < greedy_cost` and `layout_cost <= greedy_layout_cost`"
    )

    st.markdown("#### Hamiltonian Terms")
    st.dataframe(
        [
            {"Term": "H_cost", "Form": "∑ c(u,v) x(i,u,t) x(i,v,t+1)", "Role": "Physical hop or duration cost"},
            {"Term": "H_lookahead", "Form": "w dist(v, partner) x(i,v,T)", "Role": "Leave ions near next gate partners"},
            {"Term": "H_settle", "Form": "ε x(i,v,T) for v ≠ start", "Role": "Tie-break against moving bystanders"},
            {"Term": "H_one-hot", "Form": "P ∑ (∑_v x(i,v,t) - 1)²", "Role": "One position per ion per step"},
            {"Term": "H_capacity", "Form": "P ∑ x(i,v,t) x(j,v,t)", "Role": "At most one ion per slot or segment"},
            {"Term": "H_movement", "Form": "P ∑ x(i,u,t) x(i,w,t+1)", "Role": "Only connected edges or stay"},
            {"Term": "H_crossing", "Form": "2P Rosenberg(x_a,u,t · x_b,v,t)", "Role": "Forbid anti-parallel transport swaps"},
        ],
        width="stretch",
    )

    st.markdown("#### Cost Model (`TIME_COSTS_US`)")
    st.markdown("`ion_swap`: **300 µs** · `merge_split`: **80 µs** · `move`: **10 µs** · `swap` (in-trap): **10 µs**")

    st.markdown("#### References")
    st.markdown(
        "- Bach, Safro, Younis — *SHAW Position Graph Abstraction*, [arXiv:2501.12470](https://arxiv.org/abs/2501.12470)\n"
        "- Glover, Kochenberger, Du — *QUBO Models*, [arXiv:1811.11538](https://arxiv.org/abs/1811.11538)\n"
        "- Cyclone — *Fast Shuttling Timings*, [arXiv:2511.15910](https://arxiv.org/pdf/2511.15910)\n"
        "- Moveless — *Physical Ion Swaps*, [arXiv:2508.03914](https://arxiv.org/pdf/2508.03914)\n"
        "- BQSKit (LBNL) · dimod / dwave-samplers (D-Wave)"
    )
