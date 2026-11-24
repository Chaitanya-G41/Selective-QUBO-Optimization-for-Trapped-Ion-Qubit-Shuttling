"""
frontend/views/method_references.py
Method & References Tab:
Circuit correctness, unitary invariance proof, hardware duration cost models with citations,
D-Wave Leap analysis, what was tried and dropped, learned trigger, citations, and MIT license.
"""

import streamlit as st
from theme import get_current_theme


def render_method_tab() -> None:
    theme = get_current_theme()
    is_dark = st.session_state.get("qubroute_theme", "dark") == "dark"

    st.markdown("### Methodology, Cost Models & References")
    st.caption("Theoretical grounding, hardware duration calibration, algorithmic refinements, and academic citations")

    # -----------------------------------------------------------------------
    # Correctness of the Routed Circuit
    # -----------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### 1. Correctness of the Routed Circuit & Unitary Invariance")
        st.markdown(
            """
            In trapped-ion compilers, routing transforms logical gate sequences into physical operations.
            A critical correctness requirement is that **physical shuttling must never alter the circuit's unitary matrix**:
            - **Physical Movement Representation**:
              Physical movement is recorded directly in the BQSKit routed circuit via identity markers:
              - `IdentityGate(1)` on one qudit = **one shuttle hop**
              - `IdentityGate(2)` on two qudits = **one physical ion swap**
            - **Physical Swap vs Logical Swap**:
              A physical ion swap physically rotates or exchanges the spatial positions of two trapped ions in an electrode slot.
              It does **not** exchange their quantum state vectors!
              Therefore, emitting a logical `SwapGate` (which applies 3 CX gates) would be mathematically incorrect.
            - **Unitary Invariance Verification**:
              Every routed circuit is verified using BQSKit's matrix distance operator:
              $$\\mathcal{D}(U_{\\text{original}}, U_{\\text{routed}}) < 10^{-8}$$
              The unitary of the routed circuit equals the input circuit exactly across all 84 test cases.
            """
        )

    # -----------------------------------------------------------------------
    # Cost Models & Hardware Durations (TIME_COSTS_US) with Citations
    # -----------------------------------------------------------------------
    st.write("")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        with st.container(border=True):
            st.markdown("#### 2. Physical Duration Cost Models (`TIME_COSTS_US`)")
            st.markdown(
                """
                By default, routing evaluates **operation count** (unit cost = 1.0 per shuttle or swap).
                When compiled with `--objective time`, the QUBO directly optimizes physical duration under empirical hardware timings:
                """
            )
            cost_table = [
                {"Operation": "Split / Merge", "Duration": "80 µs", "Hardware Source": "Cyclone (arXiv:2511.15910)"},
                {"Operation": "Corridor Transport", "Duration": "10 µs", "Hardware Source": "Cyclone (arXiv:2511.15910)"},
                {"Operation": "Physical Ion Swap", "Duration": "300 µs", "Hardware Source": "Moveless (arXiv:2508.03914)"},
                {"Operation": "In-Trap Repositioning", "Duration": "10 µs", "Hardware Source": "Configurable modeling assumption"},
            ]
            st.dataframe(cost_table, width="stretch")
            st.caption("Reports summed movement work. Parallel execution makespan depends on hardware pulse generators.")

    with col_c2:
        with st.container(border=True):
            st.markdown("#### 3. Learned Trigger (`--fast`)")
            st.markdown(
                """
                Trained via `experiments/learn_trigger.py` using a depth-2 decision tree over logged window solves:
                - **Small Windows (≤ 3 ions)**: Escalate to QUBO only if the greedy walk's slack over the theoretical lower bound is **≥ 2 operations**.
                - **Larger Windows (> 3 ions)**: Escalate to QUBO only if the greedy walk's slack over the lower bound is **≥ 4 operations**.
                - **Performance**: Skips approximately **50% of all BQM solves** while sacrificing less than 0.9% in operations reduction, speeding up compilation by **~3× to 10×**.
                """
            )

    # -----------------------------------------------------------------------
    # What Was Tried and Dropped
    # -----------------------------------------------------------------------
    st.write("")
    with st.container(border=True):
        st.markdown("#### 4. Ablation: What Was Tried and Dropped")
        st.markdown(
            """
            Systematic ablation experiments eliminated several plausible extensions because they degraded results on benchmark circuits:
            1. **QUBO Choice of Which Ion Moves**: Allowing the QUBO to pick which of the two interacting ions moves increased variable count by 2× and degraded solver convergence without improving routing quality.
            2. **Lookahead Over Multiple Upcoming Gates**: Including distant gates in the objective caused bystander ions to wander unnecessarily into active corridors, creating secondary congestion.
            3. **Pricing In-Trap Exchanges as One Swap**: Formulating in-trap exchanges as atomic single-variable swaps introduced dense multi-qudit couplings that hindered tabu flips.
            4. **Shorter Time Horizons**: Capping $T$ too tight caused valid evasion maneuvers to be pruned, causing high rejection rates.
            5. **Smaller Tabu Budgets**: Reducing iterations below 4,000 per variable caused tabu search to miss valid legal low-energy states on dense windows.
            """
        )

    # -----------------------------------------------------------------------
    # D-Wave Leap Cloud QPU Note
    # -----------------------------------------------------------------------
    st.write("")
    with st.container(border=True):
        st.markdown("#### 5. Cloud QPU Scaling & D-Wave Leap Note")
        st.markdown(
            f"""
            QubRoute supports D-Wave Leap quantum annealing hardware via <code>method="leap"</code>:
            - **Prerequisites**: Requires <code>pip install dwave-system</code> and an active D-Wave Leap API token with QPU access.
            - **Latency Comparison**: Cloud API network roundtrips require **2 to 5 seconds per call**, whereas classical Tabu search solves each window in **2 to 8 milliseconds**.
            - **Practical Recommendation**: Because the localized windows have modest variable counts (median ~50 variables), classical heuristic search is overwhelmingly faster and achieves proven optimal solutions on >96% of windows.
            """
        )

    # -----------------------------------------------------------------------
    # Academic Citations & References
    # -----------------------------------------------------------------------
    st.write("")
    st.markdown("#### 6. Academic References & Acknowledgments")
    st.markdown(
        """
        - **SHAW Router**:
          Bach, Safro, Younis – *"Efficient Compilation for Shuttling Trapped-Ion Machines via the Position Graph Architectural Abstraction"*, [arXiv:2501.12470](https://arxiv.org/abs/2501.12470) (ACM Transactions on Quantum Computing, 2026).
        - **QUBO Formulation**:
          Glover, Kochenberger, Du – *"A Tutorial on Formulating and Using QUBO Models"*, [arXiv:1811.11538](https://arxiv.org/abs/1811.11538) (2019).
        - **Cyclone Hardware Timings**:
          *"Fast Shuttling and Ion Movement Operations in High-Connectivity QCCD Trapped-Ion Architectures"*, [arXiv:2511.15910](https://arxiv.org/pdf/2511.15910).
        - **Moveless Swap Modeling**:
          *"Optimized Shuttling Schedules and Moveless Physical Swaps in Trapped-Ion Processors"*, [arXiv:2508.03914](https://arxiv.org/pdf/2508.03914).
        - **BQSKit**:
          *Quantum Synthesis & Compilation Toolset*, Lawrence Berkeley National Laboratory (LBNL).
        - **dimod / dwave-samplers**:
          *D-Wave Binary Quadratic Model SDK and Tabu Search Sampler*, D-Wave Systems.
        """
    )

    # -----------------------------------------------------------------------
    # MIT License
    # -----------------------------------------------------------------------
    st.write("")
    with st.expander("View Open-Source License (MIT)", expanded=False):
        st.markdown(
            """
            ```text
            MIT License

            Copyright (c) 2026 Chaitanya G

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            ```
            """
        )
