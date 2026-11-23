"""Structured circuit families for routing quality and latency benchmarks."""
import math
import networkx as nx
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate, RXGate, RYGate, RZGate

FAMILIES = ("ghz_chain", "qft", "qaoa_ring", "qaoa_cubic", "vqe_brickwork", "parity_check_proxy")

def circuit_for(family: str, n: int, graph_seed: int = 17) -> Circuit:
    """Build a deterministic interaction benchmark, not a hardware fidelity model.

    parity_check_proxy has no measurement/reset and is not a complete QEC cycle.
    QFT omits final output reversal. VQE is a generic RY/CX ansatz.
    """
    if n < 2:
        raise ValueError("At least two qudits are required")
    if family == "qaoa_cubic" and (n < 4 or n % 2):
        raise ValueError("Cubic QAOA requires an even qudit count >= 4")
    c = Circuit(n)
    cx = lambda a, b: c.append_gate(CNOTGate(), (a, b))
    if family == "ghz_chain":
        c.append_gate(HGate(), (0,))
        for i in range(n - 1):
            cx(i, i + 1)
    elif family == "qft":
        # Controlled phase via RZ/CX, up to global phase; no output reversal.
        for a in range(n):
            c.append_gate(HGate(), (a,))
            for b in range(a + 1, n):
                theta = math.pi / 2 ** (b - a)
                c.append_gate(RZGate(), (a,), [theta / 2])
                cx(a, b)
                c.append_gate(RZGate(), (b,), [-theta / 2])
                cx(a, b)
                c.append_gate(RZGate(), (b,), [theta / 2])
    elif family.startswith("qaoa"):
        graph = nx.cycle_graph(n) if family == "qaoa_ring" else nx.random_regular_graph(3, n, seed=graph_seed)
        for i in range(n):
            c.append_gate(HGate(), (i,))
        for layer in range(3):
            for a, b in sorted(graph.edges()):
                cx(a, b)
                c.append_gate(RZGate(), (b,), [0.37 + layer * 0.1])
                cx(a, b)
            for i in range(n):
                c.append_gate(RXGate(), (i,), [0.23])
    elif family == "vqe_brickwork":
        for layer in range(6):
            for i in range(n):
                c.append_gate(RYGate(), (i,), [0.11 * (i + 1)])
            for a in range(layer % 2, n - 1, 2):
                cx(a, a + 1)
    elif family == "parity_check_proxy":
        # Repeated bipartite interaction kernel only: NO measurement/reset/QEC claims.
        half = n // 2
        for _ in range(3):
            for offset in range(3):
                for a in range(half):
                    cx((a + offset) % half, half + a)
    else:
        raise ValueError(family)
    return c

