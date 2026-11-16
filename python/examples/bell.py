"""Bell state: exact amplitudes and seeded shot counts.

Run with the quantumforge wheel installed:
    python python/examples/bell.py
"""

from quantumforge import Circuit, StateVector


def main() -> None:
    circuit = Circuit()
    circuit.add_gate("H", 0)
    circuit.add_cnot(0, 1)

    # Exact state-vector simulation.
    state = circuit.run(StateVector.zero(2))
    probs = state.probabilities()
    print("Bell state probabilities:", [round(p, 6) for p in probs])
    assert abs(probs[0] - 0.5) < 1e-12 and abs(probs[3] - 0.5) < 1e-12

    # Seeded sampling: reproducible counts, only |00> and |11>.
    sampler = Circuit()
    sampler.add_gate("H", 0)
    sampler.add_cnot(0, 1)
    sampler.measure(0, 0)
    sampler.measure(1, 1)
    result = sampler.run_shots(StateVector.zero(2), shots=1024, seed=42)
    counts = result.counts()  # list of (bits, count), bits least-significant first
    print("1024 seeded shots:", [(tuple(bits), n) for bits, n in counts])
    assert all(bits[0] == bits[1] for bits, _ in counts)
    assert sum(n for _, n in counts) == 1024
    print("OK: Bell correlations hold exactly")


if __name__ == "__main__":
    main()
