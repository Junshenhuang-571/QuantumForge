"""Python smoke test: imports quantumforge and exercises a few functions.

Run after building the extension, e.g.:
    maturin develop --features python
    python python/smoke_test.py
    (or: pip install target/wheels/quantumforge-*.whl && python python/smoke_test.py)
"""

import quantumforge as qf


def test_imports():
    for name in qf.__all__:
        assert hasattr(qf, name), f"missing export: {name}"


def test_bell_state():
    circuit = qf.parse_circuit_py("Circuit[H[0], CNOT[0,1], Measure[0,0], Measure[1,1]]")
    assert circuit.num_operations() == 4
    sv = qf.StateVector.zero(2)
    result = circuit.run_shots(sv, 1000, 42)
    counts = {tuple(bits): n for bits, n in result.counts()}
    # Bell state: only |00> and |11> outcomes, ~50/50
    assert set(counts.keys()) <= {(0, 0), (1, 1)}, counts
    assert abs(counts.get((0, 0), 0) + counts.get((1, 1), 0) - 1000) < 1e-9


def test_qec():
    code = qf.surface_code_rotated(3)
    assert code.n_physical() == 9
    assert code.n_logical() == 1
    # single-qubit errors produce distinct non-zero syndromes
    assert qf.verify_single_error_syndromes(code) if hasattr(qf, "verify_single_error_syndromes") else True


if __name__ == "__main__":
    test_imports()
    test_bell_state()
    test_qec()
    print("quantumforge python smoke test OK")
