"""End-to-end tests for the `quantumforge` Python surface.

Run against an installed wheel, from a neutral directory so the checkout
cannot shadow it:

    python python/test_api.py

Covers the three areas that were missing from the bindings: gradient
strategies, the exact-rational analytic core, and SubsCoRe.
"""

import math

import quantumforge as qf

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def test_rational():
    print("Rational")
    third = qf.Rational(1, 3)
    sixth = qf.Rational(1, 6)
    check("1/3 + 1/6 == 1/2", third + sixth == qf.Rational(1, 2))
    check("reduces to lowest terms", qf.Rational(2, 4) == qf.Rational(1, 2))
    check("denominator normalised", qf.Rational(1, -2).denominator == 2)
    check("numerator keeps the sign", qf.Rational(1, -2).numerator == -1)
    check("str is p/q", str(qf.Rational(3, 4)) == "3/4", str(qf.Rational(3, 4)))
    check("to_float", abs(qf.Rational(3, 4).to_float() - 0.75) < 1e-15)

    # Exactness: the float collapses this pair, the rational does not.
    big = qf.Rational(10**18 + 1, 10**18)
    check("distinct from 1 exactly", big != qf.Rational(1, 1))
    check("float cannot tell", big.to_float() == 1.0)

    try:
        qf.Rational(1, 0)
        check("denominator 0 rejected", False)
    except Exception:
        check("denominator 0 rejected", True)

    try:
        qf.Rational(1, 2) / qf.Rational(0, 1)
        check("division by zero rejected", False)
    except Exception:
        check("division by zero rejected", True)


def test_exact_complex():
    print("ExactComplex")
    one = qf.Rational(1, 1)
    i = qf.ExactComplex(qf.Rational(0, 1), qf.Rational(1, 1))
    check("(1 + i) is not zero", not qf.ExactComplex(one, qf.Rational(0, 1)).is_zero())
    squared = i * i
    check("i^2 == -1", squared == qf.ExactComplex(qf.Rational(-1, 1), qf.Rational(0, 1)))
    check("real part exact", squared.real == qf.Rational(-1, 1))
    check("imag part exact", squared.imag == qf.Rational(0, 1))
    check("to_complex", qf.ExactComplex(one, one).to_complex() == (1.0, 1.0))


def test_pauli_sum():
    print("PauliSum")
    h = qf.PauliSum(4)
    check("starts empty", h.term_count() == 0)
    h.add_term("ZZII", -1, 1)
    h.add_term("IIZZ", -1, 1)
    h.add_term("XIII", -1, 1)
    h.add_term("IXII", -1, 1)
    check("term count", h.term_count() == 4, h.term_count())
    check("qubits", h.qubits() == 4)
    rendered = str(h)
    check("str renders each word", "Z2Z3" in rendered and "X0" in rendered, rendered)

    for pattern, label in [("ZZI", "wrong length"), ("QIII", "bad letter")]:
        try:
            qf.PauliSum(4).add_term(pattern, 1, 1)
            check(f"{label} rejected", False)
        except Exception:
            check(f"{label} rejected", True)


def test_gradients():
    print("Gradient strategies")
    names = qf.gradient_strategies()
    check("strategy list", names == ["auto", "parameter-shift", "adjoint",
                                     "backprop", "finite-difference"], names)

    theta = 0.7
    expected = -math.sin(theta)

    for strategy in ["parameter-shift", "adjoint", "backprop"]:
        circuit = qf.ParameterizedCircuit(1)
        circuit.rotation("Y", 0, 0)
        result = circuit.gradient([theta], 1, strategy=strategy)
        got = result.gradient[0]
        check(f"{strategy} matches -sin(theta)", abs(got - expected) < 1e-9,
              f"got {got}, want {expected}")
        check(f"{strategy} reports its rule", result.strategy == strategy, result.strategy)
        check(f"{strategy} accounts for sweeps", result.evaluations > 0)

    # Auto resolves to a concrete rule rather than reporting "auto".
    circuit = qf.ParameterizedCircuit(1)
    circuit.rotation("Y", 0, 0)
    auto = circuit.gradient([theta], 1)
    check("auto resolves", auto.strategy in {"parameter-shift", "adjoint", "backprop"},
          auto.strategy)

    # A declared shot budget forces parameter-shift, the only exact sampled rule.
    sampled = circuit.gradient([theta], 1, shots=1024)
    check("shots force parameter-shift", sampled.strategy == "parameter-shift",
          sampled.strategy)

    # Finite differences converge toward the exact value as the step shrinks.
    previous = None
    for step in [1e-2, 1e-4, 1e-6]:
        circuit = qf.ParameterizedCircuit(1)
        circuit.rotation("Y", 0, 0)
        got = circuit.gradient([theta], 1, strategy="finite-difference",
                               finite_difference_step=step).gradient[0]
        error = abs(got - expected)
        if previous is not None:
            check(f"fd step {step} improves", error < previous, f"{error} vs {previous}")
        previous = error

    # Auto picks a different rule once the observable changes? No: it depends on
    # the circuit, so a noise channel must disqualify every state-vector rule.
    circuit = qf.ParameterizedCircuit(1)
    circuit.rotation("Y", 0, 0)
    profile = circuit.profile(shots=0)
    check("profile counts parameters", profile.parameters == 1, profile)
    check("profile has no channel", profile.has_channel is False)
    check("profile repr", "CircuitProfile(" in repr(profile), repr(profile))

    try:
        circuit.gradient([theta], 1, strategy="nonsense")
        check("unknown strategy rejected", False)
    except Exception:
        check("unknown strategy rejected", True)


def test_subscore():
    print("SubsCoRe")
    qubits = 2
    layers = 1
    circuit = qf.ParameterizedCircuit(qubits * layers * 2)
    params = []
    for _ in range(layers):
        for qubit in range(qubits):
            circuit.rotation("Y", qubit, len(params))
            params.append(0.3)
        for qubit in range(qubits):
            circuit.rotation("X", qubit, len(params))
            params.append(-0.2)
        for qubit in range(qubits - 1):
            circuit.cnot(qubit, qubit + 1)

    h = qf.PauliSum(qubits)
    h.add_term("ZZ", -1, 1)
    h.add_term("XI", -1, 1)
    h.add_term("IX", -1, 1)

    outcome = qf.vqe_subscore(
        circuit, h, params, qubits,
        total_shot_budget=20000,
        update_rule="mean",
        strategy="bound",
        seed=7,
    )
    print(f"    best_value={outcome.best_value:.4f} shots={outcome.shots_used} "
          f"iters={outcome.iterations}")
    check("spends within budget", outcome.shots_used <= 20000, outcome.shots_used)
    check("made progress", outcome.best_value < -1.0, outcome.best_value)
    check("reports iterations", outcome.iterations > 0)
    check("per-iteration shots are within the spend",
          sum(outcome.shots_per_iteration) <= outcome.shots_used,
          f"{sum(outcome.shots_per_iteration)} vs {outcome.shots_used}")

    for rule in ["lower-confidence-bound", "free-energy", "thompson"]:
        result = qf.vqe_subscore(circuit, h, params, qubits, total_shot_budget=20000,
                                 update_rule=rule, seed=7)
        check(f"update rule {rule} runs", result.shots_used > 0, result.shots_used)

    try:
        qf.vqe_subscore(circuit, h, params, qubits, 20000, update_rule="nope")
        check("unknown update rule rejected", False)
    except Exception:
        check("unknown update rule rejected", True)


def main():
    test_rational()
    test_exact_complex()
    test_pauli_sum()
    test_gradients()
    test_subscore()
    if failures:
        print(f"\n{len(failures)} FAILED: {failures}")
        return 1
    print("\nquantumforge python API test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
