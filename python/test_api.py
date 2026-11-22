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


def test_stabilizer():
    print("StabilizerTableau")
    tableau = qf.StabilizerTableau(1)
    tableau.x(0)
    check("X|0> measures |1> deterministically", tableau.measure(0) == (True, 1.0),
          tableau.measure(0))

    bell = qf.StabilizerTableau(2)
    bell.h(0)
    bell.cnot(0, 1)
    first, probability = bell.measure(0)
    check("Bell qubit is random", probability == 0.5, probability)
    second, collapsed = bell.measure(1)
    check("measuring collapses", collapsed == 1.0, collapsed)
    check("Bell qubits agree", first == second)

    a = qf.StabilizerTableau(2)
    a.h(0)
    a.cnot(0, 1)
    b = qf.StabilizerTableau(2)
    b.h(0)
    b.cnot(0, 1)
    check("same state is equivalent", a.equivalent(b))
    check("different state is not", not a.equivalent(qf.StabilizerTableau(2)))
    check("repr", repr(a) == "StabilizerTableau(2 qubits)", repr(a))

    # measure_seeded draws a real outcome and reproduces from the seed.
    plus = qf.StabilizerTableau(1)
    plus.h(0)
    _, probability = plus.measure_seeded(0, 1)
    check("seeded measurement is random on |+>", probability == 0.5, probability)

    def draw(seed):
        tableau = qf.StabilizerTableau(1)
        tableau.h(0)
        return tableau.measure_seeded(0, seed)

    check("same seed reproduces", draw(42) == draw(42), (draw(42), draw(42)))
    ones = sum(draw(seed)[0] for seed in range(200))
    check("both outcomes occur", 40 < ones < 160, ones)

    fixed = qf.StabilizerTableau(1)
    check("deterministic qubit ignores the seed",
          [fixed.measure_seeded(0, seed) for seed in (0, 7, 99)]
          == [(False, 1.0), (False, 1.0), (False, 1.0)])

    # With real draws the Bell qubits must agree.
    bell_ones = 0
    for seed in range(100):
        tableau = qf.StabilizerTableau(2)
        tableau.h(0)
        tableau.cnot(0, 1)
        first, _ = tableau.measure_seeded(0, seed)
        second, collapsed = tableau.measure_seeded(1, seed)
        check(f"Bell qubits agree (seed {seed})", first == second and collapsed == 1.0,
              (first, second, collapsed))
        bell_ones += first
    check("Bell outcomes balanced", 20 < bell_ones < 80, bell_ones)


def test_cost_model():
    print("Cost-aware optimization")
    circuit = qf.Circuit()
    circuit.add_gate("H", 0)
    circuit.add_gate("H", 0)
    circuit.add_cnot(0, 1)
    profile = qf.cost_profile(circuit)
    check("profile counts", (profile.total_gates(), profile.depth, profile.qubits) == (3, 3, 2),
          (profile.total_gates(), profile.depth, profile.qubits))
    check("depth", qf.circuit_depth(circuit) == 3, qf.circuit_depth(circuit))
    check("two-qubit count", qf.count_two_qubit(circuit) == 1)

    model = qf.CircuitCostModel(cnot_gate=10.0)
    cost = model.estimate(profile)
    check("weighted estimate", abs(cost - 12.0) < 1e-12, cost)

    optimized, result = qf.optimize_for_cost_model(circuit, qf.OptimizeConfig.all(), model)
    check("rewrite applied", result.applied, repr(result))
    check("cost drops", result.candidate_cost < result.original_cost,
          (result.original_cost, result.candidate_cost))
    check("inverse pair removed",
          result.original_profile.total_gates() - result.candidate_profile.total_gates() == 2,
          (result.original_profile.total_gates(), result.candidate_profile.total_gates()))
    check("input not mutated", qf.cost_profile(circuit).total_gates() == 3)


def test_qaoa_transfer():
    print("QAOA parameter transfer")
    donor = qf.MaxCutProblem.random_3_regular(8, 1)
    acceptor = qf.MaxCutProblem.random_3_regular(12, 2)
    check("3-regular donor", (donor.num_qubits, len(donor.edges)) == (8, 12),
          (donor.num_qubits, len(donor.edges)))
    check("3-regular acceptor", (acceptor.num_qubits, len(acceptor.edges)) == (12, 18),
          (acceptor.num_qubits, len(acceptor.edges)))
    check("erdos-renyi", len(qf.MaxCutProblem.random_er(6, 0.5, 3).edges) == 8)
    check("barabasi-albert", len(qf.MaxCutProblem.random_ba(6, 2, 4).edges) == 9)
    check("reweighted copy", len(donor.with_random_weights(5, 0.5, 1.5).edges) == 12)

    options = qf.TransferOptions(depth=4, donor_iterations=30)
    check("defaults", (options.depth, options.donor_learning_rate) == (4, 0.05),
          (options.depth, options.donor_learning_rate))
    outcome = qf.run_parameter_transfer(donor, acceptor, options)
    check("r_s in range", 0.8 < outcome.approximation_ratio < 1.0, outcome.approximation_ratio)
    check("targeted beats transfer", outcome.approximation_ratio > outcome.transferred_ratio,
          (outcome.transferred_ratio, outcome.approximation_ratio))
    check("target layer", outcome.target_layer == 1, outcome.target_layer)
    check("sweeps recorded", len(outcome.refinement_energies) == 2,
          len(outcome.refinement_energies))
    check("repr", repr(outcome).startswith("TransferOutcome("), repr(outcome))

    try:
        qf.MaxCutProblem.random_3_regular(5, 1)
        check("odd 3-regular rejected", False)
    except ValueError:
        check("odd 3-regular rejected", True)


def test_adiabatic():
    print("Adiabatic QC")
    h0 = [0.0, -1.0, -1.0, 0.0]  # -X
    h1 = [1.0, 0.0, 0.0, -1.0]   # +Z
    schedule = qf.AdiabaticSchedule.from_dense(1, h0, h1)
    check("qubits", schedule.qubits() == 1)
    check("ground energy", abs(schedule.problem_ground_energy() + 1.0) < 1e-12,
          schedule.problem_ground_energy())
    check("linear path", schedule.s_at(0.5) == 0.5, schedule.s_at(0.5))
    check("path label", qf.AnnealPath.fast_start(0.3).label() == "fast-start(a=0.3)",
          qf.AnnealPath.fast_start(0.3).label())
    polynomial = schedule.with_path(qf.AnnealPath.polynomial(2.0))
    check("with_path", "poly(p=2)" in repr(polynomial), repr(polynomial))

    circuit = qf.Circuit()
    circuit.add_gate("H", 0)
    plus = circuit.run(qf.StateVector.zero(1))
    short = qf.run_adiabatic(schedule, plus, 10.0)
    long = qf.run_adiabatic(schedule, plus, 100.0)
    check("longer anneal is lower", long.final_energy < short.final_energy,
          (short.final_energy, long.final_energy))
    check("approaches the ground state",
          abs(long.final_energy - long.exact_ground_energy) < 1e-3,
          (long.final_energy, long.exact_ground_energy))
    check("repr", repr(long).startswith("AdiabaticResult("), repr(long))


def test_qtm():
    print("Quantum Turing machine")
    transitions = [
        [qf.QtmTransition(1, 0, "stay"), qf.QtmTransition(1, 1, "stay")],
        [qf.QtmTransition(0, 0, "stay"), qf.QtmTransition(0, 1, "stay")],
    ]
    machine = qf.QuantumTuringMachine(2, 2, 4, 4, transitions, 0)
    check("qubits", machine.qubits() == 7, machine.qubits())
    check("buildable", machine.is_buildable() is None)
    check("reversible", machine.reversibility_gap() == 0, machine.reversibility_gap())
    check("involution produces no entropy", machine.entropy_production_per_step() == 0.0,
          machine.entropy_production_per_step())
    check("no Landauer heat", machine.landauer_heat_per_step(1.0) == 0.0)

    config = qf.QtmConfiguration([0, 0, 0, 0], 0, 0)
    stepped = machine.step_classical(config)
    check("one step toggles the state", stepped.state == 1, stepped.state)
    ran = machine.run_classical(config, 5)
    check("odd step count leaves state 1", ran.state == 1, ran.state)
    check("transition repr", "direction=left" in repr(qf.QtmTransition(1, 0, "left")),
          repr(qf.QtmTransition(1, 0, "left")))
    check("config repr", repr(config).startswith("QtmConfiguration("), repr(config))

    try:
        qf.QuantumTuringMachine(2, 2, 4, 4, transitions, 5)
        check("out-of-range initial state rejected", False)
    except ValueError:
        check("out-of-range initial state rejected", True)
    try:
        qf.QtmTransition(1, 0, "sideways")
        check("unknown direction rejected", False)
    except ValueError:
        check("unknown direction rejected", True)


def test_thermodynamics():
    print("Noise and thermodynamics")
    h = qf.PauliSum(2)
    h.add_term("ZZ", -1, 1)
    h.add_term("XI", -1, 1)
    h.add_term("IX", -1, 1)

    probabilities, energy, free_energy = qf.thermal_gibbs(h, 1.0)
    check("gibbs normalised", abs(sum(probabilities) - 1.0) < 1e-12, sum(probabilities))
    check("gibbs symmetric", abs(probabilities[1] - probabilities[2]) < 1e-12)
    check("gibbs ground favoured", probabilities[0] > probabilities[1])
    check("gibbs free energy", abs(free_energy - (-2.529681)) < 1e-5, free_energy)
    check("gibbs mean energy", abs(energy - (-1.83538)) < 1e-5, energy)

    coarse = qf.shot_noise_temperature(h, 1024)
    fine = qf.shot_noise_temperature(h, 65536)
    check("noise falls with shots", fine < coarse, (coarse, fine))
    check("noise scales as 1/sqrt(N)", abs(coarse / fine - 8.0) < 1e-9, coarse / fine)

    free, exact, distance = qf.vqt(h, 1.0, 2, 50)
    check("vqt free energy", abs(free - (-2.0)) < 1e-9, free)
    check("vqt exact free energy", abs(exact - free_energy) < 1e-9, (exact, free_energy))
    check("vqt trace distance", 0.0 < distance < 1.0, distance)

    fidelity = qf.dissipative_prepare([1.0, 0.0, 0.0, 0.0], 0.05, 50)
    check("dissipative preparation", abs(fidelity - 1.0) < 1e-9, fidelity)


def main():
    test_rational()
    test_exact_complex()
    test_pauli_sum()
    test_gradients()
    test_subscore()
    test_stabilizer()
    test_cost_model()
    test_qaoa_transfer()
    test_adiabatic()
    test_qtm()
    test_thermodynamics()
    if failures:
        print(f"\n{len(failures)} FAILED: {failures}")
        return 1
    print("\nquantumforge python API test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
