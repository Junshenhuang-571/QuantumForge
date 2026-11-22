# QuantumForge Python API reference

`quantumforge` is an abi3 wheel (CPython >= 3.8) built with
[maturin](https://github.com/PyO3/maturin). Everything runs locally on the CPU;
CUDA and LLVM acceleration are compile-time Rust features, not separate
packages.

```python
import quantumforge as qf
```

`qf.__all__` lists 45 names. `python/test_exports.py` asserts that this page
mentions every one of them, so a name added to the bindings without being
documented fails CI.

---

## Contents

- [Installation](#installation)
- [Exact-rational analytic core](#exact-rational-analytic-core)
- [Gradient strategies](#gradient-strategies)
- [SubsCoRe: adaptive observation cost control](#subscore-adaptive-observation-cost-control)
- [Circuits and states](#circuits-and-states)
- [Parameterized circuits](#parameterized-circuits)
- [Quantum error correction](#quantum-error-correction)
- [Algorithms](#algorithms)
- [Tensor networks](#tensor-networks)
- [Circuit optimization](#circuit-optimization)

---

## Installation

```bash
pip install quantumforge
```

Or build from a checkout (needs Rust and maturin):

```bash
maturin build --release --features python -o dist
pip install dist/*.whl
```

Verify the install:

```bash
python python/smoke_test.py
python python/test_api.py
```

---

## Exact-rational analytic core

The analytic core does not use floating point. `Rational` and `ExactComplex`
expose that arithmetic directly, so a calculation that must be exact can be
written in Python rather than approximated.

### `Rational(numerator, denominator)`

An exact rational, reduced to lowest terms with the sign on the numerator.

```python
>>> qf.Rational(1, 3) + qf.Rational(1, 6)
Rational(1, 2)
>>> qf.Rational(2, 4)
Rational(1, 2)
>>> qf.Rational(1, -2).denominator
2
>>> str(qf.Rational(3, 4))
'3/4'
```

| member | meaning |
| --- | --- |
| `numerator` | `int`, the signed numerator |
| `denominator` | `int`, always positive |
| `to_float()` | `float`, lossy for non-dyadic values |
| `is_zero()` | `bool` |
| `+ - * /`, unary `-` | exact; division by zero raises `ValueError` |

Exactness is the point:

```python
>>> big = qf.Rational(10**18 + 1, 10**18)
>>> big == qf.Rational(1, 1)
False
>>> big.to_float()
1.0
```

### `ExactComplex(real, imag)`

An exact complex number; both parts are `Rational`.

```python
>>> i = qf.ExactComplex(qf.Rational(0, 1), qf.Rational(1, 1))
>>> i * i
ExactComplex(-1, 0)
>>> (i * i).real
Rational(-1, 1)
```

| member | meaning |
| --- | --- |
| `real`, `imag` | the parts, as `Rational` |
| `to_complex()` | `(re, im)` floats |
| `is_zero()` | `bool` |
| `+ - * /`, unary `-` | exact |

### `PauliSum(qubits)`

An exact linear combination of Pauli words, used as a Hamiltonian.

```python
>>> h = qf.PauliSum(4)
>>> h.add_term("ZZII", -1, 1)     # coefficient is an exact rational
>>> h.add_term("XIII", -1, 1)
>>> h.term_count()
2
>>> h.qubits()
4
```

| member | meaning |
| --- | --- |
| `add_term(pattern, numerator, denominator)` | adds `num/den * pattern`; the pattern must be exactly `qubits` letters from `I X Y Z`, else `ValueError` |
| `qubits()` | number of qubits the sum acts on |
| `term_count()` | number of distinct words |
| `str(h)` | human-readable form, e.g. `(-1) Z2Z3 + (-1) X0` (1-indexed qubits, identities omitted) |

---

## Gradient strategies

Four exact-or-bounded rules plus an automatic chooser, mirroring
`quantumforge::gradients`. See the Rust guide `docs/wiki/Gradient-Strategies.md`
for the theory.

### `gradient_strategies()`

Returns the accepted rule names, including `"auto"`:

```python
>>> qf.gradient_strategies()
['auto', 'parameter-shift', 'adjoint', 'backprop', 'finite-difference']
```

### `ParameterizedCircuit.gradient(...)`

```python
circuit.gradient(params, qubits, observable="Z", observable_qubit=0,
                 strategy="auto", finite_difference_step=1e-3, shots=0)
```

Returns a `GradientResult`. `observable` is `"X"`, `"Y"` or `"Z"`.

`strategy="auto"` resolves from the circuit profile: a noise channel
disqualifies every state-vector rule (`ValueError`), a non-zero `shots` budget
selects `parameter-shift` (the only exact rule that survives sampling), and
otherwise `adjoint` (one backward sweep instead of two evaluations per gate).

```python
>>> circuit = qf.ParameterizedCircuit(1)
>>> circuit.rotation("Y", 0, 0)
>>> result = circuit.gradient([0.7], 1, strategy="parameter-shift")
>>> result.gradient
[-0.6442176872376912]
>>> result.strategy
'parameter-shift'
```

### `GradientResult`

| member | meaning |
| --- | --- |
| `gradient` | `list[float]`, one entry per parameter |
| `strategy` | the rule that actually ran (`auto` is resolved) |
| `evaluations` | state-vector sweeps consumed |

### `CircuitProfile` and `ParameterizedCircuit.profile(shots)`

```python
>>> profile = circuit.profile(shots=1024)
>>> profile.parameters, profile.has_channel, profile.shots
(1, False, 1024)
```

| member | meaning |
| --- | --- |
| `operations` | operations in the circuit |
| `parameters` | trainable parameters |
| `parameterized_operations` | rotation occurrences (shared parameters count per gate) |
| `has_channel` | whether a noise channel is present |
| `shots` | the declared shot budget |

---

## SubsCoRe: adaptive observation cost control

`vqe_subscore` implements arXiv:2502.01704: it minimizes `<H>` while spending a
fixed shot budget, cooling the confident-region threshold as the run progresses
so early iterations are cheap and later ones are accurate.

```python
outcome = qf.vqe_subscore(circuit, hamiltonian, params, qubits,
                          total_shot_budget=40000,
                          eta_sq=None, initial_shots=128,
                          initial_core_variance=None, core_min=None,
                          update_rule="mean", strategy="bound", seed=0)
```

| argument | meaning |
| --- | --- |
| `circuit` | a `ParameterizedCircuit` ansatz |
| `hamiltonian` | a `PauliSum` with exact rational coefficients |
| `params` | starting parameter vector |
| `qubits` | qubit count |
| `total_shot_budget` | shots available in total |
| `eta_sq` | single-shot variance; defaults to the objective's own |
| `initial_shots` | shots for the first observation |
| `initial_core_variance` | threshold to start from; defaults to `eta_sq / 512` |
| `core_min` | threshold floor; defaults to `eta_sq / 4096` |
| `update_rule` | `mean`, `thompson`, `guarded-thompson`, `free-energy`, `lower-confidence-bound` |
| `strategy` | `bound` (adaptive) or `center` (fixed) |
| `seed` | RNG seed |

### `SubsCoReOutcome`

| member | meaning |
| --- | --- |
| `best` | best parameter vector found |
| `best_value` | GP-mean energy at `best` |
| `shots_used` | total shots consumed |
| `iterations` | subspace iterations completed |
| `shots_per_iteration` | shot spend per iteration |
| `core_variances` | threshold after each iteration |

**Which update rule to use.** Measured on the 4-qubit demo instance over 12
seeds at a 40,000-shot budget: `mean` reaches -4.4222, `lower-confidence-bound`
-4.3953 (statistically indistinguishable), and the literal `free-energy`
-4.1620 with a much larger spread. Posterior sampling (`thompson`,
`guarded-thompson`) is far worse. Prefer `mean`; see
`docs/wiki/SubsCoRe.md` for the measurements and the reasoning.

---

## Circuits and states

### `Circuit`

```python
>>> circuit = qf.Circuit()
>>> circuit.add_gate("H", 0)
>>> circuit.add_cnot(0, 1)
>>> circuit.num_operations()
2
```

| member | meaning |
| --- | --- |
| `add_gate(gate, target)` | `I X Y Z H S Sdg T Tdg` |
| `add_cnot(control, target)`, `add_cz`, `add_swap` | two-qubit gates |
| `add_rx(angle, target)`, `add_ry`, `add_rz` | parameterized rotations |
| `measure(target, classical_bit)` | measurement |
| `run(state)` | state-vector simulation, returns a new `StateVector` |
| `run_shots(state, shots, seed)` | seeded sampling, returns `ShotResult` |
| `to_svg()`, `draw()` | rendering |
| `to_json()`, `Circuit.from_json(json)` | serialization |
| `num_operations()` | operation count |

### `StateVector`

`StateVector.zero(qubits)` is the all-zero state.

| member | meaning |
| --- | --- |
| `amplitudes()` | `list[complex]` |
| `probabilities()` | `list[float]` |
| `probability(index)` | one outcome's probability |
| `num_qubits()` | qubit count |

### `DensityMatrix`

`DensityMatrix.maximally_mixed(qubits)`, `DensityMatrix.from_state(state)`.

| member | meaning |
| --- | --- |
| `apply_gate(gate, target)`, `apply_cnot(control, target)` | evolution |
| `apply_channel(channel, qubit, param)` | `depolarizing`, `amplitude_damping`, `phase_damping` |
| `partial_trace(keep_qubits)` | reduced state |
| `purity()`, `num_qubits()` | diagnostics |

### `ShotResult`

| member | meaning |
| --- | --- |
| `counts()` | `list[(bitstring, count)]` |
| `shots()` | shots taken |

---

## Parameterized circuits

```python
>>> circuit = qf.ParameterizedCircuit(2)      # two trainable parameters
>>> circuit.rotation("Y", 0, 0)
>>> circuit.rotation("X", 1, 1)
>>> circuit.cnot(0, 1)
>>> circuit.expectation_z([0.3, -0.2], 2, 0)
0.9553364891256061
```

| member | meaning |
| --- | --- |
| `ParameterizedCircuit(parameter_count)` | constructor |
| `ParameterizedCircuit.with_inputs(parameter_count, input_count)` | classmethod; adds non-trainable encoded inputs |
| `rotation(axis, target, parameter)` | `axis` is `"X"`, `"Y"` or `"Z"` |
| `encode_rotation(axis, target, input)` | rotation by an encoded input |
| `cnot(control, target)` | entangler |
| `parameter_count()`, `input_count()` | counts |
| `expectation_z(params, qubits, observable_qubit)` | exact `<Z>` |
| `expectation_z_shots_with_ci(params, qubits, observable_qubit, shots, seed)` | `(value, (lo, hi))` |
| `mse_loss_and_gradient(data_x, data_y, params, qubits)` | `(loss, gradient)` |
| `parameter_shift_gradient(inputs, params, qubits)` | `list[float]` |
| `profile(shots)` | `CircuitProfile` |
| `gradient(...)` | see [Gradient strategies](#gradient-strategies) |

---

## Quantum error correction

Constructors: `repetition_code_bitflip(n)`, `repetition_code_phaseflip(n)`,
`shor_code()`, `steane_code()`, `surface_code_planar(d)`,
`surface_code_rotated(d)`.

`encode_repetition_bitflip(state, n)` encodes a `StateVector`.

### `QecCode`

| member | meaning |
| --- | --- |
| `n_physical()`, `n_logical()` | block sizes |
| `stabilizers()` | generator strings |

### `PauliString`

`PauliString(n)`, then `set(qubit, pauli)` with `pauli` as a byte (`0=I,1=X,2=Y,3=Z`), and `weight()`.

---

## Algorithms

| name | returns |
| --- | --- |
| `qft_circuit_py(num_qubits)` | `Circuit` |
| `hhl_2x2_py(a00, a01, a10, a11, b0, b1)` | `Circuit` for the 2x2 HHL algorithm |
| `hhl_solve_py(a00, a01, a10, a11, b0, b1)` | `(x0, x1)` |
| `hhl_solve_general_py(rows, b)` | `list[float]` |
| `hhl_solve_quantum_general_py(rows, b)` | `list[float]` |
| `shor_period_finding_py(n, a)` | `Circuit` |
| `shor_factor_py(n)`, `shor_factor_large_py(n)`, `shor_factor_quantum_py(n)` | `(p, q)` or `None` |

### `Grover`

`Grover(oracle_circuit, num_qubits)`, then `iterations(n)`, `build()`, `run(shots, seed)`.

### `QAOABuilder`

`QAOABuilder(num_qubits, edges)` where edges are `(i, j, weight)`; then
`layers(p)`, `parameters(betas, gammas)`, `build()`.

### `QAOAAnsatzBuilder`

`QAOAAnsatzBuilder(num_qubits, edges)` where edges are `(i, j)`; then
`layers(p)`, `build(params)`, `num_params()`.

### `HEABuilder`

`HEABuilder(num_qubits, layers)`, `entanglement(pattern)` with `linear`,
`circular`, `full` or `pairwise`, then `build(params)`, `num_params()`.

---

## Tensor networks

### `MPS`

`MPS.zero_state(n_sites, max_bond)` or
`MPS.random(n_sites, phys_dim, max_bond)`; then
`apply_single_qubit_gate(gate)`, `entanglement_entropy(cut)`, `num_sites()`.

### `Tensor`

`Tensor.new([i, j, k])` (a 3-index tensor), `Tensor.identity(d)`, `shape`.

### `TEBD`

`TEBD(mps, dt, max_bond, truncation_tol)`; then `step()`,
`find_ground_state(max_steps, energy_tol)`.

### `DMRG`

`DMRG(mps, max_bond, truncation_tol, max_sweeps)`; then `run()` returns the
ground-state energy.

---

## Circuit optimization

`OptimizeConfig.all()` and `OptimizeConfig.conservative()` return presets;
`optimize_circuit_py(circuit, config)` rewrites a `Circuit` in place and returns
the operation count removed.

---

## Stabilizer formalism

```python
>>> tableau = qf.StabilizerTableau(2)
>>> tableau.h(0)
>>> tableau.cnot(0, 1)
>>> tableau.measure(0)
(False, 0.5)
>>> tableau.equivalent(qf.StabilizerTableau(2))
False
```

`StabilizerTableau` tracks a Clifford state symbolically (Gottesman-Knill), so
it avoids the exponential cost of state-vector simulation. `measure(q)` returns
`(outcome, probability)`; the probability is `0.5` when the outcome is random
and `1.0` when it is determined. `h`, `s`, `x`, `z`, `cnot` apply Clifford
gates, `num_qubits()` reports the width, and `equivalent(other)` compares
stabilizer groups — it is a real comparison, not a width check.

## Cost-aware circuit optimization

```python
>>> circuit = qf.Circuit()
>>> circuit.add_gate("H", 0)
>>> circuit.add_gate("H", 0)
>>> circuit.add_cnot(0, 1)
>>> profile = qf.cost_profile(circuit)
>>> profile.total_gates(), profile.depth, profile.qubits
(3, 3, 2)
>>> model = qf.CircuitCostModel(cnot_gate=10.0)
>>> model.estimate(profile)
12.0
>>> optimized, result = qf.optimize_for_cost_model(circuit, qf.OptimizeConfig.all(), model)
>>> result.applied, result.original_cost, result.candidate_cost
(True, 12.0, 10.0)
```

| name | meaning |
| --- | --- |
| `cost_profile(circuit)` | `CircuitCostProfile`: gate counts, depth, width |
| `circuit_depth(circuit)` | logical depth from qubit and classical dependencies |
| `count_two_qubit(circuit)` | two-qubit gate count |
| `CircuitCostModel(...)` | 11 weights (`single_qubit_gate`, `cnot_gate`, `cz_gate`, `swap_gate`, `t_gate`, `tdg_gate`, `measurement`, `conditional_gate`, `channel`, `depth`, `qubit`); `estimate(profile)` prices a profile |
| `optimize_for_cost_model(circuit, config, model)` | `(circuit, CostAwareOptimizeResult)`; the input is never mutated and the rewrite is kept only when it strictly lowers cost |
| `CostAwareOptimizeResult` | `original_profile`, `candidate_profile`, `original_cost`, `candidate_cost`, `applied` |

## QAOA parameter transfer

```python
>>> donor = qf.MaxCutProblem.random_3_regular(8, 1)
>>> acceptor = qf.MaxCutProblem.random_3_regular(12, 2)
>>> outcome = qf.run_parameter_transfer(donor, acceptor, qf.TransferOptions(depth=4, donor_iterations=30))
>>> round(outcome.approximation_ratio, 4), round(outcome.transferred_ratio, 4)
(0.8812, 0.6797)
```

`MaxCutProblem` is built with `MaxCutProblem(vertices, edges)`, or the
generators `random_3_regular(n, seed)`, `random_er(n, p, seed)`,
`random_ba(n, m, seed)`; `with_random_weights(seed, lo, hi)` reweights a copy.

`run_parameter_transfer(donor, acceptor, options)` runs TQA initialization,
Adagrad on the donor, angle transfer, target-layer selection and a
ridge-regularized optimization of that layer.

| name | meaning |
| --- | --- |
| `TransferOptions(...)` | `depth`, `tqa_dt`, `donor_learning_rate` (default `0.05`; the paper's `0.1` stalls under finite-difference gradients), `donor_epsilon`, `donor_iterations`, `compare_full_optimization` |
| `TransferOutcome` | `donor_gammas`/`donor_betas`, `optimized_gammas`/`optimized_betas`, `target_layer`, `refinement_energies`, `energy`, `approximation_ratio` (`r_s`), `transferred_ratio` (`r_n`), `full_optimization_ratio` |

## Adiabatic quantum computing

```python
>>> h0 = [0.0, -1.0, -1.0, 0.0]   # -X
>>> h1 = [1.0, 0.0, 0.0, -1.0]    # +Z
>>> schedule = qf.AdiabaticSchedule.from_dense(1, h0, h1)
>>> schedule.problem_ground_energy()
-0.9999999999999998
>>> circuit = qf.Circuit(); circuit.add_gate("H", 0)
>>> plus = circuit.run(qf.StateVector.zero(1))
>>> result = qf.run_adiabatic(schedule, plus, 100.0)
>>> round(result.final_energy, 6)
-0.999987
```

| name | meaning |
| --- | --- |
| `AdiabaticSchedule.from_dense(qubits, h0, h1)` | row-major dense `2^n × 2^n` driver and problem Hamiltonians |
| `schedule.with_path(path)`, `schedule.path()`, `schedule.s_at(tau)`, `schedule.problem_ground_energy()`, `schedule.qubits()` | path and spectrum accessors |
| `AnnealPath.linear()`, `.fast_start(a)`, `.polynomial(p)`, `.at(tau)`, `.label()` | the annealing path `s(t)` |
| `run_adiabatic(schedule, initial, total_time, tolerance=1e-8, max_steps=100000)` | `AdiabaticResult` |
| `AdiabaticResult` | `final_state`, `initial_energy`, `final_energy`, `exact_ground_energy`, `total_time` |

Longer anneals land closer to `exact_ground_energy`: at `total_time = 10` the
example above reaches `-0.994533`, at `100` it reaches `-0.999987`.

## Quantum Turing machines

```python
>>> transitions = [
...     [qf.QtmTransition(1, 0, "stay"), qf.QtmTransition(1, 1, "stay")],
...     [qf.QtmTransition(0, 0, "stay"), qf.QtmTransition(0, 1, "stay")],
... ]
>>> machine = qf.QuantumTuringMachine(2, 2, 4, 4, transitions, 0)
>>> machine.qubits(), machine.reversibility_gap()
(7, 0)
>>> machine.step_classical(qf.QtmConfiguration([0, 0, 0, 0], 0, 0))
QtmConfiguration(head=0, state=1, tape=[0, 0, 0, 0])
```

| name | meaning |
| --- | --- |
| `QtmTransition(next_state, write, direction)` | one table entry; `direction` is `left`, `right` or `stay` |
| `QtmConfiguration(tape, head, state)` | a classical configuration |
| `QuantumTuringMachine(states, alphabet, tape_cells, head_cells, transitions, initial_state)` | `transitions[state][read]` is a `QtmTransition` or `None`; sizes must be powers of two |
| `machine.is_buildable()` | `None` when the table is a reversible bijection |
| `machine.reversibility_gap()` | reachable configurations the table does not match |
| `machine.entropy_production_per_step()`, `machine.landauer_heat_per_step(temperature)` | thermodynamics per step |
| `machine.step_classical(config)`, `machine.run_classical(config, max_steps)` | classical evolution; `step` returns `None` at halt |

## Noise and thermodynamics

```python
>>> h = qf.PauliSum(2)
>>> h.add_term("ZZ", -1, 1)
>>> h.add_term("XI", -1, 1)
>>> h.add_term("IX", -1, 1)
>>> probabilities, energy, free_energy = qf.thermal_gibbs(h, 1.0)
>>> [round(p, 6) for p in probabilities]
[0.379227, 0.120773, 0.120773, 0.379227]
>>> round(free_energy, 6)
-2.529681
>>> round(qf.shot_noise_temperature(h, 1024), 6)
0.054127
>>> round(qf.shot_noise_temperature(h, 65536), 6)
0.006766
```

`shot_noise_temperature` scales as `1/sqrt(shots)`: 64x the shots gives 8x
smaller noise, which is the identity behind SubsCoRe's thermostat.

| name | meaning |
| --- | --- |
| `thermal_gibbs(hamiltonian, beta)` | `(probabilities, <E>, F)` of the Gibbs state |
| `shot_noise_temperature(hamiltonian, shots)` | energy standard deviation of the shot estimator |
| `vqt(hamiltonian, temperature, layers, steps)` | `(F, F_exact, trace_distance)` from variational quantum thermalization |
| `dissipative_prepare(target, dt, steps)` | fidelity of jump-operator preparation of a real target vector |

## Parsing

`parse_circuit_py(expr)` parses the circuit DSL, `parse_qasm_py(qasm)` parses
OpenQASM, `export_qasm_py(circuit)` emits it, and `parse_gate_py(gate)` echoes a
gate name. All raise `ValueError` on malformed input.
