# Noise and statistical mechanics

QuantumForge treats quantum noise as open-system dynamics, not as a black box.
This guide collects the five places where a statistical-mechanics model is
used, the exact mapping, and the numbers that were measured to check it. Every
number comes from a runnable example or test; nothing here is asserted without
a command that reproduces it.

## 1. The shot budget is a thermostat

An observation averaged over `N` shots has variance `eta^2 / N`. The `T` in
`exp(-f/T)` must carry energy units, so the temperature is

    T = eta / sqrt(N),

not `eta^2 / N`. Spending fewer shots is literally running hotter. This is
already in `subscore`: `UpdateRule::LowerConfidenceBound` (`argmin(<E> - T
sigma)`) is statistically indistinguishable from the plain quench, while the
literal Gibbs form `UpdateRule::FreeEnergy` (`argmin(<E> - T S)` with
`S = 1/2 ln(2 pi e var)`) is worse and less stable. The two are not the same
ordering: `argmin(mu - T f(sigma))` is invariant only under affine `f`.

See the [SubsCoRe guide](SubsCoRe.md) and `references/shot-noise-thermodynamics.md`.

## 2. Gate-noise channels: thermalisation and entropy production

`NoiseChannel` is a real open-system channel on `DensityMatrix`, with Kraus
trajectories on `StateVector`. Two statistical-mechanics statements are
implemented and measured.

### The H-theorem holds only for unital channels

`examples/noise_thermo_probe.rs` applies each built-in channel twelve times to
`|+>` and records purity, `p1`, and the von Neumann entropy `S`.

| channel | S(t) | verdict |
|---|---|---|
| depolarizing `p=0.1` | 0 -> 0.2449 -> 0.4629 -> 0.6769 (t=12), monotone up to `ln 2 = 0.6931` | unital |
| phase damping `g=0.1` | 0 -> 0.1193 -> 0.2616 -> 0.5444 (t=12), monotone | unital |
| amplitude damping `g=0.1` | 0 -> 0.1096 -> 0.2047 -> 0.2454 (t=7) -> 0.2088 (t=12) | **non-monotone** |

Amplitude damping is non-unital: it is a refrigerator pumping the qubit to
`|0><0|`, so its system entropy rises then falls. Do not call it
finite-temperature thermalisation; it is the `T = 0` limit.

### A finite-temperature channel prepares the Gibbs state

`NoiseChannel::ThermalDamping { gamma_down, beta_delta }` adds the excitation
branch, with `gamma_up = gamma_down * exp(-beta*Delta)`. Its fixed point is the
Gibbs state with `p1/p0 = exp(-beta*Delta)`, and the population approaches it
at rate `1 - gamma_up - gamma_down` per application. `examples/noise_thermo_probe.rs`
measures `p1(t)` against the prediction to nine decimals and the entropy
against the binary entropy of the Gibbs weight.

### Turn coherence times into noisy circuits

`noise_model::ThermalGateNoiseModel` applies the same bath physics to a whole
circuit. Configure `T1`, `T2`, dimensionless `beta_delta = beta * Delta`, and
single- and two-qubit gate durations (all times must use the same unit):

```rust
use quantumforge::{Circuit, Gate};
use quantumforge::noise_model::ThermalGateNoiseModel;

let mut ideal = Circuit::new();
ideal.push(Gate::H, 0);
ideal.cnot(0, 1);
let model = ThermalGateNoiseModel::new(30.0, 20.0, 1.0, 0.02, 0.2)?;
let noisy = model.apply(&ideal);
assert!(noisy.summary.thermal_channels > 0);
let rho = noisy.circuit.run_density(quantumforge::DensityMatrix::from_state(
    &quantumforge::StateVector::zero(2)?,
))?;
```

For an operation duration `dt`, the total energy-relaxation probability is
`1 - exp(-dt/T1)`. It is split into upward and downward transitions using
detailed balance, so the stationary excited-state ratio is `exp(-beta*Delta)`.
An additional phase-damping channel matches the configured `T2` after
relaxation. The constructor enforces `T2 <= 2*T1`, the Markovian two-level
bound. `apply` adds gate noise using the configured single- and two-qubit
durations. When a caller already has operation timings, `apply_scheduled` also
adds thermal noise during each qubit's idle intervals and uses the supplied
duration for each gate. `OperationTiming` values are indexed in the same order
as circuit operations and use the same time unit as `T1`/`T2`:

```rust
use quantumforge::noise_model::{OperationTiming, ThermalGateNoiseModel};
use quantumforge::{Circuit, Gate};

let mut ideal = Circuit::new();
ideal.push(Gate::H, 0);
ideal.cnot(0, 1);
let model = ThermalGateNoiseModel::for_schedule(30.0, 20.0, 1.0)?;
let noisy = model.apply_scheduled(&ideal, &[
    OperationTiming { start: 0.0, duration: 0.02 },
    OperationTiming { start: 0.10, duration: 0.20 },
])?;
assert!(noisy.summary.idle_thermal_channels > 0);
```

The model checks operation order on shared qubits and classical bits, but does
not choose a schedule or enforce hardware pulse alignment. It remains an
independent, Markovian single-qubit bath model: it does not represent correlated
errors, leakage, gate-specific coherent errors, drift, or readout error. Use the
separate readout model for measurement noise. It is a reproducible simulation
assumption, not a calibration report for any real device.

For an end-to-end optimizer report on a small unitary circuit, the CLI accepts
`--thermal T1 T2 BETA_DELTA ONE_QUBIT_TIME TWO_QUBIT_TIME`. The report compares
the optimized circuit with and without these added channels using an exact
density-matrix simulation from the all-zero state. This CLI path is limited to
8 qubits and does not use caller-supplied schedules; use `apply_scheduled` from
the library for schedule-aware idle noise.

The collisional model is the same physics from the bath side: a partial swap
`V = cos(phi) I - i sin(phi) S` against a bath in `tau` gives

    E(rho) = cos^2(phi) rho + sin^2(phi) tau + i cos(phi) sin(phi) [rho, tau],

so for diagonal states it is the convex mixture `c^2 rho + s^2 tau`. With
`beta*Delta = 1` (target `p1 = 0.268941421`) and `phi = 0.35`
(`cos^2 = 0.882421094`) the fitted per-step factor is `0.882421094` and the
coherence decays by `|c^2 + i c s (tau11 - tau00)| = 0.894888` per collision.

`DensityMatrix::von_neumann_entropy()` is the entropy used above; it returns
`QuantumError::NonHermitianDensityMatrix` for a non-Hermitian input.

## 3. Readout error is a classical channel

`readout::ConfusionMatrix` stores `P(observed | prepared)` (columns sum to one)
and mitigates by the matrix inverse, or by iterative Bayesian unfolding when a
positivity-preserving estimate is wanted. `mitigate_distribution` applies one
inverse per qubit to a full computational-basis distribution, and
`ShotResult::mitigate_readout` does it directly from shot counts.

Example: a symmetric `10%` readout on a prepared `(0.8, 0.2)` gives an observed
`(0.74, 0.26)`; the inverse recovers `(0.8, 0.2)` exactly and `unfold` recovers
it to `1e-6` while staying positive.

## 4. QEC thresholds by finite-size scaling

`qec::noise` injects Pauli data noise and flips syndrome bits, decodes with an
exact minimum-weight decoder, and counts a failure when the net Pauli is a
nontrivial logical operator. The decoder is a breadth-first search over the
`2^s` syndrome states (`s` = number of stabilizers), so it is exact and fast.

`examples/qec_threshold.rs` runs two studies:

- The bit-flip repetition family `d = 3, 5, 7, 9`. Its threshold is exactly
  `p = 1/2`; the measured crossing is `0.5150` (8 000 trials per point). This
  validates the simulator and `threshold_crossing`.
- The rotated `d = 3` surface code. With perfect readout the logical error rate
  scales as `p^2` at low `p` (`0.0015` at `p = 0.01`, `0.0302` at `p = 0.05`)
  and the pseudo-threshold `L = p` is around `0.1`. With single-round noisy
  readout the curve has a linear term and no threshold, because one measurement
  error can cause a logical failure; a real threshold study needs several
  rounds so measurement errors are corrected in time as well as space.

The model is phenomenological (data errors plus noisy readout), not a full
circuit-level simulation; the crate's `QecCode` builders do not expose per-gate
noise in the syndrome circuit.

## 5. Landauer accounting for irreversible steps

`QuantumTuringMachine::entropy_production_per_step()` computes the entropy
discarded by one classical step under a uniform input distribution:
`sum_y (n_y / Omega) ln n_y`, where `n_y` is the number of configurations that
map to `y`. `landauer_heat_per_step(temperature)` is `k_B T` times that.

The erasing table in `examples/qtm_probe.rs` discards exactly one bit per step,
so the production is exactly `ln 2` nats and the heat at `300 K` is
`2.87e-21 J` per step. A reversible table gives zero.

## Where to look

| Want | Go to |
|---|---|
| the probe with every number above | `cargo run --example noise_thermo_probe` |
| add schedule-aware idle noise to a circuit | `cargo run --example scheduled_thermal_noise` |
| the stopping-rule campaign | `cargo run --release --example subscore_entropy` |
| the threshold study | `cargo run --release --example qec_threshold` |
| exact Gibbs states and sampling | `cargo run --release --example thermal_states` |
| noise-driven optimization | `cargo run --release --example langevin_double_well` |
| dissipative state preparation | `cargo run --release --example dissipative_prep` |
| entangled dissipative preparation | `cargo run --release --example dissipative_entangled` |
| autonomous dissipative QEC | `cargo run --release --example dissipative_qec` |
| variational quantum thermalization | `cargo run --release --example vqt` |
| noise-driven VQE | `cargo run --release --example langevin_vqe` |
| readout-error modelling and mitigation | `cargo run --release --example readout_vqe` |
| the thermal channel in the REPL | `qf> channel th 0 0.2 1.0` then `qf> entropy` |
| the Gibbs state in the REPL | `qf> thermal ham=tfim beta=1.0` |
| the crate docs | <https://docs.rs/quantumforge> |

## Second pass: what the noise work needed to be honest

- **The dense Lindblad step does not scale.** `lindblad_step_code` applies
  `L = A P_-` with `A, S` Paulis in `O(jumps * 2^n)`; the dense form is
  `O(jumps * 8^n)` and is intractable at n = 9. The surface code converges in
  100 steps with the Pauli-native step.
- **A projector jump does not pump.** `(I - S)/2` only dephases; the population
  stays where it was. Pumping needs the off-diagonal `A P_-`.
- **A diagonal field cannot select a logical state.** It only moves coherences,
  which the channel damps, so the populations are identical for both field
  signs (measured). Selection needs the off-diagonal `X_L (I -+ Z_L)/2`.
- **Exact thermal states are impossible in general.** `exp` and `ln` are
  transcendental, so only a rational Boltzmann base gives exact rational
  weights. `thermal::exact_gibbs_weights` is that corner, and it says so.
- **The Langevin temperature is an energy, not a variance.** Use the shot
  estimator's standard deviation, `sqrt(sum_w |c_w|^2 / shots)`, not `eta^2/N`.
- **The FD claim is checked, not assumed.** `langevin_samples` at fixed `T`
  reproduces the double-well population ratio `exp(Delta f/T)` within 25%.
- **The collisional rate formula is for diagonal states.** A coherent start
  carries a commutator cross term and does not follow `cos^2(phi)` per step.
- **CUDA is not stubbed for the new channels.** `ThermalDamping` and the
  dissipative steps return `UnsupportedOperation`; a kernel needs nvrtc and a
  GPU, neither verifiable here.
