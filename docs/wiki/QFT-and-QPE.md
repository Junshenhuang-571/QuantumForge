# QFT and phase estimation

`qft_circuit` and `qpe_circuit` are verified against independent mathematical
references. This page records the conventions they implement, because most
bugs in this area are convention bugs that leave the probabilities correct.

## Qubit ordering

Qubit `q` is bit `q` of the basis-state index: qubit 0 is the **least
significant** bit. A quick check is that `X` on qubit 0 from `|00>` produces
index 1. Every statement below assumes that convention.

## Quantum Fourier transform

For `N = 2^n`, the implemented transform maps

```
|x>  ->  1/sqrt(N) * sum_y exp(2*pi*i*x*y/N) |y>
```

up to a global phase. Two details are easy to get wrong:

1. The controlled phase is an **exact** decomposition
   `Rz_c(θ/2) · Rz_t(θ/2) · CNOT · Rz_t(−θ/2) · CNOT`. The former
   `CNOT · Rz · CNOT` sandwich was anti-controlled: it produced the right
   magnitudes and the wrong phases, i.e. the inverse DFT under a qubit
   reversal.
2. The register swap is applied **before** the Hadamard and controlled-phase
   stages. Swapping afterwards gave the wrong phase ordering for this crate's
   bit convention.

`tests/qft_qpe.rs` checks all eight three-qubit basis inputs against a DFT
computed directly from `exp(2*pi*i*x*y/N)`, allowing only a global phase. The
residual is below 1e-20.

```rust
use quantumforge::algorithms::qft_circuit;
use quantumforge::{Circuit, Gate, StateVector};

let qft = qft_circuit(2);
let mut circuit = Circuit::new();
circuit.push(Gate::X, 0); // |01>
for operation in qft.operations() {
    circuit.operations_mut().push(*operation);
}
let state = circuit.run(StateVector::zero(2).unwrap()).unwrap();
assert_eq!(state.amplitudes().len(), 4); // equal magnitudes, varying phases
```

## Phase estimation

`qpe_circuit(unitary, precision_qubits)` estimates the eigenphase φ of the
supplied unitary. For `U|ψ> = exp(2*pi*i*φ)|ψ>`, measuring the precision
register gives `m ≈ φ * 2^p`.

The circuit has three parts: Hadamards on the precision register, controlled
powers of the unitary, and the exact inverse of `qft_circuit`. The old
implementation applied the unitary's gates directly to the target register,
so the result was independent of the unitary. Every gate is now controlled by
its precision qubit.

The controlled-`Rz` decomposition places `+θ/2` on the **higher-indexed**
qubit and `−θ/2` on the lower one, with two `CNOT(high, low)` gates. Reversing
those roles makes the gate anti-controlled and doubles the estimated
eigenphase.

For `Rz(θ)|1> = exp(i*θ/2)|1>`, the eigenphase is `θ/(4π)`:

| θ | φ | p = 3 peak |
|---|---|---|
| 0° | 0 | `000` |
| 90° | 1/8 | `001` |
| 180° | 1/4 | `010` |
| 270° | 3/8 | `011` |

A phase that is not representable at the chosen precision splits across the
two neighbouring bins — for θ = 45°, the mass sits in `000` and `001`.

```rust
use quantumforge::algorithms::qpe_circuit;
use quantumforge::{Circuit, Gate, StateVector};

let mut unitary = Circuit::new();
unitary.push(Gate::Rz(std::f64::consts::FRAC_PI_2), 0);
let qpe = qpe_circuit(unitary, 3);

let mut prep = Circuit::new();
prep.push(Gate::X, 3); // target eigenstate |1>
for operation in qpe.operations() {
    prep.operations_mut().push(*operation);
}
let state = prep.run(StateVector::zero(4).unwrap()).unwrap();
let probabilities: Vec<f64> = state.amplitudes().iter().map(|a| a.norm_sqr()).collect();
let mass_001: f64 = (0..16)
    .filter(|index| index & 0b111 == 0b001)
    .map(|index| probabilities[index])
    .sum();
assert!(mass_001 > 0.99, "Rz(pi/2) eigenphase should peak at 001, got {mass_001}");
```

## Scope of the controlled-unitary support

`qpe_circuit` controls single-qubit `X`, `Y`, `Z`, `H`, `Rx`, `Ry` and `Rz`
gates. Controlled multi-qubit operations from the supplied unitary are skipped
rather than applied incorrectly, and measurement or noise operations are not
controlled. A unitary outside that surface should be expressed as
single-qubit gates before calling `qpe_circuit`.
