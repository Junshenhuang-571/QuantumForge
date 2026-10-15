# QuantumForge

A quantum-computation kernel in Rust: an **exact-rational analytic core**, a
**CPU simulator** (state vectors + density matrices + seeded shots), an
**interactive REPL**, and optional **CUDA / LLVM** execution backends — all
sharing one set of semantics.

[![CI](https://github.com/Junshenhuang-571/QuantumForge/actions/workflows/ci.yml/badge.svg)](https://github.com/Junshenhuang-571/QuantumForge/actions/workflows/ci.yml)

QuantumForge spans **five models of quantum computation** under one roof:

| Paradigm | What it is | `qf` command |
|---|---|---|
| **Circuit / gate model** | the standard qubits-and-gates model | `add`, `run`, `draw`, `shots` |
| **Variational (VQE / QML)** | parameterized circuits optimized classically | `vqe`, `train`, `qnn`, `gradient` |
| **Measurement-based (MBQC)** | the "one-way" model — cluster states measured adaptively | `mbqc edges=.. qubits=N pattern=SPEC` |
| **Adiabatic (AQC)** | evolve a ground state from a driver to a problem Hamiltonian | `anneal NAME T=.. tol=.. steps=..` |
| **Quantum Turing machine** | reversible transition unitaries on configuration superpositions | `qtm machine=counter\|scan\|erase cells=N` |

Every paradigm command is **data-driven** — supply your own cluster graph,
problem Hamiltonian, or tape size rather than running a fixed demo. The headline
feature is the `qf` REPL: build circuits incrementally, draw them as ASCII
diagrams, run and measure them, and run variational algorithms (VQE, QML
training) with results cross-checked against exact ground-state energies.
129 tests back the semantics in the default build (incl. 23 property-based /
robustness tests and REPL end-to-end).

## Highlights

- **Exact, not floating-point guesswork.** The analytic core
  (`PauliSum` / `ExactComplex` / `Rational`) does exact rational arithmetic, so
  VQE converges to the *exact* ground-state energy to machine precision and
  gradients agree two independent ways (adjoint vs parameter-shift).
- **Interactive `qf` REPL** — live state, `draw` renders a real circuit diagram
  with vertical connectors across wires:
  ```
  q0 ───H──────•────────────────────•──────M─────────────────
  q1 ──────────⊕──────•─────────────│─────────────M──────────
  q2 ─────────────────⊕───Ry0.79────•────────────────────M───
  ```
- **Simulation**: state vectors, density matrices with noise channels
  (depolarizing / amplitude / phase damping), seeded reproducible shots, and a
  matrix-free sparse Lanczos eigensolver for ground-state energies.
- **Variational / QML**: `ParameterizedCircuit` with data encoding, adjoint +
  parameter-shift gradients (exact and finite-shot), and Adam trainers for
  regression and classification.
- **Backends**: optional `cuda` (NVRTC device-resident state vectors, density
  matrices, batched shots) and `llvm` (lower scalar expressions to LLVM IR /
  runtime-JIT), both validated against the Rust reference.

## Requirements

- Rust (edition 2024) and Cargo.
- Optional: CUDA toolkit + NVRTC (for `--features cuda`), LLVM tools
  (`opt`, `llc`) and `clang` (for `--features llvm` JIT).

## Build & test

```bash
cargo build            # default (CPU-only) build
cargo test             # 129 tests (default build; llvm/cuda tests need feature flags)
```

Run the interactive REPL:

```bash
cargo run --bin qf
```

## The `qf` REPL

Type `help` inside the REPL for the full command list. A quick tour:

```
qf> qubits 3
qf> add H[0]
qf> add CNOT[0,1]
qf> add Ry[Pi/4,2]
qf> draw                       # ASCII circuit diagram
qf> run
qf> measure                    # seeded shots -> reproducible counts
```

Core commands: `qubits`, `circuit`, `draw`, `add`, `clear`, `run`, `noisy`,
`state`, `probs`, `ptrace Q1,Q2,...` (partial trace to a subsystem), `mix`
(maximally-mixed state), `expect X|Y|Z Q`, `bloch Q`, `shots N SEED`,
`shots-noise N SEED`, plus the analytic core (`eval`, `simplify`, `diff`,
`pauli`, `assume`).

### Variational & QML commands

| Command | What it does |
|---|---|
| `gradient X\|Y\|Z Q` | Adjoint vs parameter-shift gradient, cross-validated |
| `vqe <ham>` | Variational Quantum Eigensolver — minimizes `<H>` over the circuit's rotations and compares to the exact ground state |
| `train {regression\|classify} [DATA]` | Supervised QML (Adam). `DATA` is a flat brace list `{feat...,target, ...}` |
| `qnn [DATA]` | Minimal 1-parameter `VariationalQnn`, Adam or gradient descent |

Built-in VQE Hamiltonians (exact rational coefficients): `tfim`, `ising`,
`heisenberg`, `xxz`, `hubbard`. Example:

```
qf> qubits 3
qf> add Ry[0.3,0] ... (build a Ry+CNOT ansatz)
qf> vqe hubbard
  vqe ground energy = -0.686141
  exact ground energy = -0.686141
  |vqe - exact| = 5.92e-12  ✓ converged
```

## Command-line binary

```bash
cargo run -- quantumforge demo                          # one-qubit sim + QNN training
cargo run -- quantumforge eval "2 + 3*Pi"
cargo run -- quantumforge simplify "x + x + 2 + 3"
cargo run -- quantumforge shots 2 1000 42 'Circuit[H[0], CNOT[0,1], Measure[0,0]]'
```

With feature flags: `llvm-ir` / `jit-eval` (llvm), `cuda-demo` (cuda).

## Library usage

```rust
use quantumforge::{Circuit, Gate, StateVector, parse_circuit};

// Build from a Wolfram-style expression, or the builder API.
let circuit = parse_circuit("Circuit[H[0], CNOT[0,1], Measure[0,0]]").unwrap();
let initial = StateVector::zero(2).unwrap();
let result = circuit.run_shots(&initial, 1000, 42).unwrap();
```

Key types: `Circuit` / `Gate` / `NoiseChannel`, `StateVector`, `DensityMatrix`,
`ParameterizedCircuit` (QML), `VariationalQnn`, and the analytic module's
`PauliSum` / `ExactComplex` / `Rational` / `ScalarExpr`.

## Optional features

```bash
cargo build --features llvm    # LLVM IR lowering + runtime JIT for scalar exprs
cargo build --features cuda    # CUDA/NVRTC device-resident simulation
cargo test --features cuda     # validated against the CPU reference
```

## Project layout

```
src/lib.rs            kernel entry, Circuit / StateVector / ParameterizedCircuit / VariationalQnn
src/bin/qf.rs         the interactive REPL
src/bin/main.rs       the CLI (demo/eval/simplify/shots/...)
src/analytic.rs       exact-rational core: PauliSum, ScalarExpr, Rational
src/quantum.rs        operators, observables, time-dependent evolution
src/parser.rs         Wolfram-style circuit + scalar expression parser
src/qtm.rs            Quantum Turing Machines (Bernstein–Vazirani unitary formulation)
src/mbqc.rs           Measurement-Based QC (cluster states + measurement patterns)
src/aqc.rs            Adiabatic QC (H(s) schedule driver + ground-state tracking)
src/cuda.rs           CUDA backend (feature = cuda)
src/llvm_backend.rs   LLVM IR + JIT (feature = llvm)
tests/                129 integration tests (incl. property-based + REPL end-to-end)
benches/core.rs       criterion benchmarks (state-vector, density, exact core)
RUST_CAS_ROADMAP.md   design notes and progress log
```

## CI

Every push and PR runs GitHub Actions (`.github/workflows/ci.yml`):
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo build`,
and `cargo test`. Reproduce locally:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## License

Licensed under the GNU General Public License v3.0 — see [LICENSE](LICENSE).
