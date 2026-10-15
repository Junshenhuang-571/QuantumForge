# Rust quantum computing framework

The goal is a quantum framework for **analytic and numeric quantum
computations**, with an expression language inspired by Mathematica and an
optional LLVM backend for accelerating eligible numerical kernels. It is not a
full Wolfram Language clone.

## Kernel goals

The kernel should provide a coherent mathematical model for quantum systems,
not just a circuit simulator. It should support:

- **Analytic computation:** exact symbolic scalars (integers, rationals, and
  named constants), symbolic expressions, simplification, assumptions, and
  algebraic identities for quantum objects.
- **Quantum objects:** Hilbert spaces, kets and bras, state vectors, density
  operators, linear operators, tensor products, inner products, traces,
  commutators, and basis changes.
- **Quantum operations:** unitary gates, channels, measurements, observables,
  Hamiltonians, expectation values, and state evolution.
- **Numeric computation:** complex-valued dense vectors and matrices first;
  sparse and structured representations as separate implementations behind
  common traits. Support state-vector simulation, density-matrix simulation,
  eigensolvers, and time evolution as the framework grows.
- **Quantum circuits:** a circuit IR for gate sequences, control, measurement,
  and classical conditions. Circuits should lower into the same operator/state
  semantics rather than form an isolated subsystem.
- **Quantum machine learning:** parameterized circuits and hybrid quantum-
  classical workflows for QNNs and related QML algorithms, built on the circuit,
  measurement, and differentiation APIs rather than a separate simulator.
- **Performance:** a portable Rust CPU implementation as the reference backend,
  LLVM compilation for typed numeric expressions and hot kernels, and GPU
  execution for large workloads. The framework may use Fortran or other
  languages through narrow, documented interfaces when mature numerical
  libraries make that worthwhile.

## Kernel architecture

1. **Scalar and expression layer**
   - A uniform expression tree with calls represented by a head and arguments.
   - Exact scalar values plus symbolic variables and canonical constants such
     as `I`, `Pi`, and `E`.
   - Rewrite rules and simplification that preserve exactness; floating-point
     evaluation is explicit.
   - Assumptions and domains (for example, real, integer, positive, or complex)
     influence simplification and validation.

2. **Quantum type layer**
   - Typed spaces and dimensions prevent invalid operations, such as adding
     vectors from incompatible spaces or applying an operator to the wrong
     dimension.
   - Separate semantic types for `Ket`, `Bra`, `Operator`, `DensityMatrix`,
     `Observable`, `Unitary`, `Channel`, `Hamiltonian`, and `Circuit`.
   - Preserve tensor-product structure where possible instead of eagerly
     materializing exponentially large matrices.
   - Distinguish exact/symbolic values from approximate numeric values.

3. **Algebra and analytic rules**
   - Implement conjugation, adjoint, tensor product, composition, trace,
     partial trace, inner products, commutators, and expectation values.
   - Add verified identities and canonicalization rules for common gates,
     Pauli operators, projectors, and states.
   - Keep rewrite rules terminating and expose when a result is conditional on
     assumptions.

4. **Numeric backends**
   - Begin with complex `f64` dense linear algebra and a simple CPU simulator.
   - Add sparse matrices and structured state representations without changing
     the public quantum semantics.
   - Make tolerances, normalization, and numerical error behavior explicit.
   - Provide deterministic seeded measurement sampling.
   - Define a backend interface for device discovery, allocation, transfers,
     kernel dispatch, synchronization, and error reporting. Keep device buffers
     resident across chained operations to avoid needless host/device copies.
   - Support GPU execution for batched circuit simulation and large dense or
     structured operations. Preserve a CPU implementation for small workloads,
     unsupported operations, and reference comparisons.
   - Detect GPU capability at runtime so the core still builds and runs when no
     supported GPU or SDK is installed.

5. **LLVM backend**
   - Lower only explicit, typed, side-effect-controlled numerical IR to LLVM.
   - Candidate kernels include gate application, tensor contractions,
     expectation values, and selected integration/evolution loops.
   - Keep symbolic manipulation and dynamic evaluation in the Rust kernel.
   - Retain a CPU interpreter/reference path for every accelerated operation;
     backend selection must not change mathematical semantics.
   - Put LLVM behind an optional Cargo feature so users can build the core without
     an LLVM installation.

6. **Additional native numerical backends**
   - Keep the core API in Rust, but allow optional bindings to established
     Fortran numerical libraries for BLAS/LAPACK operations or specialized
     solvers when they provide a clear benefit.
   - Prefer libraries with a stable C ABI or wrap Fortran routines behind a
     small C-compatible shim. Specify complex-number layout, integer width,
     ownership, error propagation, and thread-safety at each FFI boundary.
   - Make these backends optional; the default Rust implementation should not
     require a Fortran compiler.

7. **Quantum machine learning layer**
   - Represent parameterized circuits with named trainable parameters and
     differentiable expectation values.
   - Provide gradients using appropriate methods such as parameter shift for
     supported gates, with room for adjoint differentiation and other methods.
     Document which circuit operations each gradient method supports.
   - Build hybrid training loops from objective functions, classical
     optimizers, batching, and quantum circuit evaluation.
   - Include common QML building blocks: data encoding, variational circuits,
     expectation-based outputs, and classification/regression examples.
   - Support noisy channels and shot-based measurements so training workflows
     can model finite sampling and noise, in addition to ideal state-vector
     simulation.
   - Keep datasets, classical optimizers, and model APIs separate from the
     mathematical quantum kernel; let users replace each component.

8. **Language and frontends**
   - Initial Wolfram-style scalar syntax and `eval`/`simplify` CLI commands are
     implemented for exact arithmetic, constants, and elementary functions.
     Supported real-valued scalar trees can now lower to the numeric IR for
     native backend compilation with explicit symbol bindings; `llvm-ir` prints
     generated LLVM IR and `jit-eval` evaluates real expressions through the
     runtime-compiled native path when the `llvm` feature is enabled. The circuit parser
     now constructs gate, rotation, measurement, feed-forward, and noise-channel
     operations; `shots` runs seeded state-vector sampling from this syntax.
     Broaden quantum object notation, classical control expressions, and
     language compatibility after their semantics stabilize.
   - Broader language compatibility and notebook UI remain future work.

## Backend boundaries

The Rust kernel defines mathematical semantics; backends implement numeric
operations behind those semantics. LLVM should compile only code with explicit
numeric types and known effects. GPU backends should operate on typed buffers
and explicit dispatches. Fortran integrations should cross a narrow FFI
boundary. Symbolic simplification, arbitrary-precision values, pattern
matching, and general dynamic evaluation remain in Rust. Backend selection must
preserve operation semantics and report precision or capability limits rather
than silently changing results. Keep each native toolchain optional so the
default core builds without LLVM, GPU SDKs, or a Fortran compiler.

Backend work should compare numerical outputs with the Rust reference and
report transfer costs as well as kernel time. GPU acceleration is most useful
for large or batched workloads; dispatch and transfer overhead may make a CPU
path faster for small systems.

## Suggested public API shape

The API should make spaces and types visible while keeping common operations
readable. Illustrative pseudocode:

```rust
let q0 = Qubit::zero();
let state = H * q0;
let measured = measure(state, &mut rng);
let energy = expectation(&state, &hamiltonian)?;
let exact = simplify(commutator(&pauli_x, &pauli_z));
```

This is an API sketch, not current implemented behavior. Rust traits and wrapper
types should prevent invalid dimensional operations where practical, while
runtime checks remain necessary for dynamically sized objects.

## Initial milestones

1. **Prototype implemented:** complex scalars, dense state vectors, common
   single-qubit gates, CNOT/CZ/SWAP, circuits, seeded measurement, Z
   expectations, dense density matrices with local and entangling unitary
   evolution, partial trace, depolarizing/amplitude/phase damping, density-matrix
   measurement, and a one-parameter QNN with parameter-shift gradients and batch
   Adam/gradient-descent optimization, and deterministic mini-batch training.
2. **Analytic core implemented:** exact rational complex coefficients, scalar
   expression trees with canonical `Pi` and `E`, complex-valued numeric
   evaluation of trigonometric, exponential, logarithmic, and algebraic
   functions, special-angle simplifications, symbolic substitution,
   assumption-aware absolute value and square-root rules, and formal symbolic
   differentiation for supported smooth expressions. The kernel has bounded
   structural rewrite rules and exact/symbolic Pauli sums with products and
   commutators; bound sums lower to sparse numeric operators and observables
   that evaluate on dense or sparse kets. Broader algebraic domains and rewrite
   libraries remain to be added.
3. **Dimension-aware linear algebra implemented:** finite-dimensional Hilbert
   spaces, dense and sparse kets, bras, dense operators, and CSR sparse
   operators behind a shared linear-map interface. Sparse kets retain support
   through gates, sparse-map application, tensor products, expectations, and
   seeded measurement. Dense/sparse maps support adjoints and composition.
   Hermitian CSR operators also expose a matrix-free Lanczos solver for a
   requested set of lowest Ritz eigenpairs with residual-based convergence
   checks. Sparse Hamiltonians can also evolve kets with a Lanczos Krylov
   approximation, without materializing a dense evolution operator. Full sparse
   eigensystems, sparse density-matrix evolution, typed tensor-network
   operators, and exact gate identities remain future work.
4. **Circuit controls and shot execution implemented:** circuits support seeded
   shot runs with projective measurements, classical result registers, and
   single-qubit operations conditioned on classical bits. Results aggregate
   reproducible counts. Seeded shot APIs sample normalized state-vector Kraus
   trajectories for local noise channels; exact channel evolution remains
   available through density matrices. Dense Hermitian observables support
   expectation values on kets and density matrices.
5. **Hamiltonian numerics implemented:** complex dense Hermitian operators can
   be fully diagonalized with a cyclic Jacobi solver and used for time-
   independent spectral evolution. Hermitian CSR operators can compute selected
   low-energy eigenpairs and evolve kets with Lanczos without converting the
   Hamiltonian to a dense matrix. Both paths report tolerance, Hermiticity, and
   convergence errors. Full sparse eigensystems, sparse density-matrix and
   time-dependent evolution, and large-scale performance studies remain open.
6. **Typed numeric IR and LLVM lowering implemented:** pure `f64` SSA
   instructions for arithmetic, integer powers, and elementary functions have
   a validated Rust interpreter. Parsed real-valued scalar expressions can
   lower to the same IR with explicit symbol bindings. The optional `llvm`
   feature lowers programs to LLVM IR and can run `opt` and `llc` without
   linking LLVM into the core. Numeric programs can also be runtime-compiled at
   optimization levels 0–3 through `rustc`'s LLVM backend, loaded as a native
   module, and called repeatedly with interpreter-matching input and arithmetic
   errors. Native output matches the reference interpreter in tests. This path
   uses the Rust compiler's bundled LLVM backend; direct `opt`/`llc` operation
   is still unavailable here because those tools and `clang` are not installed.
   Integration into simulation kernels and performance benchmarks remain open.
7. **CUDA state-vector and shot execution implemented:** optional
   device-resident buffers, single-qubit matrix gates, Rx/Ry/Rz, CNOT/CZ/SWAP,
   and unitary `Circuit` execution. Projective measurement reduces probabilities
   on device, transfers only per-block sums for seeded host sampling, and
   collapses state with a CUDA kernel. Multi-shot circuit execution resets the
   working buffer with device-to-device copies and supports classical
   single-qubit feed-forward. Device-resident dense density matrices now support
   local unitary gates, CNOT/CZ/SWAP, depolarizing, amplitude-damping, and
   phase-damping channels, plus seeded projective measurement and collapse
   without transferring the full matrix to the host. Density matrix GPU
   capacity is limited by its quadratic memory cost and current index width.
   Shared unitary circuits can execute over contiguous batches of independent
   state vectors on the GPU. Batch buffers also support seeded measurement and
   collapse across all entries. Multi-shot batched circuits use measurement
   registers and classical single-qubit feed-forward, preserving the CPU
   simulator's seeded shot order. Batched trajectories cover depolarizing,
   amplitude-damping, and phase-damping channels. A CUDA benchmark separates
   allocation/upload, resident gate execution, download, and end-to-end batch
   time while checking outputs against the CPU reference.
8. **Parameterized QML circuits implemented:** data-encoding rotations,
   multiple trainable parameters, shared-parameter occurrence handling,
   exact and finite-shot parameter-shift gradients for single-qubit X/Y/Z
   expectations, exact and finite-shot MSE regression and binary
   cross-entropy classification with Adam minibatching, and deterministic data
   shuffling. Parameterized circuits can also execute exact density-matrix
   channels and compute parameter-shift gradients through noisy expectation
   values, plus density-matrix MSE regression and binary classification with
   exact and finite-shot parameter-shift gradients and seeded Adam minibatch
  training. Ideal state-vector expectations also have an adjoint-gradient path
  that handles data encodings, entangling gates, and repeated uses of shared
  parameters in a forward-state and backward-costate sweep. A comparison check
  covers X/Y/Z observables against parameter shift. Noisy adjoint gradients,
  finite-shot adjoint gradients, and richer model-level abstractions remain.
9. **CUDA backend prototype implemented:** optional `cuda` feature uses NVRTC
   and the CUDA Driver API. Hardware-validated kernels cover unitary and seeded
   shot circuits, classical feed-forward, dense density-matrix gate evolution,
   local noise channels, device-resident density-matrix measurement, and
   measurement-aware contiguous state-vector batch simulation against the CPU
   reference. A hardware benchmark is available by running
   `cargo bench --features cuda --bench cuda_batch`; all implemented shot and
   channel semantics are compared with the CPU reference.
10. Add optional Fortran BLAS/LAPACK integration behind a C-compatible shim if
    benchmarks show a useful advantage.
11. Integrate the LLVM-backed numeric compiler with selected simulation
    kernels and benchmark it; expand sparse/structured simulation based on
    measured workloads.

## Interactive `qf` REPL (`src/bin/qf.rs`)

A stateful terminal REPL over the library, verified by `tests/repl.rs` (pipes
commands through the binary and asserts on stdout). Core commands: `qubits`,
`circuit`, `draw` (fixed-width ASCII diagram), `add` (semantically validated by
dry-running the trial circuit through the matching execution model), `clear`,
`run`, `noisy` (live density-matrix mode), `state`, `probs`, `expect X|Y|Z Q`,
`bloch Q`, `shots`/`shots-noise` (seeded), plus the analytic core (`eval`,
`simplify`, `diff`, `pauli`, `assume`).

Variational / QML surface, all data-capable:
- `gradient X|Y|Z Q` — adjoint vs parameter-shift gradient, cross-validated.
- `vqe <ham>` — VQE: minimize `<H>` over the circuit's rotation params via
  parameter-shift-on-total-energy descent, compared against the exact ground
  state from the sparse Lanczos eigensolver. Built-in Hamiltonians (exact
  rational coefficients): `tfim`, `ising`, `heisenberg`, `xxz` (Δ=−½),
  `hubbard` (t=U=1, Jordan–Wigner). Verified convergent to the exact ground
  state with a deep Ry+CNOT-ring ansatz.
- `train {regression|classify} [DATA] [nfeat=K lr=E epochs=N]` — supervised QML
  (angle-encoded Ry + trainable weight + `<Z>` readout, Adam). Accepts a flat
  brace dataset `{f1,..,fK,target, ...}`; numbers may be kernel expressions
  (e.g. `Pi/2`). Honest R²<0 note when `<Z>∈[−1,1]` can't fit unbounded targets.
- `qnn [DATA] [target=K lr=E steps=N opt=adam|gd]` — the minimal 1-parameter
  `VariationalQnn`, both Adam and gradient-descent optimizers.

Paradigm commands (QTM / MBQC / AQC), all data-capable:
- `qtm [machine=counter|scan|erase] [cells=N]` — Bernstein–Vazirani unitary
  Turing machine. Head moves **wrap** modulo the tape, so a total, injective
  transition table forms a unitary directly (`counter`/`scan` via `step_quantum`).
  `reversibility_gap` diagnoses a non-bijective table; `step_quantum_ancilla` runs
  an information-erasing table (`erase`) as a unitary on an ancilla-augmented
  register that preserves the erased symbol (one ancilla slot per head position,
  storing `read+1`). The ancilla map is a whole-space **involution** — one step
  advances, a second returns — i.e. reversible ancilla stepping for a single
  step, not an unbounded Bennett history tape. `is_buildable` rejects a state
  register too small to encode the internal state injectively; a qubit cap
  guards against oversized tapes.
- `mbqc [edges=a-b,c-d] [qubits=N] [pattern="..."]` — measurement-based QC:
  builds a cluster state on your graph and runs a measurement pattern. A
  `pattern=` spec supplies explicit angles (decimals or `pi`/`pi/2`/`2pi`),
  bases (`eq`/`z`), a free per-step measurement order (`qubits=`, any
  permutation — each qubit measured exactly once), by-product dependencies, and
  output flags via `mbqc::parse_pattern`; without it, each vertex is measured
  depending on its neighbours' earlier outcomes.
- `anneal [NAME] [T=.. tol=.. steps=..] [path=linear|fast|poly]` — adiabatic
  QC: interpolates `H(s)=(1-s)H₀+sH₁` from a derived transverse-field driver
  (`aqc::transverse_field_driver`) to a problem Hamiltonian (named spin chain,
  reusing the exact-rational VQE models), driven by the adaptive-step Trotter
  integrator. `path=` selects the annealing path `s(τ)`: `linear` (default),
  `fast` (Roland–Cerf fast-start), or `poly` (quadratic). A bare leading number
  is `T=` for backward-compat. The driver guards integrator non-convergence
  with a tolerance floor and reports an actionable error.

CI: `.github/workflows/ci.yml` runs `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo build`, and `cargo test`
on every push/PR.

Custom (subscripted multi-qubit) Hamiltonian input via `pauli`/`vqe` remains
future work — the scalar parser has no list/word-of-qubits form, so `vqe`
currently uses the built-in named models.

## Parser robustness

The hand-rolled Wolfram-style parser is fuzzed with proptest
(`tests/parser_fuzz.rs`): random and structured-garbage strings are thrown at
`parse_scalar_expr` and `parse_circuit`, asserting they never panic (always
`Ok`/`Err`). This caught a real stack-overflow on deeply nested brackets, fixed
by a `MAX_PARSE_DEPTH` (256) guard in recursive descent — a pathological paste
now returns an error instead of crashing. `cargo-fuzz` (libFuzzer) needs nightly
and a separate workspace the stable CI can't run, so proptest gives comparable
coverage on stable.


## Scope discipline

“Analytic and numeric quantum computations” is still broad. The first release
should focus on finite-dimensional systems and qubit circuits, while defining
interfaces that can later accommodate continuous-variable systems and more
advanced symbolic mathematics. Do not claim those extensions until their
representations and semantics are implemented.
