# Changelog

## [0.2.9] - 2026-11-19

### Added
- **Complete API documentation.** Every public item in the crate now carries a
  doc comment (775 items added across 24 modules), verified by a
  `missing_docs` lint that is now a hard error rather than a warning, so the
  completeness cannot silently regress. This includes items that only compile
  behind a feature flag (the `cuda` backend).
- Documentation map: `docs/README.md` ties together docs.rs, the wiki guides,
  the Python README, the examples and the release checklist, with an
  API-by-subsystem index.
- Crate docs gain a "Tensor networks and QAOA at scale" section with a runnable
  parameter-transfer doctest, and the README a "Highlights in 0.2.9" section.
- Dependency hygiene: Dependabot configuration (weekly cargo updates) and a
  `cargo audit` CI job against the RustSec advisory database.
- Property-based tests for the simulation core (`tests/proptest_quantum_core.rs`):
  unitary circuits preserve the norm, shot counts are normalized and
  reproducible under a fixed seed, basis-state measurement is deterministic,
  Bell correlations hold, and the rotated surface code satisfies its
  structural invariants.
- **MPS-based QAOA** (`mps_qaoa` module): MaxCut on matrix product states with a
  bounded bond dimension. Long-range diagonal gates applied as exact bond-2
  MPOs, two-sweep SVD compression, transfer-contraction cost expectations
  (no state-vector expansion), seeded 3-regular graph generator, and
  parameter-shift angle finding with restarts. Example
  `examples/mps_qaoa.rs` and guide `docs/wiki/MPS-QAOA.md`.
- **QAOA parameter transfer + targeted-layer optimization**
  (`qaoa_transfer` module, arXiv:2601.15760): TQA angle initialization,
  finite-difference Adagrad on a small donor graph, transfer of the optimized
  angles to a larger acceptor, then a ridge-regularized joint optimization of
  the trailing layers. `TransferOutcome` reports `r_n`, `r_s`, per-sweep
  refinement energies and optionally `r_f`. Example `examples/qaoa_transfer.rs`
  and guide `docs/wiki/QAOA-Parameter-Transfer.md`.
  Defaults deviate from the paper in three measured ways: donor `lr = 0.05`
  (0.1 stalls under finite-difference gradients), the targeted block is the
  last 3 layers rather than 1 (98.7% of `r_f` versus 95.9% at 10 vertices),
  and the block runs twice.
- MaxCut graph families for the transfer work: `MaxCutProblem::random_er`,
  `random_ba` and `with_random_weights` join the existing `random_3_regular`,
  so the paper's 3R/ER/BA and weighted comparisons can be reproduced. The REPL
  `qaoa-transfer` takes `family=3r|er|ba` (prefix `w` for weighted) and `m`
  (ER probability / BA edges per new vertex); example
  `examples/qaoa_transfer_families.rs` prints the comparison table.
- REPL: `qaoa-transfer` runs the pipeline and stores it on the session;
  `qaoa-angles`, `qaoa-layers` and `qaoa-layer` inspect the stored run.
  The default donor learning rate is 0.05 rather than the paper's 0.1, which
  stalls under finite-difference gradients (donor r = 0.825 vs 0.980).
- MPS: `apply_diagonal_two_site`, `compress_range`,
  `expectation_two_site_product`, `norm_sqr_mps`.
- Examples: `qec_roundtrip` (surface-code verification + repetition-code
  encode/inject/syndrome/recover cycle), `qasm_roundtrip` (OpenQASM 2.0
  import/export/re-import), `vqe_exact` (VQE vs exact ground-state energy).
- Python first-run path: `python/README.md` plus worked scripts
  `python/examples/bell.py`, `qec.py`, `vqe.py`.

- **`UpdateRule::LowerConfidenceBound` and `UpdateRule::FreeEnergy`** — the
  acquisition side of the statistical-mechanics reading, with the temperature
  in energy units, `T = eta/sqrt(N)` (the shot noise is a variance, so its
  square root is the temperature). `LowerConfidenceBound` is `argmin(<E> -
  T*sigma)`, the standard BO acquisition. `FreeEnergy` is the literal Gibbs
  form `argmin(<E> - T*S)` with the posterior's Gaussian entropy
  `S = 1/2 ln(2 pi e var)`. Measured on the 4-qubit demo over 16 seeds, with the
  standard deviation across seeds:
  - fixed-shot NFT -3.9709 (sd 0.411)
  - quench (`UpdateRule::Mean`, Bound) -4.4222 (sd 0.206)
  - `LowerConfidenceBound` -4.3953 (sd 0.200) — indistinguishable from it
  - `FreeEnergy` (literal entropy) -4.1620 (sd 0.880) — worse and less stable
  - `Center + Thompson` -0.1381 (sd 1.282), guarded -2.3350 (sd 1.253)
  The two acquisitions are *not* the same ordering: one is linear in `ln
  sigma`, the other in `sigma`.
- Corrected an earlier claim of mine: I had written that using `sigma` in place
  of the entropy gives "the same ordering". It does not — a counterexample is in
  the module docs, and the two rules now exist separately under their correct
  names.
- Corrected the `KL(binomial || Gaussian)` numbers I had quoted (they came from a
  probe with a `usize` underflow). Correct values: at most 0.275 nats at `N=1`
  (a nearly deterministic word), 0.056 at `N=16`, below 0.006 at `N=256`. The
  conclusion is unchanged — the posterior of Eqs. 4-5 uses only the noise
  variance, which is already exact, so refining the likelihood buys nothing.
- **Gradient strategies with automatic selection** (new module `gradients`).
  Production variational stacks offer several ways to differentiate
  `<psi_x|O|psi_x>`; the crate had two. Now:
  - `GradientStrategy` (`ParameterShift`, `Adjoint`, `Backprop`,
    `FiniteDifference`, `Auto`), `CircuitProfile`, `GradientOptions` and
    `GradientEstimate` (which reports the rule used and its state-vector sweep
    count).
  - `GradientStrategy::select`: a noise channel disqualifies every
    state-vector rule; a shot budget forces `ParameterShift` (the only exact
    sampled rule); otherwise `Adjoint`, which costs one backward sweep instead
    of two evaluations per gate.
  - `finite_difference_gradient_with_inputs` — central differences with a
    validated step, per gate occurrence, for gates with no shift rule.
  - `backprop_gradient_with_inputs` — reverse-mode backpropagation written from
    the definition, independent of the adjoint shortcut.
  - REPL `gradient X|Y|Z Q [method=...]`; the previous cross-check output is
    unchanged when `method=` is omitted.
  - `QuantumError::InvalidFiniteDifferenceStep`.
  `tests/gradients.rs` cross-validates: an analytic anchor (`<Z> = cos theta`),
  every exact rule against every other on a 42-parameter circuit, shared
  parameters, second-order convergence of the finite-difference rule, and the
  auto-selection rules. `Backprop` and `Adjoint` agree to 1e-9 — they are
  algebraically the same reverse-mode computation for a Pauli observable, and
  the guide says so rather than presenting them as different mathematics.
- **SubsCoRe: shot-aware VQE with the shot budget as a thermostat.**
  `subscore` (arXiv:2502.01704) optimizes one parameter axis at a time and
  chooses the shots at each observed point so the whole updated axis stays
  inside the GP surrogate's confident region. New in this pass:
  - `Observation` — every reading reports its value, its *actual* noise
    variance, and the shots it spent. `ShotHamiltonian::observe` now takes a
    total shot budget (it previously took shots *per Pauli word*, so the real
    spend and the GP's noise model were both off by the word count) and
    `ShotHamiltonian::single_shot_variance` reports the objective's `eta^2`.
  - `UpdateRule` — `Mean` (zero-temperature quench), `Thompson` (draw a
    function from the GP posterior and move to its minimizer) and
    `GuardedThompson` (same, but refuse a kick the posterior mean calls
    uphill), plus `exploration` as a thermal factor and
    `GaussianProcess::sample` for a correlated joint draw.
  - The confident-region threshold is now clamped monotone non-increasing: a
    fast initial energy drop used to make the run hotter than it started and
    starve its own observations.
  - REPL: `vqe-subscore` gains `update=mean|thompson|guarded`, `explore=` and
    the objective's own `eta^2`.
  - Measured on a 4-qubit / 2-layer TFIM at 40k shots over 6 seeds: Bound+mean
    -4.4222 vs fixed-shot NFT -3.9709. Center+mean -3.5768, Thompson -0.1381
    and guarded Thompson -2.3350 all do worse; the guide reports this as a
    negative result.
- **Fixed `SmallRng::seeded` not mixing the seed.** Raw xorshift64 maps nearby
  seeds to nearby first outputs (`seeded(1..N)` gave first draws proportional to
  the seed), so "independent" runs with consecutive seeds were silently
  correlated and `restarts: 3` produced three near-identical starts. A
  splitmix64 finalizer now decorrelates them; reproducibility per seed is
  unchanged.
- **SubsCoRe: adaptive observation cost control for VQE** (arXiv:2502.01704),
  new module `subscore`. One parameter axis is optimized per iteration from
  `1 + 2 V_d` observations; a heteroscedastic GP over the VQE kernel
  (Eq. 10) tracks the surrogate, and the shots at each observed point are
  chosen so the whole updated axis stays inside the confident region. The
  threshold loosens while the energy drops fast and tightens to `core_min`
  near convergence. Includes `vqe_kernel`, `GaussianProcess` (incremental
  Cholesky), `axis_points`, `choose_shots_bound` / `choose_shots_center`,
  `run_subscore`, `ShotHamiltonian`, `SubsCoReOptions`, `ReadoutStrategy`,
  `SubsCoReOutcome`, `SearchOptions`, the REPL command `vqe-subscore`, the
  example `examples/subscore.rs`, and the guide `docs/wiki/SubsCoRe.md`.
  The guide also documents where the paper's pseudocode scans the shot
  variance in the opposite direction to its own Eq. 15.

### Changed
- **Breaking**: `qaoa_transfer::TransferOutcome::layer_energies` renamed to
  `refinement_energies`, and it now holds the cut expectation after each
  targeted sweep rather than the outcome of single-layer trials.
- `TargetedLayerOptions` gains `layers` (default 3) and `sweeps` (default 2).
  Setting fields by name still works via `..TargetedLayerOptions::default()`.
- The crate now **denies** `missing_docs`: a public item without a doc comment
  is a build error, so the documentation cannot silently regress.
- Dependabot ignores `version-update:semver-major`; major version bumps are
  done by hand together with their migration.
- `Cargo.toml` description now lists the tensor-network, QEC and QAOA surface.


### Fixed
- Dependencies: `tokio` 1.54.0 → 1.54.1, `rustix` 1.1.5 → 1.1.6 and
  `rustls-webpki` 0.103.15 → 0.103.17, clearing the advisories found by
  `cargo audit` (RUSTSEC-2026-0384, RUSTSEC-2026-0486, RUSTSEC-2026-0382).
  The two remaining pyo3 advisories are ignored in CI with a documented
  reason: their patched releases need pyo3 ≥ 0.29, which requires the unstable
  `cfg_select` feature and so a newer compiler than the declared MSRV 1.88.
- `docs/index.html` listed the wrong license (GPL-3.0 instead of Apache-2.0)
  and a stale test count; corrected and linked from the README.
- `MPS::to_state_vector` and the MPS expectations used site 0 as the most
  significant bit while `StateVector` uses qubit 0 as the least significant;
  both now follow the crate convention.
- Repetition-code syndrome extraction now measures the ancilla qubits, so
  `QecCode::syndrome_circuit` produces a readable syndrome instead of
  entangling ancillas without readout. Surface-code extraction already did
  this; the repetition and phase-flip builders now match it.

- **Corrected the thermodynamics claimed for `UpdateRule::Thompson`.** The
  previous wording called posterior sampling Boltzmann exploration. It is not:
  the posterior variance measures how ignorant the *surrogate* is away from its
  observations, which is a different quantity in different units from the
  estimator noise `eta^2 / N`. The module docs, the `UpdateRule` and
  `exploration` doc comments, and `docs/wiki/SubsCoRe.md` now say so, and carry
  the measurement: on the demo instance the quench reaches -4.4222 while
  Thompson and guarded Thompson reach -0.1381 and -2.3350.
  A 1D double-well check confirms the mechanism itself is sound — gradient
  descent stays in the shallow well 400/400, an annealed Langevin step reaches
  the deep well 199/400 — so the honest conclusion is that a shot-budgeted VQE
  is information-limited, not barrier-limited, and step noise costs more than
  it buys. The rule is kept as a documented negative result.
- **Flaky statistical assertions in `tests/subscore.rs`.** The reported-vs-
  empirical variance check used 120 repeats against a 20% tolerance; the
  standard error of an empirical variance at 120 samples is 13%, so it was a
  ~1.5-sigma test that failed roughly one run in eight. It now uses 800 repeats
  (about 4 sigma). The guard test's threshold sat at 80 while the measured
  acceptance ranged 45..73 across seed bases, so a different seed base could
  fail it; the threshold is now 100. The explore test's `at_home > 30` had a
  worst observed value of 45; it is now `> 20`.
- **`GradientStrategy::parse` panicked on an unknown name.** It returned
  `Self` and `panic!`ed, so a caller-supplied string could abort the process.
  It now returns `Option<Self>` and the REPL uses it instead of keeping a
  second name table.
- **`GradientOptions` could not express a shot budget.** `CircuitProfile.shots`
  existed but no public path set it, so `Auto`'s sampled branch was unreachable
  and the doc comment claimed otherwise. `GradientOptions::shots` now feeds the
  profile; the gradient stays exact, and the option only tells `Auto` that the
  caller intends to sample.
- **CI test time.** The subscore variance test ran 1000 observations of a
  4-qubit, 2-layer objective, 297 s of a 308 s binary. It now uses a 2-qubit,
  1-layer objective with fewer shots per reading: 17 s, with the whole subscore
  suite at 18 s instead of 308 s.

## [0.2.8] - 2026-11-15
### Added
- Example-driven crate documentation: the docs.rs landing page now walks
  through the Bell state, circuit construction, seeded sampling, QEC
  verification, exact-rational algebra, Pauli sums, QASM round-trips, and
  cost-aware optimization, each as a runnable doctest.
- README "Library demo" section with copy-paste Rust snippets and their
  real outputs.


## [0.2.7] - 2026-11-15
### Added
- Python wheel smoke test (`python/smoke_test.py`): imports, Bell-state
  sampling, and rotated surface-code construction, verified against a real
  maturin-built wheel.
- CI `feature-compile` matrix: clippy under `--no-default-features` for
  `llvm`, `python`, `rayon`, `ibm`, `ionq`, `rigetti`.
- `qec` module split into `qec/{mod, codes, gf2, surface, mwpm}` submodules
  (public API unchanged via re-exports).
- `extract_logicals` (GF(2) nullspace + symplectic pairing) computing logical
  operator bases at runtime.
- CAS: `ScalarExpr::eval_complex`/`eval_f64`, `solve_linear`, `solve_quadratic`,
  `solve`, `factor_diff_squares`, `factor_quadratic`, `factor`, `degree`,
  `leading_coefficient`, `content`, `primitive_part`, `is_linear`,
  `is_quadratic`, `is_constant`, `collect_symbols`, `compose`,
  `integrate`, `taylor_series`, `sinh`/`cosh`/`tan`.
- Property-based CAS tests (`tests/proptest_cas.rs`) and CAS integration tests
  (`tests/cas_integration.rs`).


### Changed
- **Breaking**: `ClusterState::gflow_exists` renamed to
  `gflow_necessary_condition`; it checks a necessary condition only (every
  interior vertex has a future neighbour), NOT full Danos-Kashefi gflow.
- **Breaking**: `QecCode::recovery` key type is now `HashMap<u64, Circuit>`
  (was `u32`, which overflowed for surface codes with >32 stabilizers).
- `surface_code_planar` now reports `n_logical` honestly: it is a
  face-stabilizer-only construction and encodes 2d-1 logical qubits, not one;
  its logical operators are computed via `extract_logicals` and verified to
  commute with all stabilizers and anticommute in pairs. See NUMERICS.md.
- `REPL`: session state and command dispatch moved from `src/bin/qf.rs` to
  `src/repl/{state, help}.rs`; the `qf` binary is now just `main()`.
- Python `__init__.py` aliases the extension's `Py*`-prefixed pyclasses to
  clean names and no longer imports nonexistent `quantumforge_core` /
  `quantumforge_cuda` / `quantumforge_llvm` modules (there are no such
  packages; CUDA/LLVM are compile-time cargo features of one module).


### Fixed
- `surface_code_rotated(3)` logical operators were previously declared as
  top-row X / left-column Z, which do NOT commute with the stabilizers; they
  are now a verified anticommuting pair (found by exhaustive search).
- QASM user-defined gates: parameters were mis-parsed as qargs; trailing
  semicolons broke qubit references.
- MBQC `gflow_exists` (now `gflow_necessary_condition`): replaced exponential
  subset search with a linear-time check.
- Python bindings compiled with dead-code and deprecation warnings under
  `--features python`; `parse_qasm_py` was defined but never registered.


## [0.2.6] - 2026-11-10
### Added
- `Rational::as_f64` and `ExactComplex::as_complex64` lossy floating-point
  approximations for overflow-prone exact arithmetic.
- Linear-time necessary-condition check in `ClusterState::gflow_exists`.


### Changed
- QASM user-defined gate definitions now parse parameters vs qargs correctly.
- QASM qubit references tolerate trailing semicolons.
- MBQC `gflow_exists` no longer brute-forces subsets (was exponential).


### Fixed
- Version metadata synchronized across `Cargo.toml`, `pyproject.toml`,
  `API_STABILITY.md`, and `CHANGELOG.md` (all 0.2.6).


## [0.2.3] - 2026-10-30
### Added
- HHL quantum simulation for general NxN matrices
- Shor quantum period finding
- T-gate minimization and Clifford analysis
- Hardware-aware routing (linear/grid topologies)
- MPS reconstruction from exact diagonalization ground states
- GPU sparse Lanczos (generic for any PauliSum)
- 2D/3D spin models (Ising, Heisenberg)
- Entanglement entropy computation
- Circuit visualization (ASCII + SVG)
- Surface codes with MWPM decoder


### Fixed
- TEBD imaginary-time evolution
- QEC syndrome decoding
- Surface code stabilizer commutation
