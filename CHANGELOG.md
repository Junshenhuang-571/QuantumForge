# Changelog

## [Unreleased]

### Added
- **Complete API documentation.** Every public item in the crate now carries a
  doc comment (775 items added across 24 modules), verified by a
  `missing_docs` lint that is now a hard error rather than a warning, so the
  completeness cannot silently regress. This includes items that only compile
  behind a feature flag (the `cuda` backend).
- Documentation map: `docs/README.md` ties together docs.rs, the wiki guides,
  the Python README, the examples and the release checklist, with an
  API-by-subsystem index.
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
- MPS: `apply_diagonal_two_site`, `compress_range`,
  `expectation_two_site_product`, `norm_sqr_mps`.
- Examples: `qec_roundtrip` (surface-code verification + repetition-code
  encode/inject/syndrome/recover cycle), `qasm_roundtrip` (OpenQASM 2.0
  import/export/re-import), `vqe_exact` (VQE vs exact ground-state energy).
- Python first-run path: `python/README.md` plus worked scripts
  `python/examples/bell.py`, `qec.py`, `vqe.py`.

### Changed
- Add behavior or API changes here, including migration notes.

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
