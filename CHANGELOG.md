# Changelog

## [Unreleased]

### Added
- Add user-visible changes here as they are merged.

### Changed
- Add behavior or API changes here, including migration notes.

### Fixed
- Add user-visible bug fixes here.

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
