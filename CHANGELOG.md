# Changelog

## [Unreleased]

### Added
- Add user-visible changes here as they are merged.

### Changed
- Add behavior or API changes here, including migration notes.

### Fixed
- Add user-visible bug fixes here.

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
