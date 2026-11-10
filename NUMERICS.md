# Numerical Policy

## Tolerances

| Context | Tolerance | Type | Rationale |
|---------|-----------|------|-----------|
| State vector comparison (identity checks) | 1e-12 | absolute | Amplitudes are O(1); absolute is simplest and sufficient |
| Probability sums | 1e-10 | absolute | Sum of O(1) values; 1e-10 is well above roundoff |
| Ground energy (Lanczos vs exact) | 1e-6 | absolute | Lanczos truncation; documented in cross-path tests |
| Observable expectation | 1e-12 | absolute | Direct matrix-vector; no accumulation |
| Sampling confidence intervals | Wilson 95% CI | statistical | 1/√N scaling; documented per-output |
| Circuit equivalence | 1e-9 | relative | Global phase tolerance; state vectors compared element-wise |
| GPU vs CPU energy | 1e-6 | absolute | f64 throughout; PCIe transfer introduces no rounding |

**Rule**: Use absolute tolerance when values are O(1) (amplitudes, probabilities, energies). Use relative tolerance when comparing quantities that may span many orders of magnitude (circuit costs, T-counts).

## Exact Arithmetic (ExactComplex / Rational)

Exact arithmetic trades rounding error for **coefficient explosion**. This is documented, not hidden:

- `Rational` uses `i64` numerator/denominator. Operations check for overflow via `checked_*`.
- When a rational operation overflows, the function returns `Err(AnalyticError::Overflow)`. It does **not** silently fall back to floating point.
- `ExactComplex` stores `Rational` real and imaginary parts. Multiplication of two rationals can produce numerators/denominators up to 128 bits before reduction. If reduction fails (overflow), the error propagates.
- **Threshold**: Any operation where intermediate products exceed `i64::MAX` returns an error. There is no automatic fallback to f64. Callers must handle the error or switch to a floating-point path explicitly.
- **Mitigation**: Use `canonicalize()` and `cse()` (common subexpression elimination) to reduce expression swell before evaluation. Documented in `examples/exact_identities.rs`.

## Lanczos

- **Convergence criterion**: Residual norm < 1e-10 (relative to largest eigenvalue magnitude). Checked after each restart.
- **Restart policy**: Full reorthogonalization every 10 iterations (configurable). GPU path uses `faer::self_adjoint_evd` for the Ritz values.
- **Non-convergence**: Returns `Err(LinearError::NotConverged)` with the last residual and iteration count. **Never** returns a plausible-looking wrong eigenvalue. Callers must handle the error.
- **Maximum iterations**: `max(100, 10 * dimension)` for dense; sparse uses `min(dimension, 1000)` with early termination on convergence.
- **Memory**: Sparse Lanczos uses up to 50% of available RAM (configurable via `cpu_max_qubits_sparse()`).

## Randomness

- **PRNG**: `SmallRng` (xoshiro256+) seeded with `0x9e37_79b9_7f4a_7c15` (not seed 0, which has poor statistical properties).
- **Reproducibility**: All random operations accept an explicit seed parameter. `StateVector::zero()` is deterministic. Random state generation uses the provided seed.
- **Limitations**: `SmallRng` is **not** suitable for cryptographic use or Monte Carlo error bounds. Period ≤ 2^256-1. No subnormal handling. Documented in `src/lib.rs` doc comment.
- **Scope**: Randomness is used only for: initial state generation in variational algorithms, and test data. Never for security-sensitive operations.

## Known Failure Modes

1. **Exact arithmetic overflow**: `Rational` operations can overflow `i64`. Returns error, no fallback. Mitigate with CSE/canonicalization. A lossy floating-point approximation is available via `Rational::as_f64` / `ExactComplex::as_complex64` when an exact result is not required.
2. **Lanczos non-convergence**: Returns error. Does not silently return wrong eigenvalues. Increase max iterations or use exact diagonalization for small systems.
3. **GPU context release**: Fixed via `AtomicBool` in `CudaBackendInner::Drop`. Double-release is now a no-op.
4. **Parser depth/size limits**: `MAX_PARSE_DEPTH = 256`, `MAX_CIRCUIT_OPS = 10_000`. Exceeding either returns an error. Protects against stack overflow on malicious input.
5. **QASM reset**: Not supported in pure-state simulation. Returns explicit error (not silent no-op).
6. **MBQC gflow**: necessary condition checked in linear time (a vertex needs at least one neighbour in the future set). The full Danos–Kashefi sufficiency check remains exponential and is not implemented; `gflow_exists` reports the necessary condition only.
7. **MPS truncation**: Bond dimension χ controls accuracy. At TFIM critical point, χ must scale as χ ~ L^(1/6) for energy convergence. Documented in calibration tests.
8. **SmallRng**: Not cryptographically secure. Not suitable for Monte Carlo error bounds. Period and statistical quality documented.
9. **SymbolicPauliSum**: convolution polynomial algebra over `ScalarExpr` coefficients; coefficients can be simplified via `SymbolicPauliSum::simplify`. Not a full CAS (no factorization/pattern matching beyond `Simplifier` rules); `verify_identity` hard-gates to pure polynomials.
10. **Surface code rotated**: boundary stabilizers verified only for d=3. `surface_code_rotated` panics for other d; use d=3 or `surface_code_planar`. Experimental `surface_code_rotated_search` and `verify_code` (commutation + single-error) are provided for research.
