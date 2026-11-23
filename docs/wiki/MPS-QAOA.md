# MPS-based QAOA

QuantumForge can run QAOA on a **matrix product state** with a bounded bond
dimension instead of a `2^n` state vector. This is the same regime as
MPS-JuliQAOA (Feeney et al., arXiv:2508.05883): MaxCut on 3-regular graphs,
diagonal cost Hamiltonian, bond-dimension/accuracy tradeoffs, and built-in
angle finding.

Module: `quantumforge::mps_qaoa` (Rust). Example: `examples/mps_qaoa.rs`.

## Why

A state-vector simulation of `n` qubits needs `2^n` amplitudes. QAOA states at
small depth `p` have bounded entanglement, so an MPS with bond dimension `χ`
represents them using `O(n χ²)` numbers — the simulation then scales
polynomially instead of exponentially, at the cost of a controlled truncation
error.

## What is implemented

| Piece | Implementation |
|---|---|
| Cost layer | `exp(-i γ w Z_u Z_v)` per edge, applied as an exact bond-2 MPO gate (`MPS::apply_diagonal_two_site`) |
| Mixer layer | `Rx(2β)` per qubit |
| Bond control | two-sweep SVD compression (`MPS::compress_range`), `max_bond` + `truncation_tol` |
| Cost expectation | transfer contraction (`MPS::expectation_two_site_product`), no state expansion |
| Local observables and DMRG normalization | MPS transfer contractions, no state expansion |
| Problem | `MaxCutProblem::random_3_regular(n, seed)` (configuration model), `exact_maxcut()` for `n ≤ 24` |
| Angle finding | parameter-shift gradient ascent with random restarts (`find_angles_mps`) |

Because the cost layer is diagonal, the entangling gates are applied as MPOs
rather than SWAP networks: the cost is linear in the edge length with a small
constant, and no qubit permutation has to be undone.

## Usage

```rust
use quantumforge::mps_qaoa::{find_angles_mps, run_maxcut_mps, MaxCutProblem, MpsQaoaConfig};

let problem = MaxCutProblem::random_3_regular(20, 42);
let config = MpsQaoaConfig {
    layers: 2,
    max_bond: 16,
    truncation_tol: 1e-12,
    optimizer_steps: 25,
    restarts: 2,
    seed: 7,
};
let found = find_angles_mps(&problem, &config)?;
println!("ratio = {:.4}", found.approximation_ratio(&problem));

for chi in [4, 8, 16, 32] {
    let o = run_maxcut_mps(&problem, &found.betas, &found.gammas, chi, 1e-12)?;
    println!("chi={chi:3} ratio={:.4} bond={}", o.approximation_ratio(&problem), o.max_bond_reached);
}
```

Run the worked example with `cargo run --release --example mps_qaoa`
(**release mode matters**: the SVD compression is ~50x faster than a debug
build).

## Conventions

- Cost Hamiltonian `H_C = Σ_(u,v) w (I - Z_u Z_v) / 2`, so the expectation is
  the expected cut weight and the approximation ratio is `<H_C> / maxcut`.
- Site `i` is bit `i` of the basis index (site 0 = least significant), matching
  `StateVector`.
- `QecCode`-style stabilizer machinery is unrelated; see [QEC-Codes](QEC-Codes.md).

## Validation

`tests/mps_qaoa.rs` cross-checks the MPS results against the exact state-vector
QAOA (`algorithms::qaoa_maxcut`) at large bond dimension, verifies the bond cap
is respected, checks the χ-convergence ordering, and validates the graph
generator. `tests/mps_longrange.rs` validates the long-range gates, compression
and expectations against exact state-vector arithmetic.

## Honest performance envelope

Measured on a desktop CPU (release build, 3-regular MaxCut, `p = 2`):

| n | χ | wall time per QAOA run |
|---|---|---|
| 16 | 32 | ~0.2 s |
| 20 | 32 | ~0.6 s |
| 24 | 32 | ~1.5 s |
| 32 | 64 | ~7 s |
| 64 | 64 | ~30 s |

The cost is dominated by the `O(E · n · χ³)` expectation sweeps and the SVD
compression. Publishing a 512-qubit result as in the paper would need a
GPU/parallel implementation; this module is the correct, validated foundation
and is honest about where it stands.

`MPS::to_state_vector` intentionally materializes an exponentially large
vector. Use the transfer-contraction observable APIs for large systems, and
inspect each workflow before assuming it remains tensor-network scalable. See
the [scaling strategy](../scaling-and-product-strategy.md) for the limits and
next product milestones.
