# Performance and Benchmarking

QuantumForge is a numerical simulation project, so performance work should be measured and reproducible.

## Recommended checks

Before merging performance-sensitive changes, run:

```bash
cargo test --workspace --lib --tests
cargo bench --bench core
```

The `circuit_optimization` Criterion group measures optimizer scaling, normal
execution, reuse of a pre-optimized circuit, and optimize-then-run overhead.
Use:

```bash
cargo bench --bench core -- circuit_optimization
```

For repeated runs, call `let (optimized, stats) = circuit.optimized()` once,
then reuse `optimized`. `circuit.run_optimized(state)` is convenient for a
single run, but includes the optimization cost each time it is called.

## Comparing optimization targets

`CircuitCostProfile::from_circuit` reports logical single-qubit, CNOT, CZ,
SWAP, T/Tdg, measurement, conditional, and channel counts, plus dependency
depth and qubit footprint. Use `CircuitCostModel` to weight those resources for
a particular optimization objective:

```rust
use quantumforge::optimizer::{
    CircuitCostModel, CircuitCostProfile, OptimizeConfig, optimize_for_cost_model,
};

let fault_tolerant_model = CircuitCostModel {
    single_qubit_gate: 0.0,
    t_gate: 1.0,
    tdg_gate: 1.0,
    ..CircuitCostModel::default()
};
let (optimized, decision) =
    optimize_for_cost_model(&circuit, &OptimizeConfig::all(), &fault_tolerant_model)?;
let selected_profile = CircuitCostProfile::from_circuit(&optimized);
let estimated_cost = fault_tolerant_model.estimate(&selected_profile)?;
println!("applied={}, selected cost={estimated_cost}", decision.applied);
```

These scores are relative logical estimates. They do not predict execution
time, and depth does not include routing or scheduling constraints. Choose
weights from the target backend and compare before/after profiles using the same
model; do not interpret a lower score as proof of faster hardware execution.
`optimize_for_cost_model` evaluates a rewritten copy and returns the original
circuit if the candidate does not strictly improve the estimate. Cost weights
must be finite and nonnegative; invalid or overflowing estimates return an
error rather than silently selecting a candidate.

If the change touches GPU or CUDA paths, also test:

```bash
cargo run --example gpu_bench --features gpu
cargo bench --bench cuda_batch --features cuda
```

The CUDA batch benchmark checks GPU results against CPU reference outputs, then
reports CPU time, resident-GPU kernel time, transfer timing, and end-to-end GPU
time for multiple batch sizes. One run on an NVIDIA RTX A5000 Laptop GPU
(compute capability 8.6, Windows, CUDA driver 13.2) with 8-qubit states and a
320-operation circuit produced:

| Batch states | CPU | GPU end-to-end | Relative speed |
|---:|---:|---:|---:|
| 1 | 0.065 ms | 8.653 ms | 0.01x |
| 16 | 1.039 ms | 8.939 ms | 0.12x |
| 64 | 4.074 ms | 8.737 ms | 0.47x |
| 256 | 17.563 ms | 10.051 ms | 1.75x |
| 1024 | 73.939 ms | 17.117 ms | 4.32x |

These are machine- and workload-specific measurements, averaged over five
iterations; they are evidence for batched throughput, not a general GPU
speedup claim. In this workload, transfer and launch overhead make GPU execution
slower at small batch sizes. Consequently, callers should use the explicit CUDA
batch API for large batches rather than assuming automatic dispatch is always
faster. Re-run the benchmark on the target hardware before choosing a dispatch
threshold.

## What to benchmark

Focus on the areas most likely to affect real usage:

- gate application performance on state vectors
- density-matrix operations
- sparse Lanczos convergence
- tensor-network contraction and SVD workloads
- measurement and shot generation throughput
- GPU offload benefit at larger qubit counts

## Recorded baseline

One `cargo bench --bench core` run, 2026-11-20. These are a reference point, not
a guarantee: a single run on one machine, so treat differences below ~2x as
noise. Criterion reports the mean of its sample distribution.

- Machine: 11th Gen Intel Core i9-11950H (Windows), release profile
- Toolchain: rustc 1.88.0 (6b00bc388 2025-06-23)
- Command: `cargo bench --bench core`
- Features: default (`rayon`)

### State vector

| benchmark | 8 qubits | 12 qubits | 16 qubits |
|---|---|---|---|
| `apply_all_h` | 150.25 µs | 475.71 µs | 10.594 ms |
| `cnot_chain` | 2.860 µs | 287.41 µs | 9.875 ms |

The jump from 12 to 16 qubits is the expected 16x state-vector growth (4096 to
65536 amplitudes) plus allocation, not a cliff in the gate loop.

### Density matrix

| benchmark | 4 qubits | 6 qubits | 8 qubits |
|---|---|---|---|
| `cnot` | 289.25 ns | 4.892 µs | 331.11 µs |
| `from_state` | 273.73 ns | 3.525 µs | 280.72 µs |

### Exact-rational analytic core

| benchmark | time |
|---|---|
| `rational_add_chain` | 10.343 µs |
| `rational_mul_reduce` | 5.485 µs |
| `exact_complex_add_chain` | 14.109 µs |
| `pauli_sum_build_add` | 3.465 µs |

### Measurement and optimization

| benchmark | time |
|---|---|
| `measure_12q` | 8.105 µs |
| `circuit_optimization/cancel_inverse_pairs/128` | 7.386 µs |
| `circuit_optimization/cancel_inverse_pairs/512` | 29.347 µs |
| `circuit_optimization/cancel_inverse_pairs/2048` | 116.24 µs |
| `circuit_optimization/run_redundant_512_pairs` | 38.108 ms |
| `circuit_optimization/run_preoptimized_512_pairs` | 8.368 ns |
| `circuit_optimization/optimize_then_run_512_pairs` | 30.673 µs |

`run_preoptimized` at 8 ns against 38 ms for the redundant circuit is the
clearest number here: optimizing once and reusing the result is roughly six
orders of magnitude cheaper than re-running the unoptimized circuit.

## Reporting performance changes

When submitting a benchmark-related change, include:

- hardware used
- Rust version
- feature flags enabled
- benchmark command executed
- before/after numbers if available

## Performance guidance

- Prefer correctness-first changes before optimization work.
- Keep numerical comparisons reproducible and deterministic when possible.
- Document any assumptions around GPU availability or backend differences.
