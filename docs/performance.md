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
