# GPU Acceleration

## Feature Flags

| Flag | Backend | Use Case | Compiled in CI |
|------|---------|----------|----------------|
| `--features gpu` | cudarc (CUDA 12) | Sparse Lanczos, batched circuits | yes |
| `--features cuda` | Native CUDA driver | Batched circuit execution | **no** |

**`cuda` is not covered by CI.** It links `libcuda` at build time, so it cannot
compile on a GPU-less runner, and a broken `cuda` build would not be caught by
any push or PR. The `gpu` feature is the supported path anywhere: cudarc loads
the driver dynamically at runtime and needs no toolkit. Compile `cuda` locally
with `cargo check --features cuda` before relying on it.

## GPU Paths

### 1. Sparse Lanczos (DMRG)
- Generic for any `SparseOperator`/`PauliSum`
- Works with 2D/3D spin models
- VRAM-scaled qubit limit (no hard cap)
- Automatic GPU detection

### 2. Batched Circuit Execution
- `run_circuit_batch()` — 13x speedup at 20 qubits
- Batch of 32, depth 20: CPU 29.2s vs GPU 2.2s

### 3. GPU QNN Training
- `qnn ... gpu` command in REPL
- Batched Ry rotations on CUDA

## VRAM Scaling

```rust
// Detect GPU VRAM
let vram_mb = detect_gpu_vram_mb();
let max_qubits = max_qubits_for_vram(vram_mb);
```

No hardcoded qubit caps — scales with available VRAM.

## CPU Scaling

```rust
// CPU sparse Lanczos scales with RAM
let max_qubits = cpu_max_qubits_sparse(); // 50% of RAM
```

## Building with GPU

```bash
# NVIDIA (cudarc)
cargo build --features gpu

# Native CUDA driver
cargo build --features cuda

# Both
cargo build --features "gpu cuda"
```

## Benchmarking

```bash
cargo run --example gpu_bench --features gpu
```
