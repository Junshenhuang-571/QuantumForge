# Architecture

## Overview

QuantumForge is a Rust quantum simulation library with multiple backends:

```
┌─────────────────────────────────────────────┐
│              User API (lib.rs)               │
├──────────┬──────────┬───────────┬───────────┤
│ StateVec │ Circuit  │  MPS/DMRG │  QEC      │
├──────────┴──────────┴───────────┴───────────┤
│           Tensor Network Engine             │
│  (tensor_network.rs: MPS, TEBD, Lanczos)    │
├─────────────────────────────────────────────┤
│         Backend Abstraction Layer           │
├──────────┬──────────┬───────────┬───────────┤
│   CPU    │   GPU    │   CUDA    │   LLVM    │
│ (default)│(cudarc)  │(native)   │(optional) │
└──────────┴──────────┴───────────┴───────────┘
```

## Core Modules

| Module | Purpose |
|--------|---------|
| `lib.rs` | Public API, Complex64, StateVector, Gate, Circuit |
| `quantum.rs` | SparseOperator, PauliSum, PauliString |
| `tensor_network.rs` | MPS, DMRG, TEBD, Lanczos (exact + sparse) |
| `algorithms.rs` | Grover, QFT, HHL, Shor, VQE, QAOA |
| `qec.rs` | Quantum error correction codes |
| `optimizer.rs` | Circuit optimization passes |
| `hardware.rs` | Hardware backend abstraction |
| `python.rs` | PyO3 Python bindings |

## Key Design Decisions

- **No arbitrary qubit caps**: CPU scales with RAM, GPU scales with VRAM
- **Exact rational arithmetic**: analytic module uses exact rationals
- **Backend-neutral**: CPU is reference; GPU/CUDA accelerate specific paths
- **MPS reconstruction**: Exact diagonalization results are converted to MPS via SVD

## Data Flow

1. User creates a `Circuit` or `StateVector`
2. Operations are applied (gates, measurements)
3. For large systems, MPS/DMRG is used for ground states
4. Results stored in `ground_state` field for correlation computation
