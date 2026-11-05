# Tech Stack

## Language and runtime

- Language: Rust
- Edition: 2024 (`Cargo.toml`)
- Package: `quantumforge`
- Crate type: library + cdylib (`src/lib.rs`)
- Primary execution target: Rust library and CLI; Python bindings via PyO3 are optional

## Core dependencies

- `rand = "0.8"`
- `serde = { version = "1.0", features = ["derive"] }`
- `serde_json = "1.0"`
- `faer = "0.24"`
- `num-complex = "0.4"`

## Optional feature flags

- `llvm = []`
- `cuda = []`
- `gpu = ["cudarc", "cudarc/cuda-12040"]`
- `python = ["pyo3"]`
- `ibm = ["reqwest"]`
- `ionq = ["reqwest"]`
- `rigetti = ["reqwest"]`

## Optional accelerator integrations

- CUDA via `cudarc` with `cuda-12040` compatibility
- LLVM JIT/back-end support via `llvm` feature flag
- Python bindings via `pyo3` and maturin packaging metadata
- Cloud/provider integrations with `reqwest` for IBM, IonQ, and Rigetti APIs

## Build tooling

- Cargo workspace: one package, resolver 2
- Benchmarks: Criterion (`criterion = "0.5"`)
- Property-based tests: Proptest (`proptest = "1"`)

## Runtime constraints and compatibility

The project expects a modern Rust toolchain with the 2024 edition. Optional GPU and LLVM builds require native toolkit support (`cuda`, `clang`, `opt`, `llc`), as described in the root documentation. The Python bindings target an ABI3 Python wheel for Python 3.8+ (`pyo3` with `abi3-py38`).

## Existing stack only

This file records the current stack. It does not prescribe a migration target or redesign direction.
