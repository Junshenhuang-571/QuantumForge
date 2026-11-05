# Project Structure

## Top-level module breakdown

This repository is a single Rust crate for quantum simulation and analysis. The public API is exposed by `src/lib.rs`, and the CLI interactive shell lives in `src/bin/qf.rs`.

- `src/lib.rs`: public surface and shared primitives (`Complex64`, `StateVector`, `DensityMatrix`, `Circuit`, `Gate`)
- `src/analytic.rs`: exact rational/symbolic math and Pauli algebra
- `src/parser.rs`: circuit grammar and text parsing
- `src/quantum.rs`: linear algebra abstractions such as `SparseOperator`, `PauliSum`, `HilbertSpace`
- `src/tensor_network.rs`: MPS/TEBD/DMRG and Lanczos-based approximation helpers
- `src/algorithms.rs`: Grover, QAOA, QPE, QFT, Deutsch-Jozsa, Bernstein-Vazirani variants
- `src/optimizer.rs`: parameter optimization and circuit optimization configuration
- `src/qec.rs`: quantum error correction codes and recovery logic
- `src/mbqc.rs`: measurement-based quantum computation state and patterns
- `src/aqc.rs`: adiabatic quantum evolution and annealing schedules
- `src/qtm.rs`: quantum Turing machine transition model
- `src/hardware.rs`: provider-neutral hardware backend abstraction for IBM/IonQ/Rigetti/local execution
- `src/python.rs`: PyO3 Python bindings
- `src/cuda.rs`, `src/llvm_backend.rs`: optional execution backends under feature flags
- `src/qasm.rs`, `src/qasm_export.rs`: QASM parsing/export support
- `src/serialize.rs`: serialization of circuits and operations

## Layer architecture

The project is organized more by computational domain than by a layered web application stack:

1. Core numeric and state layer
   - exact scalar algebra (`analytic.rs`)
   - canonical state objects (`lib.rs`)
   - linear operator abstractions (`quantum.rs`)

2. Circuit and parser layer
   - circuit syntax parsing (`parser.rs`)
   - QASM import/export (`qasm.rs`, `qasm_export.rs`)
   - circuit serialization (`serialize.rs`)

3. Algorithm and model execution layer
   - gate-model algorithms (`algorithms.rs`)
   - tensor-network solvers (`tensor_network.rs`)
   - MBQC (`mbqc.rs`)
   - AQC (`aqc.rs`)
   - QTM (`qtm.rs`)
   - QEC (`qec.rs`)

4. Runtime/backend layer
   - CPU reference semantics in the main library
   - optional CUDA and LLVM acceleration
   - cloud provider abstractions in `hardware.rs`

5. Interface layer
   - interactive REPL in `src/bin/qf.rs`
   - Python bindings via `python.rs`

## Entry points

- Library entry: `src/lib.rs`
- CLI entry: `src/bin/qf.rs`
- Optional Python extension: `src/python.rs`
- Examples: `examples/` and benchmark code under `benches/`

## File counts by domain

- core + state: ~6 major files
- algorithms and simulation: ~8 major files
- backend/interop: ~5 major files
- CLI/examples/docs: ~10+ supporting files

## Functional domains

- exact quantum algebra
- gate-model circuit execution
- variational/QML workflows
- tensor-network methods (TEBD/DMRG)
- QEC and fault-tolerance primitives
- hardware integration abstractions
- REPL and scripting workflows

## Project type classification

Project type: library + CLI + optional runtime backends.

This is not a monolith application server; it is a computational engine crate with a REPL surface and optional accelerator integration. The crate exposes reusable Rust APIs, optional Python bindings, and a user-facing command-line environment.
