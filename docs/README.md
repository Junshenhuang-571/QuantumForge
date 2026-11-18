# QuantumForge documentation map

Start here. Everything is either rendered from the source (rustdoc, on
[docs.rs](https://docs.rs/quantumforge)) or written up as a guide below.

## Start reading

| I want to... | Go to |
|---|---|
| install and run my first circuit | [Getting started](getting-started.md) · [README](../../README.md) |
| browse the API with examples | <https://docs.rs/quantumforge> (every public item is documented) |
| learn a subsystem in depth | the guides listed below |
| run QAOA on hundreds of qubits | [MPS-based QAOA](wiki/MPS-QAOA.md) |
| use the Python bindings | [python/README.md](../python/README.md) |
| cut a release | [Release checklist](releasing.md) |

## Guides

- [Algorithms](wiki/Algorithms.md) — algorithm modules and how they fit together
- [Architecture](wiki/Architecture.md) — module layout and data flow
- [QEC codes](wiki/QEC-Codes.md) — stabilizer codes, encoders, decoders
- [GPU acceleration](wiki/GPU-Acceleration.md) — `cuda` vs `gpu`
- [MPS-based QAOA](wiki/MPS-QAOA.md) — matrix-product-state simulation for MaxCut
- [QAOA parameter transfer](wiki/QAOA-Parameter-Transfer.md) — TQA init, transferred angles, targeted-single-layer optimization
- [Contributing](wiki/Contributing.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md)

## API by subsystem

Rustdoc anchors (substitute the version you depend on):

| Subsystem | Module |
|---|---|
| Circuits, states, gates, simulation | [`quantumforge`](https://docs.rs/quantumforge/latest/quantumforge/) (crate root) |
| Exact-rational algebra (scalars, Pauli sums, simplification) | [`analytic`](https://docs.rs/quantumforge/latest/quantumforge/analytic/) |
| Core linear algebra (spaces, operators, kets) | [`quantum`](https://docs.rs/quantumforge/latest/quantumforge/quantum/) |
| Circuit algorithms (QAOA, HHL, Shor, Grover) | [`algorithms`](https://docs.rs/quantumforge/latest/quantumforge/algorithms/), [`algorithms_templates`](https://docs.rs/quantumforge/latest/quantumforge/algorithms_templates/) |
| Cost-aware circuit optimization | [`optimizer`](https://docs.rs/quantumforge/latest/quantumforge/optimizer/) |
| Tensor networks (MPS, TEBD, DMRG) | [`tensor_network`](https://docs.rs/quantumforge/latest/quantumforge/tensor_network/) |
| MPS-based QAOA | [`mps_qaoa`](https://docs.rs/quantumforge/latest/quantumforge/mps_qaoa/) |
| QAOA parameter transfer + targeted-layer | [`qaoa_transfer`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/) |
| Quantum error correction | [`qec`](https://docs.rs/quantumforge/latest/quantumforge/qec/) |
| Stabilizer formalism | [`stabilizer`](https://docs.rs/quantumforge/latest/quantumforge/stabilizer/) |
| Adiabatic quantum computing | [`aqc`](https://docs.rs/quantumforge/latest/quantumforge/aqc/) |
| Measurement-based QC | [`mbqc`](https://docs.rs/quantumforge/latest/quantumforge/mbqc/) |
| Quantum Turing machines | [`qtm`](https://docs.rs/quantumforge/latest/quantumforge/qtm/) |
| OpenQASM 2.0 | [`qasm`](https://docs.rs/quantumforge/latest/quantumforge/qasm/), [`qasm_export`](https://docs.rs/quantumforge/latest/quantumforge/qasm_export/) |
| Circuit expression parser | [`parser`](https://docs.rs/quantumforge/latest/quantumforge/parser/) |
| Numeric IR | [`numeric_ir`](https://docs.rs/quantumforge/latest/quantumforge/numeric_ir/) |
| Interactive REPL | [`repl`](https://docs.rs/quantumforge/latest/quantumforge/repl/) |
| Serialization | [`serialize`](https://docs.rs/quantumforge/latest/quantumforge/serialize/) |
| Hardware providers (scaffolds) | [`hardware`](https://docs.rs/quantumforge/latest/quantumforge/hardware/) |
| LLVM JIT backend | [`llvm_backend`](https://docs.rs/quantumforge/latest/quantumforge/llvm_backend/) |
| CUDA backend | [`cuda`](https://docs.rs/quantumforge/latest/quantumforge/cuda/) |
| Python bindings | [`python`](https://docs.rs/quantumforge/latest/quantumforge/python/) |

## Examples

Every example is runnable (`cargo run --example <name>`); the ones marked
*(release)* are too slow in a debug build.

- `hello_world` — minimal Bell state
- `qec_roundtrip` — surface-code verification + a full repetition-code
  encode → inject → syndrome → recover cycle
- `qasm_roundtrip` — OpenQASM 2.0 import → export → re-import
- `vqe_exact` — VQE against the exact ground-state energy
- `mps_qaoa` *(release)* — QAOA on a matrix product state with a χ sweep
- `qaoa_transfer` *(release)* — parameter transfer + targeted-layer optimization
- `qaoa_transfer_families` *(release)* — the same across 3r / er / ba / weighted families
- `cost_aware_optimization` — rewrite against a weighted cost model
- `qtm_probe` — quantum Turing machine exploration
- `gpu_bench` — GPU benchmark entry point

## Conventions used throughout

- Qubit/site `i` is bit `i` of a basis index: **site 0 is the least
  significant**, matching `StateVector`.
- Expectation values and probabilities are `f64` unless the item is in
  `analytic`, which is exact-rational.
- Angles are in radians.
- Measurement outcomes are classical-register bit vectors, least-significant
  bit first.
