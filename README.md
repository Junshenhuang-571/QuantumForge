# QuantumForge

[![CI](https://github.com/Junshenhuang-571/QuantumForge/actions/workflows/ci.yml/badge.svg)](https://github.com/Junshenhuang-571/QuantumForge/actions/workflows/ci.yml) [![Crates.io](https://img.shields.io/crates/v/quantumforge)](https://crates.io/crates/quantumforge) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

QuantumForge is an open-source, local-first quantum simulation and research toolkit written in Rust. Use it to build circuits, simulate states and explicit noise models, sample measurements, and explore algorithms such as VQE and QAOA. It runs on a regular CPU and can be embedded as a Rust library; Python bindings are also available.

It is a **classical simulator**, not a connection to quantum hardware. The project is pre-1.0, so APIs may change. State-vector memory grows exponentially with qubit count; density matrices are for smaller systems. IBM Quantum, IonQ, and Rigetti modules are interface scaffolds, without device discovery or job submission.

QuantumForge is not yet a replacement for mature hardware compilers such as Qiskit or TKET. Its product direction is a reproducible local workbench for comparing circuit transformations and noise assumptions. See the [sourced platform comparison and commercial product hypotheses](docs/product-positioning.md) for the current scope, gaps, and evidence needed to validate that direction.

No known classical method efficiently simulates every arbitrary quantum circuit exactly as size and depth grow without bound. QuantumForge combines state vectors, stabilizer methods, tensor networks, and selected noisy trajectories; each has a different structure and accuracy envelope. See the [scaling strategy and product roadmap](docs/scaling-and-product-strategy.md) before choosing a method for a large workload.

## Start in a minute

Install the command-line REPL with Rust and Cargo:

```sh
cargo install quantumforge
qf
```

At the `qf>` prompt, make and sample a Bell pair:

```text
qubits 2
add H[0]
add CNOT[0,1]
draw
run
probs
add Measure[0,0]
add Measure[1,1]
shots 8 42
```

`probs` shows the ideal distribution. Add the two measurement operations before sampling; each shot is `|00>` or `|11>`. The seed (`42`) makes the counts repeatable. Type `help` for the REPL commands or `quit` to exit.

The batch `shots` command prints the seed and a 95% Wilson interval for each observed outcome, so finite-shot estimates are not mistaken for exact probabilities.

For scripts and CI, sample a circuit directly from the shell. The `shots` CLI measures every qubit automatically when the circuit has no explicit `Measure` operations:

```sh
quantumforge shots 2 1000 42 'Circuit[H[0], CNOT[0,1]]'
```

From a source checkout, run the same command with `cargo run --bin quantumforge -- shots 2 1000 42 'Circuit[H[0], CNOT[0,1]]'`.

## Optimize a circuit

The batch CLI can simplify a circuit and optionally route it onto a linear qubit topology:

```sh
quantumforge optimize 3 all --line 'Circuit[H[0], H[0], CNOT[0,2]]' > optimized.json
quantumforge optimize 3 all --input circuit.qasm --line --format qasm > optimized.qasm
quantumforge optimize 3 all --input circuit.qasm --report compile-report.json > optimized.json
quantumforge optimize 5 all --topology device-map.json --input circuit.qasm
quantumforge optimize 3 all --verify 'Circuit[H[0], H[0]]'
```

`all` enables rotation fusion, inverse cancellation, identity removal, and safe commutation rewrites; `conservative` limits changes to identity removal and inverse pairs. `--input` reads a circuit expression or OpenQASM 2.0 file; use `--input -` to read from standard input. `--line` routes over a linear topology; `--topology FILE` accepts a JSON target such as `{"num_qubits":5,"edges":[[0,1],[1,2],[1,3],[3,4]],"native_gates":["rz","x","h","cx","measure"],"directed_cx_edges":[[0,1],[1,2],[1,3],[3,4]]}`. `edges` are undirected connectivity used for routing; optional `directed_cx_edges` restrict CNOT orientation, and optional `native_gates` turns on strict gate-set validation. Supported names are `id`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`, `cx`, `cz`, `swap`, `measure`, and `conditional`. The map's qubit count must match `QUBITS`. The optimizer routes and then checks the result against these constraints; when the reverse CNOT direction is available and `h` is explicitly native, it corrects direction using Hadamard conjugation. It reports unsupported gates and couplings instead of implying that basis translation occurred. Other non-native gates are not decomposed automatically. `--verify` exhaustively compares all computational-basis inputs for unitary circuits up to 10 qubits, allowing one global phase; it refuses measurements and noise channels and does not replace independent cross-tool validation. The resource summary goes to the terminal; optimized JSON is the default output and can be loaded with `Circuit::from_json`, while `--format qasm` emits OpenQASM 2.0. `--report FILE` saves a separate machine-readable JSON record with the run settings, rewrite counts, before/after resource metrics, and verification status. QASM output rejects noise-channel operations instead of silently dropping them. When simulating JSON output, initialize the state with the same qubit count passed to the command. Device constraints are validation only: there is no global qubit placement, general gate decomposition, calibration-aware selection, or schedule generation.

See the [optimizer product assessment and roadmap](docs/optimizer-roadmap.md) for a research-backed comparison and prioritized path toward device-targeted compilation.

## Check dense simulation memory before running

Get the minimum complex-amplitude payload for a state vector and density matrix. Add a memory budget in MiB to see which dense representation exceeds it:

```sh
quantumforge estimate 30 4096
```

This is a lower-bound preflight: it excludes temporary workspaces and process overhead. It does not select a tensor-network, stabilizer, or trajectory method, and a favorable estimate does not guarantee that a simulation will complete. See the [scaling strategy](docs/scaling-and-product-strategy.md).

Library callers that want a pre-allocation check can use `StateVector::zero_with_memory_budget` or `DensityMatrix::maximally_mixed_with_memory_budget`. These constructors enforce the budget for the dense payload they allocate; later simulator workspaces still need additional headroom.

### Compare an optimized circuit under a thermal model

The optimizer can optionally run the optimized circuit through an exact density-matrix simulation with a declared `T1`, `T2`, bath gap, and one-/two-qubit gate duration. The report records reference and noisy probabilities, per-qubit Z expectations, purity, probability total-variation distance, and inserted channel counts:

```sh
quantumforge optimize 3 all --input circuit.qasm \
  --thermal 30 20 1 0.02 0.2 --verify --report experiment.json \
  > optimized.json
```

This analysis starts in the all-zero state, is limited to eight qubits, and rejects measurements and classical feed-forward. It is an exact calculation for the stated model, not a hardware result: readout, correlated errors, calibration drift, and parameter uncertainty are not included. `--thermal` requires `--report FILE` so simulation metadata stays separate from the circuit JSON on standard output. For caller-provided schedule timings and idle noise, use the library API described in the [noise guide](docs/wiki/Noise-and-Statistical-Mechanics.md).

Add `--compare --verify` to run both `all` and `conservative` optimization on the same input and target. The mode named in the command still determines the circuit written to standard output; the version 2 report embeds the input and selected output circuits, plus both candidates' circuits, resource counts, and equivalence results. When `--thermal` is also present, both candidates use the same noise parameters. Comparison requires a report and unitary circuits within the equivalence check's 10-qubit limit. Fingerprints use a deterministic, non-cryptographic FNV-1a identifier for matching artifacts; they are not integrity or security hashes. See the [reproducible optimizer experiment guide](docs/optimizer-experiment-workflow.md).

```sh
quantumforge optimize 3 all --input unitary.qasm \
  --compare --verify --thermal 30 20 1 0.02 0.2 \
  --report comparison.json > optimized.json
```

## Use QuantumForge from Rust

Add the crate to your project:

```sh
cargo add quantumforge
```

Build and simulate a Bell pair:

```rust
use quantumforge::{Circuit, Gate, StateVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.cnot(0, 1);

    let state = circuit.run(StateVector::zero(2)?)?;
    println!("{:?}", state.probabilities());
    // [0.5000000000000001, 0.0, 0.0, 0.5000000000000001]
    Ok(())
}
```

The simulator uses floating point. Exact rational arithmetic is available in the separate symbolic algebra module for supported expressions; it does not make general circuit simulation exact. See the [API docs](https://docs.rs/quantumforge) for details.

## What you can build

| Goal | Starting point |
|---|---|
| Learn gates, states, and measurement | The `qf` REPL and [getting-started guide](docs/getting-started.md) |
| Simulate noisy circuits or inspect density matrices | `DensityMatrix`, `ThermalGateNoiseModel`, noise channels, and the [noise guide](docs/wiki/Noise-and-Statistical-Mechanics.md) |
| Cross-check optimizer output independently | Optional [Qiskit validation tool](tools/validate_optimizer_qiskit.py) |
| Experiment with VQE, QML gradients, or QAOA | [algorithm guides](docs/wiki/Algorithms.md) and runnable examples in `examples/` |
| Explore tensor networks beyond small state vectors | MPS, TEBD, DMRG, and [MPS QAOA](docs/wiki/MPS-QAOA.md) |
| Build or verify quantum error-correcting codes | `qec` module and [QEC guide](docs/wiki/QEC-Codes.md) |
| Use the library from Python | [Python setup and examples](python/README.md) and [API reference](python/API.md) |

The analytic module supports exact rational and symbolic operations where defined. Simulation, optimization, and most algorithm outputs are numerical and depend on their configuration and convergence.

## Run an example

Clone the repository and run the minimal circuit example:

```sh
git clone https://github.com/Junshenhuang-571/QuantumForge.git
cd QuantumForge
cargo run --example hello_world
```

Browse [`examples/`](examples/) for QEC, QAOA, VQE, OpenQASM, and tensor-network workflows. The [documentation map](docs/README.md) links the API, guides, coverage notes, and release information.

## Optional backends

The default build uses the CPU. Optional `gpu` and `cuda` features accelerate selected simulation paths on supported NVIDIA systems; `llvm` JIT-compiles scalar numeric expressions, not quantum circuits. See the [GPU guide](docs/wiki/GPU-Acceleration.md) for requirements and current support.

## Contributing

QuantumForge is Apache-2.0 licensed. Contributions and issue reports are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](ROADMAP.md).
