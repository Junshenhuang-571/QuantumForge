# Getting Started

QuantumForge runs locally as a command-line REPL or as a Rust library. This guide takes you from an empty circuit to a measured Bell pair. You need the stable Rust toolchain (Rust 1.88 or newer) and Cargo.

## Option 1: Open the REPL

Install the published Rust crate and start `qf`:

```sh
cargo install quantumforge
qf
```

If you already cloned the repository, start the same REPL with `cargo run --bin qf` from its root.

At the `qf>` prompt, enter:

```text
qubits 2
add H[0]
add CNOT[0,1]
draw
run
probs
add Measure[0,0]
add Measure[1,1]
shots 1000 42
```

`H` puts qubit 0 into a superposition and `CNOT` entangles it with qubit 1. `probs` shows the ideal probabilities (`|00>` and `|11>` each have probability 0.5). Add the two measurement operations before sampling; `shots` draws 1,000 samples and the seed `42` makes the counts repeatable. Type `help` to see the available commands and `quit` to leave the REPL.

### Run a batch without the REPL

The `quantumforge` binary is useful for scripts and automation. It adds measurements of all qubits if the circuit has no `Measure` operations:

```sh
quantumforge shots 2 1000 42 'Circuit[H[0], CNOT[0,1]]'
```

From a checkout, use `cargo run --bin quantumforge -- shots 2 1000 42 'Circuit[H[0], CNOT[0,1]]'`. The REPL does not add measurements automatically, so use the explicit `add Measure[q,c]` commands shown above when sampling there.

### Simplify and route a circuit

Use the `optimize` command to reduce gates and report operation count, two-qubit count, and logical depth. Add `--line` to route onto a linear nearest-neighbor topology. The report is written to the terminal and optimized JSON is written to standard output, so redirect it to a file:

```sh
quantumforge optimize 3 all --line 'Circuit[H[0], H[0], CNOT[0,2]]' > optimized.json
quantumforge optimize 3 all --input circuit.qasm --line --format qasm > optimized.qasm
quantumforge optimize 3 all --input circuit.qasm --report compile-report.json > optimized.json
quantumforge optimize 5 all --topology device-map.json --input circuit.qasm
quantumforge optimize 3 all --verify 'Circuit[H[0], H[0]]'
```

Input can be a circuit expression or an OpenQASM 2.0 file. The default output is JSON; load it with `Circuit::from_json` and initialize the state with the same qubit count (`3` in this example). Use `--format qasm` for portable OpenQASM 2.0 output. `--report FILE` writes run settings, rewrite counts, before/after resource metrics, and verification status to a separate JSON file. `--verify` exhaustively compares all computational-basis inputs for unitary circuits up to 10 qubits, allowing one global phase; measurements, conditional gates, and noise channels are rejected. This checks optimizer behavior against the local simulator but is not independent cross-tool validation. `--topology FILE` accepts `num_qubits` and `edges` for an undirected coupling map, plus optional `native_gates` and directed `directed_cx_edges`; for example: `{"num_qubits":3,"edges":[[0,1],[1,2]],"native_gates":["h","rz","cx","measure"],"directed_cx_edges":[[0,1],[1,2]]}`. The native gate list enables strict validation, while the directed edges restrict CNOT orientation. If the reverse link is available and H is listed as native, the optimizer reverses a CNOT with Hadamard conjugation; other unsupported operations produce a constraint error and no circuit. The qubit count must equal `QUBITS`; `--line` and `--topology` cannot be combined. Noise channels cannot be represented in QASM, so that output format returns an error rather than dropping them. This target check does not model calibration, placement, timing, or device submission.

For a small, unitary circuit, include `--thermal T1 T2 BETA_DELTA ONE_QUBIT_TIME TWO_QUBIT_TIME --report experiment.json` to compare the optimized circuit's reference distribution with an exact density-matrix simulation under the declared thermal model. The report includes probabilities, per-qubit Z expectations, purity, and probability total-variation distance. The CLI analysis starts in `|0...0>`, supports up to 8 qubits, and excludes measurements, feed-forward, readout/correlated errors, and model-parameter uncertainty. These are model results, not hardware measurements. Use the `ThermalGateNoiseModel::apply_scheduled` library API when you have operation timings and want idle-gap noise.

Add `--compare --verify` to compare both `all` and `conservative` rewrite modes against the same input and target. The mode in the command selects which candidate is written to standard output; both candidates are checked for unitary equivalence, and the report includes resource counts and deterministic non-cryptographic fingerprints. Add `--thermal` to also simulate both candidates under the same thermal parameters. This comparison requires `--report FILE` and is limited to unitary circuits of at most 10 qubits; thermal simulation itself is limited to 8 qubits.

## Option 2: Add the Rust library

In an existing Cargo project, add QuantumForge:

```sh
cargo add quantumforge
```

Then create `src/main.rs`:

```rust
use quantumforge::{Circuit, Gate, StateVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.cnot(0, 1);

    let state = circuit.run(StateVector::zero(2)?)?;
    println!("probabilities: {:?}", state.probabilities());
    Ok(())
}
```

Run it with `cargo run`. Probabilities are floating-point values, so the printed `0.5` may appear as `0.5000000000000001`.

## Option 3: Use Python

Python bindings are built from the repository with [maturin](https://www.maturin.rs/). Follow the [Python setup guide](../python/README.md) to build and install the local wheel, then try the runnable scripts in `python/examples/`.

## Choose a next step

- Import or export OpenQASM: [`examples/qasm_roundtrip.rs`](../examples/qasm_roundtrip.rs)
- Optimize a small Hamiltonian with VQE: [`examples/vqe_exact.rs`](../examples/vqe_exact.rs)
- Build and verify a repetition or surface code: [QEC guide](wiki/QEC-Codes.md)
- Use matrix product states for larger 1D systems: [MPS guide](wiki/MPS-QAOA.md)
- Find the API and all subsystem guides: [documentation map](README.md)

## Know the simulator's limits

State-vector memory grows exponentially as qubits are added; density matrices grow faster and are intended for small systems. The simulator is useful for learning and prototyping, but its numerical results are not independently validated scientific or production outputs. QuantumForge is a classical simulator and does not submit jobs to quantum hardware.
