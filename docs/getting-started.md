# Getting Started with QuantumForge

This guide is the fastest path to installing QuantumForge, running the REPL,
and trying a first circuit.

## Prerequisites

- Rust stable toolchain
- Cargo
- Optional: CUDA or LLVM toolchains for accelerated backends

## Install from source

```bash
git clone https://github.com/Junshenhuang-571/QuantumForge.git
cd QuantumForge
cargo build
```

To run the REPL:

```bash
cargo run --bin qf
```

## First circuit

```text
qf> qubits 2
qf> add H[0]
qf> add CNOT[0,1]
qf> add Measure[0,0]
qf> add Measure[1,1]
qf> draw
qf> shots 1000 42
```

This prepares a Bell state and samples both qubits. The fixed seed makes the
sample reproducible. Use `run` then `state` to inspect the state vector rather
than sampling.

## Minimal library example

```rust
use quantumforge::{Circuit, StateVector, parse_circuit};

fn main() {
    let circuit = parse_circuit("Circuit[H[0], CNOT[0,1], Measure[0,0]]").unwrap();
    let initial = StateVector::zero(2).unwrap();
    let result = circuit.run_shots(&initial, 1000, 42).unwrap();
    println!("counts: {:?}", result.counts);
}
```

## Useful commands

Inside the REPL, try:

- `help`
- `qubits 3`
- `add H[0]`
- `draw`
- `run`
- `state`
- `probs`
- `shots 1000 42`

## Next steps

- Read the project README for the full feature overview.
- Explore the examples directory for domain-specific workflows.
- Start with a small gate-model circuit before moving to variational or QEC tasks.
