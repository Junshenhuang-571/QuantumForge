# Getting Started

## Installation

```bash
cargo add quantumforge
```

Or clone and build:

```bash
git clone https://github.com/Junshenhuang-571/QuantumForge.git
cd QuantumForge
cargo build
cargo run --bin qf
```

## Basic Usage

```rust
use quantumforge::{Circuit, Gate, StateVector};

let mut circuit = Circuit::new();
circuit.h(0);
circuit.cnot(0, 1);

let mut sv = StateVector::zero(2).unwrap();
// Apply circuit...
```

## REPL

Run `cargo run --bin qf` for an interactive REPL with commands like:
- `state` — show state vector
- `gate H 0` — apply Hadamard to qubit 0
- `measure` — measure all qubits
- `dmrg` — run DMRG
- `gpu` — GPU acceleration commands
