# Data Model

## Core domain entities

The project is dominated by computational objects rather than a relational schema. The main persistent/working data structures are in-memory objects used for circuit execution, optimization, and simulation.

### `Complex64`
- Located in `src/lib.rs`
- Represents a complex scalar with `re` and `im` parts.
- Used as the primitive numeric type across the analytic and circuit layers.

### `StateVector`
- Located in `src/lib.rs`
- Represents an amplitude vector over `2^n` basis states.
- Core execution state for gate-model circuits and measurement workflows.

### `DensityMatrix`
- Located in `src/lib.rs`
- Represents mixed-state density matrices for noisy or stochastic simulation.
- Used for depolarizing, amplitude damping, phase damping, and partial trace workflows.

### `Circuit` / `CircuitOp`
- Located in `src/lib.rs`
- Represents a gate list or executable sequence over qubits.
- Encodes operations such as `SingleQubit`, `Cnot`, `Cz`, `Swap`, and measurement-related ops.

### `Gate` / `ParameterizedCircuit`
- Located in `src/lib.rs`
- Models gate primitives and parameterized quantum circuits for VQE/QML.
- Supports rotation-axis parameterization and training configuration.

### `PauliSum` / `PauliString`
- Located in `src/analytic.rs` and `src/quantum.rs`
- Encodes symbolic or exact Pauli algebra used in analytic and optimization stages.

### `MPS` / `Tensor`
- Located in `src/tensor_network.rs`
- Represents matrix-product states and intermediate tensor contractions.
- Used by TEBD and DMRG solvers for larger systems.

### `QecCode`
- Located in `src/qec.rs`
- Represents an error-correcting code, syndrome logic, and recovery procedures.

### `HardwareBackend` / `JobRequest`
- Located in `src/hardware.rs`
- Represents a provider abstraction for external quantum hardware execution.
- Includes device metadata, task submission, and job result accounting.

## Relationships and behavior

- `Circuit` is the primary mutable model of a quantum algorithm or experiment.
- `StateVector` and `DensityMatrix` are runtime execution states produced from `Circuit` operations.
- `ParameterizedCircuit` is a derivative view used in optimization and training loops.
- `MPS` and `QecCode` are specialized representations for larger-system and fault-tolerant simulation.
- `HardwareBackend` models execution outside the local CPU reference backend.

## Key entity summary

1. `StateVector` — amplitudes for a pure quantum state.
2. `DensityMatrix` — mixed-state representation for noisy simulation.
3. `Circuit` — ordered gate operations over qubits.
4. `ParameterizedCircuit` — trainable circuit for VQE/QML.
5. `PauliSum` — exact symbolic Hamiltonian representation.
6. `MPS` — tensor-network state representation for large systems.
7. `QecCode` — error-correcting code layout and syndrome references.
8. `HardwareBackend` — provider abstraction for remote devices.

## Transaction and persistence boundaries

This repository is primarily CPU-memory based and does not define a relational database or application transaction model. Core data flows are in-memory, deterministic, and local to the Rust process. Runtime configuration is handled through feature flags, command-line parameters, and optional cloud/SDK integrations rather than a database schema.
