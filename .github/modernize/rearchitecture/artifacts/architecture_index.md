# Architecture Index

This index is not the full contract. Do not implement from this file alone; follow the artifact paths below.

## Implementation Guide

### Global artifacts
- `unit_graph.yaml`
  - who reads: all implementation workers
  - use for: unit boundary, entrypoints, dependencies, exported signatures, shared refs
  - how to consume: map your assigned unit to its source anchors and exported signatures; follow `depends_on` and `shared_refs`
- `migration_boundary.yaml`
  - who reads: all implementation workers
  - use for: runtime scope and rewrite boundaries
  - how to consume: keep scope to the runtime-reachable boundary and do not expand to all source anchors
- `wire_contracts.yaml`
  - who reads: workers touching public interfaces, CLI, QASM, or external SDKs
  - use for: API and serialization contracts
  - how to read: filter by `unit` or by the target `name`
- `shared_modules.yaml`
  - who reads: workers touching shared logic or domain surfaces
  - use for: module ownership and field-level reference tracking
  - how to read: filter rows where `used_by_units` includes the worker's unit
- `cross_unit_state.yaml`
  - who reads: workers with implicit state or session-sensitive logic
  - use for: verifying whether a cross-unit state flow exists
  - how to read: filter rows where `writer.unit` or `reader.unit` matches the unit

### Unit: cli-repl
- external trigger: CLI command loop
- must read:
  - `units/cli-repl/behavior.yaml` — command semantics, state mutation, output behavior
  - `units/cli-repl/bindings.yaml` — command routing and runtime config
  - `units/cli-repl/unit_decomposition.yaml` — optional split candidates; do not treat as target structure
- relevant global rows:
  - `unit_graph.yaml` rows for `cli-repl`
  - `wire_contracts.yaml` rows where `unit == cli-repl`
  - `shared_modules.yaml` rows where `used_by_units` includes `cli-repl`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: core-circuit-runtime
- external trigger: library API
- must read:
  - `units/core-circuit-runtime/behavior.yaml` — exact semantics of state evolution, gates, and validation
  - `units/core-circuit-runtime/bindings.yaml` — library exports and external runtime config
  - `units/core-circuit-runtime/unit_decomposition.yaml` — optional split candidates
- relevant global rows:
  - `unit_graph.yaml` rows for `core-circuit-runtime`
  - `wire_contracts.yaml` rows where `unit == core-circuit-runtime`
  - `shared_modules.yaml` rows where `used_by_units` includes `core-circuit-runtime`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: variational-qml
- external trigger: parameterized optimization loops and QML training
- must read:
  - `units/variational-qml/behavior.yaml`
  - `units/variational-qml/bindings.yaml`
  - `units/variational-qml/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `variational-qml`
  - `shared_modules.yaml` rows where `used_by_units` includes `variational-qml`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: tensor-network-runtime
- external trigger: TEBD and DMRG execution
- must read:
  - `units/tensor-network-runtime/behavior.yaml`
  - `units/tensor-network-runtime/bindings.yaml`
  - `units/tensor-network-runtime/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `tensor-network-runtime`
  - `shared_modules.yaml` rows where `used_by_units` includes `tensor-network-runtime`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: qec-subsystem
- external trigger: QEC code construction and recovery
- must read:
  - `units/qec-subsystem/behavior.yaml`
  - `units/qec-subsystem/bindings.yaml`
  - `units/qec-subsystem/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `qec-subsystem`
  - `shared_modules.yaml` rows where `used_by_units` includes `qec-subsystem`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: mbqc-model
- external trigger: MBQC measurement pattern execution
- must read:
  - `units/mbqc-model/behavior.yaml`
  - `units/mbqc-model/bindings.yaml`
  - `units/mbqc-model/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `mbqc-model`
  - `shared_modules.yaml` rows where `used_by_units` includes `mbqc-model`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: aqc-annealing
- external trigger: annealing and adiabatic scheduling
- must read:
  - `units/aqc-annealing/behavior.yaml`
  - `units/aqc-annealing/bindings.yaml`
  - `units/aqc-annealing/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `aqc-annealing`
  - `shared_modules.yaml` rows where `used_by_units` includes `aqc-annealing`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: hardware-backends
- external trigger: external provider jobs
- must read:
  - `units/hardware-backends/behavior.yaml`
  - `units/hardware-backends/bindings.yaml`
  - `units/hardware-backends/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `hardware-backends`
  - `wire_contracts.yaml` rows for `hardware-backends`
  - `shared_modules.yaml` rows where `used_by_units` includes `hardware-backends`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: python-bindings
- external trigger: Python extension API
- must read:
  - `units/python-bindings/behavior.yaml`
  - `units/python-bindings/bindings.yaml`
  - `units/python-bindings/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `python-bindings`
  - `wire_contracts.yaml` rows for `python-bindings`
  - `shared_modules.yaml` rows where `used_by_units` includes `python-bindings`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence

### Unit: qtm-runtime
- external trigger: quantum Turing machine execution
- must read:
  - `units/qtm-runtime/behavior.yaml`
  - `units/qtm-runtime/bindings.yaml`
  - `units/qtm-runtime/unit_decomposition.yaml`
- relevant global rows:
  - `unit_graph.yaml` rows for `qtm-runtime`
  - `shared_modules.yaml` rows where `used_by_units` includes `qtm-runtime`
- before DONE report: artifacts_read, implemented_must_preserve, unresolved_or_deferred, verification_evidence
