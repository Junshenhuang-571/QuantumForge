# Quantum circuit optimizer: product assessment and roadmap

This note compares QuantumForge's current circuit-optimization workflow with the capabilities described in the Qiskit and pytket compiler documentation. It is intended to guide an incremental path from a useful local optimizer toward a research and professional compilation tool. It is not a claim of feature parity with either compiler.

## Current workflow

The `quantumforge optimize` command accepts a QuantumForge circuit expression or an OpenQASM 2 circuit, runs the existing conservative or full rewrite passes, optionally routes to a line topology, and writes JSON or OpenQASM 2. It reports gate and depth changes and routing SWAP overhead. The library also exposes weighted logical cost estimates and checked routing for an explicitly supplied coupling map.

| Workflow capability | Current state |
| --- | --- |
| Batch circuit input and output | QuantumForge expression / OpenQASM 2 input; JSON / OpenQASM 2 output |
| Local logical rewrites | Existing identity removal, inverse cancellation, rotation fusion, and commutation-assisted cancellation |
| Basic connectivity routing | Checked shortest-path routing; CLI supports a line or user-supplied undirected JSON coupling map |
| Resource summary | Gate count, two-qubit count, logical depth, rewrite counts, and routing SWAP overhead |
| Explicit weighted logical cost | Library-level cost profile and cost model |
| Device layout search | Not exposed in the batch optimizer; current routing preserves logical indices |
| Device-native gate translation | Not exposed in the batch optimizer |
| Directed couplings, calibration, and scheduling | Not modeled by the batch optimizer; custom CLI maps are undirected |
| Independent equivalence verification | Not part of the CLI workflow |
| OpenQASM 3 workflow | Not supported by this batch path |

## What established compiler workflows include

Qiskit's documented staged transpiler separates initialization, layout, routing, translation, physical optimization, and scheduling. Its target can supply operation and connectivity constraints; its preset levels balance runtime against optimization effort. Qiskit also documents that choosing a good initial layout can avoid routing and that translation and scheduling are distinct from routing. See the [transpiler stages guide](https://qiskit.qotlabs.org/docs/guides/transpiler-stages), [preset pass managers](https://qiskit.qotlabs.org/docs/api/qiskit/transpiler_preset), and [`generate_preset_pass_manager`](https://qiskit.qotlabs.org/docs/api/qiskit/qiskit.transpiler.generate_preset_pass_manager).

pytket describes compilation as transforming an equivalent circuit to satisfy architecture connectivity and user constraints such as a restricted gate set while minimizing an objective such as CX count. It also has predicates for checking that compiled circuits satisfy target requirements. See [Compilation passes](https://docs.quantinuum.com/tket/user-guide/examples/circuit_compilation/compilation_example.html) and the [TKET overview](https://docs.quantinuum.com/tket/).

These references suggest that the key product boundary is a target-aware, inspectable compilation workflow, rather than simply adding more rewrite rules.

## Prioritized product roadmap

### P0 — Make results trustworthy and easy to reproduce

1. The first `--report FILE` artifact now records tool version, input format, selected mode and passes, target choice, rewrite counts, before/after resources, and the fact that equivalence was not independently checked. Next, add normalized input/output digests and richer warning details while keeping circuit output independently consumable by scripts.
2. The `--verify` option now exhaustively checks unitary action on all computational-basis states up to ten qubits, allowing a single global phase and rejecting non-unitary operations. It uses QuantumForge's own simulator, so next cross-check a representative corpus against an independent simulator or compiler and document tolerance and supported gates.
3. Publish stable CLI examples and ensure the generated API documentation builds for each release. At the time of this assessment, the [docs.rs page for 0.2.9](https://docs.rs/crate/quantumforge/0.2.9) reports a failed documentation build and points users to 0.2.8 as the last successful build; inspect the build log and restore the published docs before calling the workflow release-ready.
4. Add benchmark fixtures with reproducible seeds and report compile time alongside two-qubit count, depth, and routing overhead. Never present a logical cost estimate as hardware fidelity or runtime.

### P1 — Compile to an explicit device target

1. Define a serializable target description: physical qubit count, directed couplings, native gate set, and optional per-operation error/duration metadata.
2. Separate layout, routing, basis translation, and post-routing optimization into named passes. Preserve an explicit logical-to-physical mapping and include it in reports.
3. Add native-gate translation and direction correction before claiming a circuit is executable on a target. Validate every output operation against the target description.
4. Offer optimization objectives such as lexicographic two-qubit count then depth, or user-supplied weighted logical cost; expose the objective and its units in the report.

### P2 — Support professional interchange and integration

1. Add OpenQASM 3 import/export with explicit handling for measurements, control flow, timing, and unsupported constructs. OpenQASM 2 output should continue to fail clearly when the circuit contains operations it cannot represent.
2. Provide a stable Rust library API and a Python-facing batch interface so teams can compose optimization with existing experiment workflows.
3. Add backend adapters only after the target model and validation contracts are stable. Device submission, account management, and calibration fetching should remain separate integrations.

## Acceptance criteria for a device-targeted compile

- Every emitted gate is in the selected target's native set and every two-qubit operation respects its directed connectivity.
- The returned mapping explains where each logical qubit lives, including the final mapping after routing.
- The output is semantically checked for the supported circuit subset, or the report names why checking was skipped.
- The report is machine-readable, deterministic when seeded, and includes input/output resource counts and compiler version.
- Benchmarks compare quality and compilation time against a named baseline on public circuits; claims include the tested subset and hardware assumptions.

## Scope and limitations

QuantumForge's checked line routing and library cost model are useful foundations, but they do not by themselves establish hardware executability or predict device performance. A generic coupling graph also does not capture directionality, native gate decompositions, calibration drift, pulse-level duration, or measurement constraints. Those require an explicit target model and validation before the tool should describe its output as device-ready.
