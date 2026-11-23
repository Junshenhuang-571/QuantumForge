# QuantumForge product position

**Research checked:** 23 November 2026. This is a product assessment, not a claim that QuantumForge has feature parity with the platforms below. Platform capabilities evolve; links point to first-party documentation.

## Short version

QuantumForge is an open-source Rust quantum simulation and research toolkit. Today it is strongest as a local laboratory for circuit simulation, algorithm experiments, explicit thermal-noise studies, and small-circuit optimizer validation. It is not yet a hardware execution platform or a mature device compiler.

**Credible product direction:** become a reproducible experiment workbench where a quantum algorithm team can compare circuit variants under explicit device constraints and noise assumptions, then export the evidence with the circuit.

That direction still needs customer discovery. There is no user or revenue evidence in this repository, so the proposed buyer and paid product below are hypotheses to validate rather than traction claims.

## Comparison with established platforms

| Platform | What its official materials emphasize | Where QuantumForge stands today |
|---|---|---|
| [IBM Qiskit](https://qiskit.qotlabs.org/docs/guides/transpiler-stages) | A broad Python SDK and staged transpiler: initialization, layout, routing, native-instruction translation, hardware-aware optimization, and optional scheduling. | QuantumForge has local rewrites, basic routing, target checks, one directed-CNOT correction, and reports. It lacks general layout search, basis translation, calibration-aware choices, and a hardware execution path. |
| [Quantinuum pytket / TKET](https://docs.quantinuum.com/tket/user-guide/examples/circuit_compilation/compilation_example.html) | An optimizing compiler with architecture placement/routing, restricted-gate predicates, cost objectives, and backend extensions for compilation and execution. | QuantumForge offers a Rust-first library and a research-focused noise/statistical-mechanics toolkit. Its target description is narrower and its optimizer is much less mature. |
| [Amazon Braket SDK](https://docs.aws.amazon.com/braket/latest/developerguide/braket-constructing-circuit.html) | A provider and cloud workflow: construct circuits, inspect device support, run local noise simulation, and submit tasks to managed simulators or QPUs. | QuantumForge runs locally and has no cloud task submission or provider integration. Its opportunity is reproducible research workflows that can complement such execution services. |
| [NVIDIA CUDA-Q](https://developer.nvidia.com/qoda) | A QPU-agnostic programming platform for hybrid quantum/classical computing, with GPU-accelerated simulation and integrations across hardware modalities. | QuantumForge has optional GPU/CUDA paths for selected simulator operations, but not CUDA-Q’s hybrid programming model or QPU ecosystem. Avoid claiming equivalent acceleration or hardware reach. |
| **QuantumForge** | Local quantum simulation and research in Rust, with Python bindings, state-vector and density-matrix simulation, tensor-network methods, QEC, variational algorithms, symbolic tools, and an evolving circuit optimizer. | The breadth is useful for exploration but is not yet a focused commercial product. The best initial wedge to validate is repeatable, noise-aware comparison of small and medium circuit experiments. |

Qiskit documents its target-aware compilation stages and describes routing as a hard optimization problem; TKET documents gate-set and architecture predicates plus cost-directed compilation. Those are the benchmarks to beat on compilation quality and validation—not claims to make lightly. [Qiskit transpiler stages](https://qiskit.qotlabs.org/docs/guides/transpiler-stages), [TKET compilation passes](https://docs.quantinuum.com/tket/user-guide/examples/circuit_compilation/compilation_example.html).

## What is real in QuantumForge now

- Local CPU quantum simulation; density matrices and explicit noise channels; optional acceleration for selected paths.
- Rust API, CLI/REPL, and a Python binding surface.
- Algorithm and research modules including VQE/QAOA, tensor networks, QEC, and symbolic tools.
- Circuit rewrites, basic topology routing, device-constraint validation, limited directed-CNOT repair, and small-circuit equivalence checking.
- A T1/T2 thermal model that can add gate noise and, given caller-provided operation timings, thermal idle noise.
- An `optimize --thermal` CLI path that puts optimization metrics and an exact small-system thermal simulation into one JSON report.
- An optional Qiskit comparison script wired into CI. A passing CI result is evidence for its tested subset only, not broad compiler parity.

The current optimizer does **not** perform general initial layout search, native-basis decomposition, calibration-aware scheduling, or cloud hardware submission. The provider modules in `src/hardware.rs` are scaffolds. Do not present them as working integrations.

## Proposed initial customer and job

**Hypothesis to test:** quantum algorithm researchers and small industrial R&D teams need a repeatable way to answer “did this circuit rewrite actually improve the experiment under our target and noise assumptions?” before spending scarce QPU time.

**First workflow:** import or construct a circuit → compare a named set of compiler variants → apply a versioned device/noise profile → inspect resource and observable changes with uncertainty → save a reproducible report and source artifacts.

This workflow is only partly implemented. The optimizer can now run one declared T1/T2 thermal model on its optimized circuit and add probability, per-qubit Z, purity, and total-variation results to its report for circuits up to eight qubits. It still lacks versioned calibration profiles, arbitrary scheduled noise in the CLI, parameter/shot uncertainty, and comparison of multiple compiler candidates in one run. A full experiment bundle with those features would be a product step, not just a documentation change.

## Commercial model to validate

Keep the Apache-2.0 core as the adoption channel. Test whether teams will pay for one or more of:

1. A team workspace for versioned device profiles, experiment comparisons, reproducible reports, and collaboration.
2. Private deployment/support for organizations that cannot send circuits or calibration data to a hosted service.
3. Paid integrations or validation packages for specific hardware providers and research pipelines.

Do not build a hosted platform or claim demand until interviews and design partners confirm which job is painful enough to fund. Generic access to a quantum simulator is already widely available; a paid offer needs saved QPU spend, reduced engineering time, or stronger evidence quality that can be measured.

## Proof an investor or professional user will expect

1. **A focused use case:** interview algorithm teams and select one repeated workflow with a costly current workaround.
2. **A fair benchmark:** compare QuantumForge with Qiskit and/or TKET on public circuits under the same topology, gate set, objective, seed policy, and hardware assumptions. Report compile time, two-qubit count, depth, equivalence result, and noise/shot uncertainty. Include failures and baselines.
3. **A complete vertical slice:** versioned target profile → optimization → validated output → noise-aware result estimate → machine-readable report. Clearly distinguish measured data from model output.
4. **External use:** design partners run their own circuits, reproduce results, and return to the product. Track activation and repeat usage; seek a paid pilot before projecting revenue.
5. **Operational credibility:** supported platforms, reproducible release artifacts, security/privacy posture, API stability, and a maintainer plan.

## Suggested next build order

1. Expand the current optimization-plus-thermal report into a versioned experiment bundle with optional caller schedules, explicit input/output fingerprints, and multiple comparable compiler candidates.
2. Add calibration-file import and versioning; start with provider-neutral data rather than promising live hardware access.
3. Implement and benchmark general native-gate translation and layout search, then compare results against the established compilers.
4. Add one provider adapter only after users identify a target and the end-to-end compilation contract is reliable.
5. Prototype a team workflow only after design partners confirm a recurring collaboration problem.

## What an initial investment should prove

Treat funding as a way to retire product risk, not as a reason to add more unrelated quantum features. A credible first financing plan should reserve effort for:

- buyer interviews and a small number of design partners around one concrete workflow;
- compiler/noise benchmark engineering with reproducible public baselines;
- one complete user journey from circuit input to a report that another scientist can reproduce;
- a paid pilot or other direct evidence of willingness to pay;
- documentation, release quality, and a support model that let an external team adopt the toolkit.

The amount, valuation, and fundraising timeline cannot be estimated from this codebase. Build the evidence and budget from the team, customer conversations, and pilot scope rather than inventing a top-down market share or revenue forecast.

## Primary references

- IBM Quantum, [Transpiler stages](https://qiskit.qotlabs.org/docs/guides/transpiler-stages) and [Introduction to transpilation](https://qiskit.qotlabs.org/docs/guides/transpile).
- Quantinuum, [TKET compilation passes](https://docs.quantinuum.com/tket/user-guide/examples/circuit_compilation/compilation_example.html) and [Running on backends](https://docs.quantinuum.com/tket/user-guide/manual/manual_backend.html).
- AWS, [Constructing circuits and noise simulation](https://docs.aws.amazon.com/braket/latest/developerguide/braket-constructing-circuit.html) and [How Braket works](https://docs.aws.amazon.com/braket/latest/developerguide/braket-how-it-works.html).
- NVIDIA, [CUDA-Q platform overview](https://developer.nvidia.com/qoda).
