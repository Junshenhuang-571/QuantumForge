# Reproducible optimizer experiment

This workflow compares the two built-in rewrite modes against the same circuit
and target, checks both outputs for unitary equivalence, and writes a JSON
artifact containing the input and candidate circuits alongside the metrics.
It is a local simulation workflow; it does not submit jobs to a device.

## Run a comparison

From the repository root, replace the sample expression with your circuit. The
thermal parameters must use one consistent time unit. The values below are an
example model, not a hardware calibration.

```sh
cargo run --release --bin quantumforge -- optimize 2 all --compare --verify --thermal 30 20 1 0.02 0.2 --report experiment.json "Circuit[H[0], CNOT[0,1], H[0], H[0]]"
```

The command writes the selected circuit to standard output and saves
`experiment.json` separately. `--compare` requires `--verify`, so a failed
equivalence check prevents a comparison report from being written. The report
includes the normalized input circuit, selected output circuit, and both
candidate circuits, so the circuits used in the comparison travel with the
measurements.

To compile a file instead, use `--input circuit.qasm` (OpenQASM 2 is supported)
or a file containing a QuantumForge `Circuit[...]` expression. For a target
topology, add `--line` or `--topology target.json`; both candidates use the
same target.

## Read the report

- `experiment_artifacts` stores the normalized input and selected output.
- `circuit_fingerprints` retains the legacy non-cryptographic FNV-1a values for
  quick matching. `artifact_digests` adds SHA-256 over the canonical JSON for
  the normalized input and selected output; use these digests to detect
  accidental artifact changes. Candidate-comparison entries also carry a digest
  for each candidate circuit. A digest does not prove who produced a report.
- `candidate_comparison.selected` and `.alternate` contain each candidate's
  circuit, equivalence result, rewrite counts, resource counts, and optional
  thermal analysis.
- `before` and `after` report structural counts. They do not predict device
  runtime or hardware fidelity.
- `thermal_noise_analysis` is an exact density-matrix simulation from the
  all-zero initial state. The CLI path is limited to unitary circuits of at
  most eight qubits. Its independent Markovian relaxation and dephasing model
  uses the parameters in the report; it does not include readout noise,
  correlated noise, or parameter uncertainty.

The report uses schema version 3. Consumers should check `schema_version` and
tolerate new fields before using reports in a long-lived pipeline. The report
also records the operating system and architecture; performance comparisons
should include the host CPU, compiler build, and run conditions separately.

## Move from a model to hardware evidence

QuantumForge currently has no calibration ingestion, general layout search,
native-basis translation, or hardware execution path. Before comparing
hardware outcomes, use the same circuit set, topology, native gates, objective,
and random-seed policy in established transpilers; publish compile time,
equivalence, two-qubit count, depth, and noise assumptions together. The
[product comparison and roadmap](product-positioning.md) describe the scope and
the research benchmarks that remain.
