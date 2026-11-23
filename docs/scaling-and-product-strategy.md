# Scaling beyond exact circuit simulation

## The honest answer to “infinite size”

No classical method can exactly simulate every arbitrary quantum circuit at
unbounded size and depth. A general pure-state vector needs `2^n` amplitudes;
entanglement, non-Clifford content, connectivity, noise, and the requested
output all affect the cost. A useful product should select among methods that
fit a circuit's structure, report approximation and sampling error, and stop
with a useful explanation when the requested accuracy exceeds the budget.

## A portfolio of simulation methods

| Circuit structure or question | Suitable method | Scaling lever | Important limit |
|---|---|---|---|
| Small, general circuit; full state required | State vector | Straightforward exact reference | Memory grows as `2^n` |
| Clifford circuit | Stabilizer tableau | Track Pauli generators instead of amplitudes | General non-Clifford gates leave this efficient class |
| Long 1D circuit with modest entanglement | MPS / TEBD | Cap bond dimension and track discarded weight | Cost rises rapidly when entanglement spreads; truncation is approximate |
| Noisy circuit and expectation estimates | Quantum trajectories | Average pure-state trajectories, report confidence intervals | More trajectories cost more; noise alone does not guarantee easy simulation |
| Low-width tensor-network geometry | Contraction / tensor network | Choose a contraction order and exploit small separators | Contraction width can grow exponentially with graph width |
| Circuits that can be split into weakly coupled pieces | Circuit cutting | Simulate smaller fragments and combine estimators | Sampling and recombination overhead can grow exponentially in the number and type of cuts |

These methods complement each other; none makes arbitrary quantum computation
free. Vidal's MPS result connects efficient simulation to restricted
entanglement. Stabilizer-decomposition methods trade runtime for non-Clifford
rank. Circuit cutting lowers fragment size at a sampling cost. Recent work on
noisy circuits proves efficient tensor-network bounds only under specific
noise, geometry, and error assumptions; it is a research direction, not a
blanket “noise makes simulation easy” rule. See the primary references below.

## What QuantumForge supports today

- Exact state-vector and density-matrix simulation for small workloads.
- A stabilizer tableau for Clifford operations.
- MPS, TEBD, DMRG, and MPS-based QAOA, with explicit bond caps in the supported
  workflows.
- Seeded trajectory support for selected noise paths.
- MPS single- and two-site expectations plus DMRG normalization that use
  transfer contractions instead of expanding the full `2^n` state vector.

This is not yet a general circuit-to-MPS compiler or an automatic simulator
router. The library still has MPS convenience methods that deliberately expand
to a state vector, and not every gate, measurement, and noise channel is wired
through one scalable interface. Users should check which API path their
workload invokes.

## The product step that makes this a platform

Build one documented `simulate_auto` workflow with an explicit memory budget,
accuracy target, seed, and requested result (state, samples, or observable):

1. Classify supported operations, connectivity, Clifford content, and circuit
   width/depth.
2. Estimate state-vector memory and MPS bond growth; reject unsafe allocations
   before they happen.
3. Route only to a method whose operation set and requested output it supports:
   exact vector, stabilizer, MPS, or sampled trajectories.
4. Return provenance with method, seed, time/memory counters, bond dimensions,
   truncation diagnostics, and confidence intervals where sampling is used.
5. Fail closed if the budget/accuracy target cannot be met; never silently
   replace an exact answer with an approximation.
6. Cross-check small instances against the exact backend and publish the
   benchmark circuits, machine details, and tolerances.

The next concrete milestones are MPS measurement sampling, truncation-weight
accounting, a nearest-neighbor unitary circuit runner, and cross-validation
against the existing state-vector path. Only after those are solid should a
trajectory backend and an automatic router expose a stable public API.

## Product and revenue hypothesis

The open-source Rust crate can be the adoption channel. A paid product should
solve a recurring team workflow, such as comparing methods against a private
target profile, sharing provenance-complete experiments, running in a private
environment, or receiving support for a validated workflow. These are
hypotheses, not demand evidence. Talk to quantum algorithm and simulation teams
before building hosted collaboration or forecasting revenue; a generic
simulator is not by itself a differentiated paid product.

## Crates.io release preparation

The package already has a description, license, repository, README, keywords,
and categories. Before a new release, update the version once the feature set
and compatibility promise are settled; inspect the package file list, run
Cargo's package dry-run, confirm the docs.rs feature set, and verify that the
README quickstart works for a clean install. Do not publish a version whose
core MPS claims exceed the tested implementation.

## Primary research

- Vidal, [Efficient classical simulation of slightly entangled quantum
  computations](https://arxiv.org/abs/quant-ph/0301063), 2003.
- Bravyi et al., [Simulation of quantum circuits by low-rank stabilizer
  decompositions](https://arxiv.org/abs/1808.00128), 2018.
- Piveteau, Schmitt, and Sutter, [Circuit cutting with classical side
  information](https://arxiv.org/abs/2503.22384), 2025.
- Shao et al., [Complexity of tensor network simulation for noisy quantum
  circuits](https://arxiv.org/abs/2606.00474), 2026.
