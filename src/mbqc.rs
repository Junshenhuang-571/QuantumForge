//! Measurement-Based Quantum Computing (MBQC) — the "one-way" model.
//!
//! MBQC computes entirely by preparing a entangled **cluster state** on a graph
//! and then measuring each qubit in a adaptively chosen basis. The measurement
//! outcomes are random, but each outcome `s ∈ {0,1}` introduces a **by-product
//! Pauli correction** `X^s`/`Z^s` that propagates forward and shifts the basis
//! of every later measurement that depends on it. This module gives you:
//!
//! 1. [`ClusterState`]: build a graph state — every qubit starts in `|+>` and
//!    each graph edge applies a `CZ` — over the gate-model kernel.
//! 2. [`MeasurementPattern`]: a dependency-ordered list of measurements, each
//!    with a nominal angle plus the input/output/dependent corrections that
//!    define how by-products flow. The runner simulates the adaptive choices
//!    and returns the corrected (deterministic-up-to-input) output bits.
//!
//! This is the Raussendorf–Briegel model restricted to a fixed graph with
//! feed-forward, which is Turing-complete and equivalent to the circuit model.

use crate::{SmallRng, StateVector};

/// A cluster (graph) state: `n` qubits, each a graph vertex; each edge `(i, j)`
/// means a `CZ(i, j)` was applied after putting every qubit in `|+>`.
#[derive(Clone, Debug)]
pub struct ClusterState {
    /// Number of qubits (vertices).
    pub qubits: usize,
    /// Undirected edges as `(i, j)` with `i < j`.
    pub edges: Vec<(usize, usize)>,
}

impl ClusterState {
    /// Build a cluster state on `qubits` vertices with the given edges.
    pub fn new(qubits: usize, edges: Vec<(usize, usize)>) -> Self {
        Self { qubits, edges }
    }

    /// Common 1-D cluster: a path graph `0-1-2-...-(n-1)`.
    pub fn path(qubits: usize) -> Self {
        let edges = (0..qubits.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        Self::new(qubits, edges)
    }

    /// Prepare the cluster state as a `StateVector`: all qubits in `|+>`, then a
    /// `CZ` per edge. (CZ is symmetric, so edge orientation is irrelevant.)
    pub fn prepare(&self) -> Result<StateVector, String> {
        let mut state = StateVector::zero(self.qubits).map_err(|e| e.to_string())?;
        for q in 0..self.qubits {
            state.apply(crate::Gate::H, q).map_err(|e| e.to_string())?; // |0> -> |+>
        }
        for &(i, j) in &self.edges {
            state.apply_cz(i, j).map_err(|e| e.to_string())?;
        }
        Ok(state)
    }

    /// The set of graph neighbours of qubit `q` (its dependency set for
    /// by-product propagation).
    pub fn neighbours(&self, q: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for &(i, j) in &self.edges {
            if i == q {
                out.push(j);
            } else if j == q {
                out.push(i);
            }
        }
        out
    }
}

/// The plane a qubit is measured in. MBQC measurements are equatorial
/// (`XY`-plane, angle `φ`) or Pauli-`Z`/`X`; we generalise to an arbitrary
/// angle `φ` measuring `cos(φ) X + sin(φ) Y`, plus the special `Z` basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeasurementBasis {
    /// Equatorial: measure `cos(φ)·X + sin(φ)·Y`. Outcome `s` applies by-product
    /// `Z^s` (a phase flip) that propagates as an angle shift to neighbours.
    Equatorial(f64),
    /// Measure in the computational (`Z`) basis. Outcome `s` applies by-product
    /// `X^s` to neighbours.
    Z,
}

/// One measurement step in a pattern.
#[derive(Clone, Debug)]
pub struct MeasurementStep {
    /// Qubit to measure.
    pub qubit: usize,
    /// Nominal basis (before by-product correction).
    pub basis: MeasurementBasis,
    /// Qubits whose `X` by-products shift this measurement's `Z` angle
    /// (i.e. earlier measurements whose outcome flips this one's phase).
    pub depends_on_x: Vec<usize>,
    /// Qubits whose `Z` by-products flip this measurement's plane
    /// (earlier measurements whose outcome mirrors X<->Y here).
    pub depends_on_z: Vec<usize>,
    /// If set, this step's (corrected) outcome is an output bit.
    pub output: bool,
}

/// Result of running a pattern.
#[derive(Clone, Debug)]
pub struct MbqcRun {
    /// The corrected measurement angles actually used (after by-product shifts).
    pub angles: Vec<f64>,
    /// Raw measurement outcomes `s_k ∈ {0,1}` for every step.
    pub outcomes: Vec<u8>,
    /// Output bits (only steps flagged `output`), in step order.
    pub outputs: Vec<u8>,
}

/// A measurement pattern over a [`ClusterState`].
#[derive(Clone, Debug, Default)]
pub struct MeasurementPattern {
    pub steps: Vec<MeasurementStep>,
}

impl MeasurementPattern {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, step: MeasurementStep) {
        self.steps.push(step);
    }

    /// Execute the pattern on a freshly prepared cluster state with a seeded
    /// RNG. Returns the corrected angles, raw outcomes, and output bits.
    ///
    /// By-product bookkeeping (Raussendorf–Briegel):
    /// - an equatorial measurement with outcome `s` contributes `Z^s`, which
    ///   *shifts the angle* `φ → φ + π·s` of every later step that depends on it
    ///   via `depends_on_x`;
    /// - a `Z`-basis measurement with outcome `s` contributes `X^s`, which
    ///   *mirrors the plane* (`φ → −φ`) of every later step that depends on it
    ///   via `depends_on_z`.
    pub fn run(&self, cluster: &ClusterState, seed: u64) -> Result<MbqcRun, String> {
        let mut state = cluster.prepare()?;
        let mut rng = SmallRng::seeded(seed);

        let mut angles = Vec::with_capacity(self.steps.len());
        let mut outcomes = Vec::with_capacity(self.steps.len());
        let mut outputs = Vec::new();
        // outcome[q] = last measured bit for qubit q (for by-product lookup).
        let mut outcome_of = std::collections::HashMap::<usize, u8>::new();

        for step in &self.steps {
            // Accumulate the corrected angle from X-by-products (phase shifts).
            let mut phi = match step.basis {
                MeasurementBasis::Equatorial(phi) => phi,
                MeasurementBasis::Z => 0.0,
            };
            for &dep in &step.depends_on_x {
                if let Some(&s) = outcome_of.get(&dep) {
                    phi += std::f64::consts::PI * s as f64;
                }
            }
            // Z-by-products mirror the plane (X <-> Y), i.e. negate the angle.
            let mut mirror = false;
            for &dep in &step.depends_on_z {
                if let Some(&s) = outcome_of.get(&dep) {
                    if s == 1 {
                        mirror = !mirror;
                    }
                }
            }
            if mirror {
                phi = -phi;
            }

            angles.push(phi);

            // Rotate into the measurement frame and measure.
            let outcome = match step.basis {
                MeasurementBasis::Equatorial(_) => {
                    // Measure cos(phi) X + sin(phi) Y: rotate by -phi about Z so
                    // the observable aligns with X, then measure X.
                    state.apply(crate::Gate::Rz(-phi), step.qubit).map_err(|e| e.to_string())?;
                    measure_x(&mut state, step.qubit, &mut rng)?
                }
                MeasurementBasis::Z => state.measure(step.qubit, &mut rng).map_err(|e| e.to_string())?,
            };
            outcome_of.insert(step.qubit, outcome);
            outcomes.push(outcome);
            if step.output {
                outputs.push(outcome);
            }
        }

        Ok(MbqcRun { angles, outcomes, outputs })
    }
}

/// Measure qubit `q` in the `X` basis by rotating `H` then measuring `Z`.
fn measure_x(state: &mut StateVector, q: usize, rng: &mut SmallRng) -> Result<u8, String> {
    state.apply(crate::Gate::H, q).map_err(|e| e.to_string())?;
    state.measure(q, rng).map_err(|e| e.to_string())
}

/// Convenience: a linear pattern that teleports/rotates a single logical qubit
/// through a 3-qubit linear cluster — the canonical MBQC demo. Measures qubit
/// 1 (input) at angle `alpha`, qubit 2 in `Z`, and reads out qubit 3 corrected
/// by both. Returns the corrected output bit.
pub fn teleport_through_path(alpha: f64, seed: u64) -> Result<u8, String> {
    let cluster = ClusterState::path(3);
    let mut pattern = MeasurementPattern::new();
    // Step A: measure qubit 1 (input) in equatorial basis at angle alpha.
    pattern.push(MeasurementStep {
        qubit: 1,
        basis: MeasurementBasis::Equatorial(alpha),
        depends_on_x: vec![],
        depends_on_z: vec![],
        output: false,
    });
    // Step B: measure qubit 2 in Z; its X-by-product mirrors qubit 3's plane.
    pattern.push(MeasurementStep {
        qubit: 2,
        basis: MeasurementBasis::Z,
        depends_on_x: vec![],
        depends_on_z: vec![],
        output: false,
    });
    // Step C: read out qubit 0, equatorial, corrected by qubit 1 (phase) and
    // qubit 2 (mirror).
    pattern.push(MeasurementStep {
        qubit: 0,
        basis: MeasurementBasis::Equatorial(0.0),
        depends_on_x: vec![1],
        depends_on_z: vec![2],
        output: true,
    });
    let run = pattern.run(&cluster, seed)?;
    Ok(run.outputs[0])
}
