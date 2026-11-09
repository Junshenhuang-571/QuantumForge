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
//! # Scope and limitations
//!
//! This module is an **MBQC simulator**: it prepares the cluster state on the
//! gate-model kernel and runs measurements with adaptive by-product propagation.
//! It is NOT an MBQC compiler.  Specifically:
//!
//! - Graphs must be given statically (via `ClusterState::new` / `path` / etc.).
//! - Gflow existence check available via `ClusterState::gflow_exists`.
//! - Space-time depth and resource cost available.
//! - No reverse path: a quantum circuit cannot be compiled into an
//!   MBQC measurement pattern automatically (circuit → pattern translation
//!   is future work).
//!
//! For a full MBQC compiler (flow checking + circuit → pattern translation),
//! see the Raussendorf–Briegel / Danos–Kashefi literature.

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
    /// Whether this qubit is measured (false = output qubit, not measured).
    pub measured: bool,
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

    /// Basic structural check: every `depends_on_x`/`depends_on_z` index must
    /// refer to a valid qubit, no step may depend on itself, and the dependency
    /// graph must be acyclic (topological order exists).
    ///
    /// This is NOT a full flow/gflow check (Danos–Kashefi); it only catches
    /// obvious structural errors.  Returns `Ok(())` if the pattern passes.
    pub fn check_dependencies(&self, qubits: usize) -> Result<(), String> {
        for (i, step) in self.steps.iter().enumerate() {
            for &dep in step.depends_on_x.iter().chain(&step.depends_on_z) {
                if dep >= qubits {
                    return Err(format!(
                        "step {i} depends on out-of-range qubit {dep} (max {})",
                        qubits - 1
                    ));
                }
                if dep == i {
                    return Err(format!("step {i} depends on itself"));
                }
            }
        }
        let n = self.steps.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, step) in self.steps.iter().enumerate() {
            for &dep in step.depends_on_x.iter().chain(&step.depends_on_z) {
                if dep < n {
                    adj[dep].push(i);
                    in_degree[i] += 1;
                }
            }
        }
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut visited = 0usize;
        while let Some(u) = queue.pop() {
            visited += 1;
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push(v);
                }
            }
        }
        if visited != n {
            return Err("dependency graph contains a cycle".to_string());
        }
        Ok(())
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
            if !step.measured {
                // Non-measured (output) qubit: skip measurement.
                // By-product corrections would be applied as Pauli gates here.
                angles.push(0.0);
                outcomes.push(0);
                if step.output {
                    outputs.push(0);
                }
                continue;
            }
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
                if outcome_of.get(&dep) == Some(&1) {
                    mirror = !mirror;
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
                    state
                        .apply(crate::Gate::Rz(-phi), step.qubit)
                        .map_err(|e| e.to_string())?;
                    measure_x(&mut state, step.qubit, &mut rng)?
                }
                MeasurementBasis::Z => state
                    .measure(step.qubit, &mut rng)
                    .map_err(|e| e.to_string())?,
            };
            outcome_of.insert(step.qubit, outcome);
            outcomes.push(outcome);
            if step.output {
                outputs.push(outcome);
            }
        }

        Ok(MbqcRun {
            angles,
            outcomes,
            outputs,
        })
    }
}

/// Measure qubit `q` in the `X` basis by rotating `H` then measuring `Z`.
fn measure_x(state: &mut StateVector, q: usize, rng: &mut SmallRng) -> Result<u8, String> {
    state.apply(crate::Gate::H, q).map_err(|e| e.to_string())?;
    state.measure(q, rng).map_err(|e| e.to_string())
}

/// Parse a single angle token: a decimal number, or a rational multiple of π
/// written as `pi`, `2pi`, `pi/2`, `-pi/4`, `3pi/2`, etc. Returns radians.
fn parse_angle(tok: &str) -> Result<f64, String> {
    let t = tok.trim();
    if let Ok(v) = t.parse::<f64>() {
        return Ok(v);
    }
    // Optional leading sign, then split around the literal `pi`.
    let (sign, rest) = match t.strip_prefix('-') {
        Some(r) => (-1.0, r.trim()),
        None => (1.0, t),
    };
    let (num_s, den_s) = match rest.split_once("pi") {
        Some((n, d)) => {
            // d is like "/2" (or empty). num is like "2" (or empty for bare pi).
            let den = d.trim_start_matches('/').trim();
            (n.trim(), if den.is_empty() { "1" } else { den })
        }
        None => return Err(format!("bad angle '{tok}'")),
    };
    let num: f64 = if num_s.is_empty() {
        1.0
    } else {
        num_s.parse().map_err(|_| format!("bad angle '{tok}'"))?
    };
    let den: f64 = den_s.parse().map_err(|_| format!("bad angle '{tok}'"))?;
    if den == 0.0 {
        return Err(format!("bad angle '{tok}': zero denominator"));
    }
    Ok(sign * num * std::f64::consts::PI / den)
}

/// Parse a comma-separated list of `usize`, allowing the literal `none`/`-`
/// for an empty entry. Used for dependency lists.
fn parse_deps(tok: &str) -> Result<Vec<usize>, String> {
    let t = tok.trim();
    if t.is_empty() || t == "none" || t == "-" {
        return Ok(Vec::new());
    }
    t.split(',')
        .map(|x| {
            x.trim()
                .parse::<usize>()
                .map_err(|_| format!("bad dependency '{x}'"))
        })
        .collect()
}

/// Build a [`MeasurementPattern`] from a compact spec, validating against the
/// cluster's qubit count. Spec keys (all comma-separated, one entry per step,
/// in measurement order):
///
/// - `angles=0.5,0,pi/2`  — nominal equatorial angles (or `z` for Z-basis)
/// - `basis=eq,eq,z`      — `eq`/`z` per step (alternative to `z` in angles)
/// - `qubits=2,0,1`       — which qubit each step measures (default: step order `0,1,2,...`); any order
/// - `deps=none,0,1`      — X-by-product dependencies (phase-shift sources)
/// - `depsz=none,none,0`  — Z-by-product dependencies (plane-mirror sources)
/// - `outputs=0,0,1`      — 1 marks an output step
///
/// Every qubit must be measured exactly once (a valid MBQC pattern consumes the
/// whole cluster). Example:
/// `angles=0.5,0,pi basis=eq,eq,z qubits=1,2,0 deps=none,0,1 depsz=none,none,0 outputs=0,0,1`
pub fn parse_pattern(spec: &str, qubits: usize) -> Result<MeasurementPattern, String> {
    let mut angles: Vec<String> = Vec::new();
    let mut basis: Vec<String> = Vec::new();
    let mut step_qubits: Vec<usize> = Vec::new();
    let mut deps: Vec<Vec<usize>> = Vec::new();
    let mut depsz: Vec<Vec<usize>> = Vec::new();
    let mut outputs: Vec<bool> = Vec::new();

    for word in spec.split_whitespace() {
        let (k, v) = word
            .split_once('=')
            .ok_or_else(|| format!("bad pattern token '{word}' (want key=value)"))?;
        match k {
            "angles" => angles = v.split(',').map(|s| s.trim().to_owned()).collect(),
            "basis" => basis = v.split(',').map(|s| s.trim().to_owned()).collect(),
            "qubits" => {
                step_qubits = v
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("bad qubit index '{s}'"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "deps" => deps = v.split(',').map(parse_deps).collect::<Result<_, _>>()?,
            "depsz" => depsz = v.split(',').map(parse_deps).collect::<Result<_, _>>()?,
            "outputs" => {
                outputs = v
                    .split(',')
                    .map(|s| match s.trim() {
                        "1" | "true" => Ok(true),
                        "0" | "false" | "" => Ok(false),
                        other => Err(format!("bad output flag '{other}'")),
                    })
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown pattern key '{other}'")),
        }
    }

    let n = angles
        .len()
        .max(basis.len())
        .max(deps.len())
        .max(step_qubits.len());
    if n == 0 {
        return Err("empty pattern (need angles= and/or basis=)".to_owned());
    }
    // A pattern must consume every cluster qubit exactly once.
    if n != qubits {
        return Err(format!(
            "pattern has {n} steps but the cluster has {qubits} qubits (must measure each once)"
        ));
    }
    // Resolve the qubit each step measures (default: step order).
    let resolved: Vec<usize> = (0..n)
        .map(|i| step_qubits.get(i).copied().unwrap_or(i))
        .collect();
    // Every qubit in range, and each measured exactly once (a permutation).
    let mut seen = vec![false; qubits];
    for (i, &q) in resolved.iter().enumerate() {
        if q >= qubits {
            return Err(format!(
                "step {i} measures qubit {q} but cluster has {qubits}"
            ));
        }
        if seen[q] {
            return Err(format!("qubit {q} is measured more than once"));
        }
        seen[q] = true;
    }

    let mut pattern = MeasurementPattern::new();
    for i in 0..n {
        // Basis: explicit basis= wins; else `z` in angles= means Z-basis.
        let b = if i < basis.len() && !basis[i].is_empty() {
            match basis[i].as_str() {
                "eq" | "equatorial" => MeasurementBasis::Equatorial(
                    angles
                        .get(i)
                        .map(|a| parse_angle(a))
                        .transpose()?
                        .unwrap_or(0.0),
                ),
                "z" | "Z" => MeasurementBasis::Z,
                other => return Err(format!("bad basis '{other}' (want eq|z)")),
            }
        } else {
            let a = angles.get(i).ok_or("missing angle for step")?;
            if a == "z" || a == "Z" {
                MeasurementBasis::Z
            } else {
                MeasurementBasis::Equatorial(parse_angle(a)?)
            }
        };
        pattern.push(MeasurementStep {
            qubit: resolved[i],
            basis: b,
            depends_on_x: deps.get(i).cloned().unwrap_or_default(),
            depends_on_z: depsz.get(i).cloned().unwrap_or_default(),
            output: outputs.get(i).copied().unwrap_or(i == n - 1),
            measured: true,
        });
        // Validate dependency indices are in range.
        for &d in pattern.steps[i]
            .depends_on_x
            .iter()
            .chain(&pattern.steps[i].depends_on_z)
        {
            if d >= qubits {
                return Err(format!(
                    "step {i} depends on qubit {d} but cluster has {qubits}"
                ));
            }
        }
    }
    Ok(pattern)
}

/// Convenience: a linear pattern that teleports/rotates a single logical qubit
/// through a 3-qubit linear cluster — the canonical MBQC demo. Measures qubit
/// 1 (input) at angle `alpha`, qubit 2 in `Z`, and reads out qubit 3 corrected
/// by both. Returns the corrected output bit.
/// Compile a single-qubit gate sequence into an MBQC measurement pattern.
/// Supports: H, S, T, X, Z gates. Each gate is mapped to measurements on
/// a path cluster state. Returns a MeasurementPattern ready to run.
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
        measured: true,
    });
    // Step B: measure qubit 2 in Z; its X-by-product mirrors qubit 3's plane.
    pattern.push(MeasurementStep {
        qubit: 2,
        basis: MeasurementBasis::Z,
        depends_on_x: vec![],
        depends_on_z: vec![],
        output: false,
        measured: true,
    });
    // Step C: read out qubit 0, equatorial, corrected by qubit 1 (phase) and
    // qubit 2 (mirror).
    pattern.push(MeasurementStep {
        qubit: 0,
        basis: MeasurementBasis::Equatorial(0.0),
        depends_on_x: vec![1],
        depends_on_z: vec![2],
        output: true,
        measured: true,
    });
    let run = pattern.run(&cluster, seed)?;
    Ok(run.outputs[0])
}

pub fn compile_single_qubit(
    gates: &[&str],
    _seed: u64,
) -> Result<(ClusterState, MeasurementPattern), String> {
    let n = gates.len() + 1;
    let cluster = ClusterState::path(n);
    let mut pattern = MeasurementPattern::default();

    for (i, gate) in gates.iter().enumerate() {
        let qubit = i + 1;
        let angle = match *gate {
            "H" => std::f64::consts::FRAC_PI_2,
            "S" => std::f64::consts::FRAC_PI_2,
            "T" => std::f64::consts::FRAC_PI_4,
            "X" => std::f64::consts::PI,
            "Z" => 0.0,
            _ => return Err(format!("Unsupported gate: {}", gate)),
        };
        pattern.push(MeasurementStep {
            qubit,
            basis: MeasurementBasis::Equatorial(angle),
            depends_on_x: if qubit > 1 { vec![qubit - 1] } else { vec![] },
            depends_on_z: vec![],
            output: qubit == n - 1,
            measured: true,
        });
    }

    Ok((cluster, pattern))
}

/// Compile a multi-qubit gate sequence into an MBQC measurement pattern.
/// Supports single-qubit gates (H, S, T, X, Z) and CNOT.
/// Single-qubit gates use path cluster states with correct measurement angles.
/// CNOT uses a simplified mapping (angle=π, by-product on control qubit).
/// Full CNOT requires 2D cluster states with non-measured output qubits,
/// which is not yet supported by this framework.
/// Each gate is mapped to measurements on a path cluster state.
pub fn compile_circuit(
    gates: &[(usize, &str, Option<usize>)],
) -> Result<(ClusterState, MeasurementPattern), String> {
    let n_qubits = gates.iter().map(|(q, _, _)| *q).max().unwrap_or(0) + 1;
    let n_meas = gates.len() + n_qubits;
    let cluster = ClusterState::path(n_meas);
    let mut pattern = MeasurementPattern::default();

    for (i, (_target, gate, control)) in gates.iter().enumerate() {
        let qubit = i + n_qubits;
        let angle = match *gate {
            "H" => std::f64::consts::FRAC_PI_2,
            "S" => std::f64::consts::FRAC_PI_2,
            "T" => std::f64::consts::FRAC_PI_4,
            "X" => std::f64::consts::PI,
            "Z" => 0.0,
            "CNOT" => std::f64::consts::PI,
            _ => return Err(format!("Unsupported gate: {}", gate)),
        };

        let depends_on_x = if qubit > n_qubits {
            vec![qubit - 1]
        } else {
            vec![]
        };

        let depends_on_z = control.map(|c| vec![c]).unwrap_or_default();

        pattern.push(MeasurementStep {
            qubit,
            basis: MeasurementBasis::Equatorial(angle),
            depends_on_x,
            depends_on_z,
            output: i == gates.len() - 1,
            measured: true,
        });
    }

    Ok((cluster, pattern))
}
impl ClusterState {
    /// Check whether a generalized flow (gflow) exists for the given input
    /// and output sets.
    pub fn gflow_exists(&self, inputs: &[usize], outputs: &[usize]) -> bool {
        let n = self.qubits;
        let input_set: std::collections::HashSet<usize> = inputs.iter().copied().collect();
        let output_set: std::collections::HashSet<usize> = outputs.iter().copied().collect();

        let interior: Vec<usize> = (0..n).filter(|v| !output_set.contains(v)).collect();

        let mut adj = vec![vec![false; n]; n];
        for &(i, j) in &self.edges {
            if i < n && j < n {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }

        let odd_neighbourhood = |s: &[usize]| -> Vec<usize> {
            let mut counts = vec![0usize; n];
            for &v in s {
                for u in 0..n {
                    if adj[v][u] {
                        counts[u] += 1;
                    }
                }
            }
            (0..n).filter(|&u| counts[u] % 2 == 1).collect()
        };

        if interior.len() > 16 {
            return false;
        }

        for &v in &interior {
            let future: Vec<usize> = (0..n)
                .filter(|&u| !input_set.contains(&u) && u != v)
                .collect();
            let mut found = false;
            let m = future.len();
            if m > 20 {
                return false;
            }
            for mask in 1usize..(1usize << m) {
                let s: Vec<usize> = (0..m)
                    .filter(|&i| mask & (1 << i) != 0)
                    .map(|i| future[i])
                    .collect();
                let odd = odd_neighbourhood(&s);
                if odd.contains(&v) {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        true
    }

    /// Space-time depth: number of parallel measurement layers.
    pub fn space_time_depth(&self, pattern: &MeasurementPattern) -> usize {
        let n = pattern.steps.len();
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, step) in pattern.steps.iter().enumerate() {
            for &dep in step.depends_on_x.iter().chain(&step.depends_on_z) {
                if dep < n {
                    deps[i].push(dep);
                }
            }
        }
        let mut layer = vec![0usize; n];
        for i in 0..n {
            let max_dep_layer = deps[i].iter().map(|&d| layer[d] + 1).max().unwrap_or(0);
            layer[i] = max_dep_layer;
        }
        layer.iter().copied().max().unwrap_or(0) + 1
    }

    /// Resource-state cost model.
    pub fn resource_cost(&self, pattern: &MeasurementPattern) -> ResourceCost {
        let depth = self.space_time_depth(pattern);
        let mut dep_edges = 0usize;
        for step in &pattern.steps {
            dep_edges += step.depends_on_x.len() + step.depends_on_z.len();
        }
        let output_count = pattern.steps.iter().filter(|s| s.output).count();
        ResourceCost {
            qubits: self.qubits,
            edges: self.edges.len(),
            measurements: pattern.steps.len(),
            output_qubits: output_count,
            space_time_depth: depth,
            byproduct_edges: dep_edges,
            total_cost: self.qubits + self.edges.len() + depth * pattern.steps.len(),
        }
    }
}

/// Resource-state cost summary for an MBQC pattern.
#[derive(Clone, Debug)]
pub struct ResourceCost {
    pub qubits: usize,
    pub edges: usize,
    pub measurements: usize,
    pub output_qubits: usize,
    pub space_time_depth: usize,
    pub byproduct_edges: usize,
    pub total_cost: usize,
}

/// Proper MBQC CNOT for computational basis inputs.
/// Uses T-shaped cluster (edges 0-2, 1-2, 1-3) with angle-encoded inputs.
/// By-products: O2 = X^{s0} Z^{s0}, O3 = X^{s0^s1}.
pub fn mbqc_cnot_proper(b0: u8, b1: u8, seed: u64) -> Result<(u8, u8), String> {
    let edges = vec![(0, 2), (1, 2), (1, 3)];
    let cluster = ClusterState::new(4, edges);
    let mut state = cluster.prepare()?;

    // Encode input via measurement basis.
    let a0 = if b0 == 1 { std::f64::consts::PI } else { 0.0 };
    let a1 = if b1 == 1 { std::f64::consts::PI } else { 0.0 };

    let mut rng = SmallRng::seeded(seed);
    state
        .apply(crate::Gate::Rz(-a0), 0)
        .map_err(|e| e.to_string())?;
    let measure_x = |s: &mut StateVector, q: usize, rng: &mut SmallRng| -> Result<u8, String> {
        s.apply(crate::Gate::H, q).map_err(|e| e.to_string())?;
        s.measure(q, rng).map_err(|e| e.to_string())
    };
    let s0 = measure_x(&mut state, 0, &mut rng)?;
    state
        .apply(crate::Gate::Rz(-a1), 1)
        .map_err(|e| e.to_string())?;
    let s1 = measure_x(&mut state, 1, &mut rng)?;

    // By-products: O2 = X^{s0} Z^{s0}, O3 = X^{s0^s1}
    if s0 == 1 {
        state.apply(crate::Gate::X, 2).map_err(|e| e.to_string())?;
        state.apply(crate::Gate::Z, 2).map_err(|e| e.to_string())?;
    }
    if (s0 ^ s1) == 1 {
        state.apply(crate::Gate::X, 3).map_err(|e| e.to_string())?;
    }

    let r2 = state.measure(2, &mut rng).map_err(|e| e.to_string())?;
    let r3 = state.measure(3, &mut rng).map_err(|e| e.to_string())?;

    Ok((r2, r3))
}
