//! MPS-based QAOA for diagonal cost Hamiltonians (MaxCut).
//!
//! QuantumForge's state-vector QAOA builders ([`crate::algorithms::qaoa_maxcut`])
//! are exact but need `2^n` amplitudes. This module runs the same QAOA
//! circuit on a matrix product state with a bounded bond dimension, so
//! low-entanglement instances (the practical regime for small `p`) scale to
//! hundreds of qubits.
//!
//! The cost layer uses diagonal two-qubit gates applied through an exact
//! bond-2 MPO ([`MPS::apply_diagonal_two_site`]), and the cost expectation is
//! a transfer contraction ([`MPS::expectation_two_site_product`]) — neither
//! expands the state vector.
//!
//! Cost convention: `H_C = sum_(u,v) w_uv (I - Z_u Z_v) / 2`, so the
//! expectation value is the expected cut weight and the approximation ratio is
//! `<H_C> / maxcut`.

use crate::tensor_network::MPS;
use crate::{Complex64, SmallRng};
use std::time::{Duration, Instant};

/// Pauli-Z as a 2x2 matrix.
const Z: [[Complex64; 2]; 2] = [
    [Complex64::ONE, Complex64::ZERO],
    [Complex64::ZERO, Complex64::new(-1.0, 0.0)],
];

/// Hadamard as a 2x2 matrix.
const H: [[Complex64; 2]; 2] = [
    [
        Complex64::new(0.707_067_811_865_476, 0.0),
        Complex64::new(0.707_067_811_865_476, 0.0),
    ],
    [
        Complex64::new(0.707_067_811_865_476, 0.0),
        Complex64::new(-0.707_067_811_865_476, 0.0),
    ],
];

/// A weighted MaxCut instance.
#[derive(Clone, Debug)]
pub struct MaxCutProblem {
    /// Number of vertices, which is also the number of qubits.
    pub num_qubits: usize,
    /// Edges `(u, v, weight)` with `u < v`.
    pub edges: Vec<(usize, usize, f64)>,
}

impl MaxCutProblem {
    /// Build a problem from a vertex count and an explicit edge list; the edges
    /// need not be sorted.
    pub fn new(num_qubits: usize, edges: Vec<(usize, usize, f64)>) -> Self {
        Self { num_qubits, edges }
    }

    /// Random 3-regular graph (configuration model, self-loops and multi-edges
    /// rejected). Deterministic for a given seed.
    pub fn random_3_regular(num_qubits: usize, seed: u64) -> Self {
        assert!(
            num_qubits >= 4 && num_qubits.is_multiple_of(2),
            "3-regular graphs need an even number of vertices >= 4"
        );
        let mut rng = SmallRng::seeded(seed);
        for _attempt in 0..10_000 {
            let mut stubs: Vec<usize> = (0..num_qubits)
                .flat_map(|v| std::iter::repeat_n(v, 3))
                .collect();
            // Fisher-Yates with the seeded generator.
            for i in (1..stubs.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                stubs.swap(i, j);
            }
            let mut edges = Vec::with_capacity(3 * num_qubits / 2);
            let mut ok = true;
            for pair in stubs.chunks(2) {
                let (a, b) = (pair[0], pair[1]);
                if a == b {
                    ok = false;
                    break;
                }
                let (u, v) = if a < b { (a, b) } else { (b, a) };
                if edges.iter().any(|&(eu, ev, _)| eu == u && ev == v) {
                    ok = false;
                    break;
                }
                edges.push((u, v, 1.0));
            }
            if ok {
                edges.sort_by_key(|a| (a.0, a.1));
                return Self::new(num_qubits, edges);
            }
        }
        panic!("failed to sample a 3-regular graph");
    }

    /// Sum of all edge weights (an upper bound on the cut value).
    pub fn total_weight(&self) -> f64 {
        self.edges.iter().map(|&(_, _, w)| w).sum()
    }

    /// Cut weight of a bitstring (bit `i` = vertex `i`).
    pub fn cut_value(&self, bits: u64) -> f64 {
        let mut total = 0.0;
        for &(u, v, w) in &self.edges {
            let bu = (bits >> u) & 1;
            let bv = (bits >> v) & 1;
            if bu != bv {
                total += w;
            }
        }
        total
    }

    /// Exact MaxCut by brute force; `None` when the instance is too large.
    pub fn exact_maxcut(&self) -> Option<f64> {
        if self.num_qubits > 24 {
            return None;
        }
        let mut best = 0.0;
        for bits in 0u64..(1u64 << self.num_qubits) {
            let value = self.cut_value(bits);
            if value > best {
                best = value;
            }
        }
        Some(best)
    }
}

/// Configuration for an MPS-QAOA run.
#[derive(Clone, Debug)]
pub struct MpsQaoaConfig {
    /// QAOA rounds (`p`).
    pub layers: usize,
    /// Maximum bond dimension.
    pub max_bond: usize,
    /// Singular-value truncation threshold.
    pub truncation_tol: f64,
    /// Gradient-descent steps used by [`find_angles_mps`] per restart.
    pub optimizer_steps: usize,
    /// Number of random restarts; the best run is returned.
    pub restarts: usize,
    /// Seed for the random initial angles.
    pub seed: u64,
}

impl Default for MpsQaoaConfig {
    fn default() -> Self {
        Self {
            layers: 2,
            max_bond: 64,
            truncation_tol: 1e-12,
            optimizer_steps: 60,
            restarts: 3,
            seed: 0,
        }
    }
}

/// Result of an MPS-QAOA run.
#[derive(Clone, Debug)]
pub struct MpsQaoaOutcome {
    /// Expected cut weight `<H_C>` at the reported angles.
    pub energy: f64,
    /// Mixer angles (one per layer).
    pub betas: Vec<f64>,
    /// Cost angles (one per layer).
    pub gammas: Vec<f64>,
    /// Largest bond dimension present in the final state.
    pub max_bond_reached: usize,
    /// Wall-clock time of the run.
    pub elapsed: Duration,
}

impl MpsQaoaOutcome {
    /// `<H_C> / reference`, where `reference` is the exact MaxCut when known
    /// and the total weight (an upper bound) otherwise.
    pub fn approximation_ratio(&self, problem: &MaxCutProblem) -> f64 {
        let reference = problem
            .exact_maxcut()
            .unwrap_or_else(|| problem.total_weight());
        if reference > 0.0 {
            self.energy / reference
        } else {
            0.0
        }
    }
}

/// QAOA-on-MPS simulator for a fixed problem.
struct QaoaMpsSim<'a> {
    problem: &'a MaxCutProblem,
    max_bond: usize,
    tol: f64,
}

impl<'a> QaoaMpsSim<'a> {
    fn new(problem: &'a MaxCutProblem, max_bond: usize, tol: f64) -> Self {
        Self {
            problem,
            max_bond,
            tol,
        }
    }

    /// `|+>^n` as an MPS.
    fn initial_state(&self) -> MPS {
        let mut mps = MPS::zero_state(self.problem.num_qubits, self.max_bond);
        for site in 0..self.problem.num_qubits {
            mps.apply_single_qubit_gate(&H, site).expect("valid site");
        }
        mps
    }

    /// One cost layer: `exp(-i * gamma * w * Z_u Z_v)` per edge, applied as a
    /// diagonal two-qubit gate (identical to the CNOT-Rz-CNOT sequence used by
    /// the state-vector builder, without the SWAP overhead).
    fn apply_cost_layer(&self, mps: &mut MPS, gamma: f64) {
        for &(u, v, w) in &self.problem.edges {
            let theta = gamma * w;
            let (c, s) = (theta.cos(), theta.sin());
            let diag = [
                Complex64::new(c, -s),
                Complex64::new(c, s),
                Complex64::new(c, s),
                Complex64::new(c, -s),
            ];
            mps.apply_diagonal_two_site(&diag, u, v, self.max_bond, self.tol)
                .expect("diagonal two-site gate");
        }
    }

    /// One mixer layer: `Rx(2 * beta)` on every qubit.
    fn apply_mixer_layer(&self, mps: &mut MPS, beta: f64) {
        let (c, s) = (beta.cos(), beta.sin());
        let rx = [
            [Complex64::new(c, 0.0), Complex64::new(0.0, -s)],
            [Complex64::new(0.0, -s), Complex64::new(c, 0.0)],
        ];
        for site in 0..self.problem.num_qubits {
            mps.apply_single_qubit_gate(&rx, site).expect("valid site");
        }
    }

    /// `<H_C> = sum w (1 - <Z_u Z_v>) / 2`, by transfer contraction.
    fn cost_expectation(&self, mps: &MPS) -> f64 {
        let norm = mps.norm_sqr_mps();
        let mut energy = 0.0;
        if norm <= 1e-30 {
            return 0.0;
        }
        for &(u, v, w) in &self.problem.edges {
            let zz = mps
                .expectation_two_site_product(&Z, u, &Z, v)
                .expect("valid sites")
                .re
                / norm;
            energy += 0.5 * w * (1.0 - zz);
        }
        energy
    }

    fn max_bond_reached(&self, mps: &MPS) -> usize {
        mps.tensors
            .iter()
            .map(|t| t.shape[0].max(t.shape[2]))
            .max()
            .unwrap_or(0)
    }

    /// Run QAOA with fixed angles and return the final state.
    fn run(&self, betas: &[f64], gammas: &[f64]) -> Result<(MPS, f64), String> {
        if betas.len() != gammas.len() {
            return Err("betas and gammas must have the same length".to_string());
        }
        let mut mps = self.initial_state();
        for layer in 0..betas.len() {
            self.apply_cost_layer(&mut mps, gammas[layer]);
            self.apply_mixer_layer(&mut mps, betas[layer]);
        }
        let energy = self.cost_expectation(&mps);
        Ok((mps, energy))
    }
}

/// Run MPS-QAOA on a MaxCut instance with fixed angles.
pub fn run_maxcut_mps(
    problem: &MaxCutProblem,
    betas: &[f64],
    gammas: &[f64],
    max_bond: usize,
    tol: f64,
) -> Result<MpsQaoaOutcome, String> {
    let started = Instant::now();
    let sim = QaoaMpsSim::new(problem, max_bond, tol);
    let (mps, energy) = sim.run(betas, gammas)?;
    Ok(MpsQaoaOutcome {
        energy,
        betas: betas.to_vec(),
        gammas: gammas.to_vec(),
        max_bond_reached: sim.max_bond_reached(&mps),
        elapsed: started.elapsed(),
    })
}

/// Find QAOA angles on the MPS by maximizing `<H_C>` with the parameter-shift
/// gradient (exact for QAOA) and a shrinking-step gradient ascent.
pub fn find_angles_mps(
    problem: &MaxCutProblem,
    config: &MpsQaoaConfig,
) -> Result<MpsQaoaOutcome, String> {
    let started = Instant::now();
    let sim = QaoaMpsSim::new(problem, config.max_bond, config.truncation_tol);
    let p = config.layers;
    let half_pi = std::f64::consts::FRAC_PI_2;

    let energy = |params: &[f64], sim: &QaoaMpsSim| -> f64 {
        let (betas, gammas) = params.split_at(p);
        sim.run(betas, gammas)
            .map(|(_, e)| e)
            .unwrap_or(f64::NEG_INFINITY)
    };

    // Exact for QAOA: d<H>/dt = 0.5*(<H>(t+pi/2) - <H>(t-pi/2)).
    let gradient = |params: &[f64], sim: &QaoaMpsSim| -> Vec<f64> {
        let mut grad = vec![0.0; params.len()];
        for i in 0..params.len() {
            let mut plus = params.to_vec();
            plus[i] += half_pi;
            let mut minus = params.to_vec();
            minus[i] -= half_pi;
            grad[i] = 0.5 * (energy(&plus, sim) - energy(&minus, sim));
        }
        grad
    };

    // One gradient-ascent run from a random start.
    let optimize_from = |start: &[f64], sim: &QaoaMpsSim| -> (f64, Vec<f64>) {
        let mut params = start.to_vec();
        let mut best = energy(&params, sim);
        let mut best_params = params.clone();
        let mut step = 0.4;
        for _ in 0..config.optimizer_steps {
            let grad = gradient(&params, sim);
            let candidate: Vec<f64> = params
                .iter()
                .zip(&grad)
                .map(|(x, g)| x + step * g)
                .collect();
            let value = energy(&candidate, sim);
            if value > best + 1e-12 {
                best = value;
                best_params = candidate.clone();
                params = candidate;
            } else {
                step *= 0.5;
                if step < 1e-9 {
                    break;
                }
            }
        }
        (best, best_params)
    };

    // Random restarts: QAOA landscapes are multimodal, and restarts are cheap
    // relative to the quantum-classical loop they replace.
    let restarts = config.restarts.max(1);
    let mut best = f64::NEG_INFINITY;
    let mut best_params: Vec<f64> = Vec::new();
    for restart in 0..restarts {
        let mut rng = SmallRng::seeded(config.seed.wrapping_add(restart as u64));
        let start: Vec<f64> = (0..2 * p)
            .map(|_| 0.5 * (rng.next_f64() - 0.5) * 2.0)
            .collect();
        let (value, params) = optimize_from(&start, &sim);
        if value > best {
            best = value;
            best_params = params;
        }
    }

    let (betas, gammas) = best_params.split_at(p);
    // Recompute once to report the bond dimension of the best state.
    let (mps, energy) = sim.run(betas, gammas)?;
    Ok(MpsQaoaOutcome {
        energy,
        betas: betas.to_vec(),
        gammas: gammas.to_vec(),
        max_bond_reached: sim.max_bond_reached(&mps),
        elapsed: started.elapsed(),
    })
}
