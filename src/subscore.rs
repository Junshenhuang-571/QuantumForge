//! SubsCoRe: adaptive observation cost control for VQE.
//!
//! Implements *subspace in confident region* from Anders, Nicoli, Wu, Elosegui,
//! Pedrielli, Funcke, Jansen, Kühn and Nakajima, **Adaptive Observation Cost
//! Control for Variational Quantum Eigensolvers** (arXiv:2502.01704, ICML 2024).
//!
//! # The problem
//!
//! A VQE minimizes `f(x) = <psi_x|H|psi_x>` over `x` in `[0, 2pi)^D`. Each
//! *observation* of `f` is itself an average over many measurement shots, so it
//! carries shot noise with variance `eta^2 / N_shots`. Since quantum time is the
//! scarce resource, the cost that matters is the **total number of shots**, not
//! the number of classical iterations.
//!
//! Because the circuit consists of rotations and fixed gates, `f` is a
//! low-order trigonometric polynomial along each axis, so a 1D subspace can be
//! identified from `1 + 2 V_d` observations (NFT / SMO). SubsCoRe runs that
//! subspace optimization, but chooses **how many shots to spend at each
//! observed point** so that the whole updated axis lies inside the *confident
//! region* — the set where the GP surrogate's posterior variance is below a
//! threshold `kappa^2`.
//!
//! The threshold adapts: it is loose while the energy is dropping quickly and
//! tightens to `core_min` as progress stalls, which is where accurate
//! observations actually pay off.
//!
//! # Layout
//!
//! - [`vqe_kernel`] — the VQE kernel of Nicoli et al. (Eq. 10).
//! - [`GaussianProcess`] — heteroscedastic GP regression with that kernel
//!   (Eqs. 4-5), with an incremental Cholesky factor so adding a point is
//!   `O(n^2)`.
//! - [`axis_points`] — the `1 + 2 V_d` equidistant points along one axis.
//! - [`choose_shots_bound`] / [`choose_shots_center`] — the two shot-allocation
//!   variants (SubsCoRe-Bound and SubsCoRe-Center).
//! - [`run_subscore`] — the full loop (Algorithm 1 of the paper).
//! - [`ShotHamiltonian`] — a shot-based `PauliSum` energy objective for a
//!   [`ParameterizedCircuit`], so the optimizer can be driven by this crate's
//!   own seeded-shot simulator.
//!
//! # Deviations from the paper
//!
//! The paper's pseudocode scans the candidate observation variance upwards from
//! `eta` while its stated objective (Eq. 15) is to *maximize* the total
//! variance, i.e. spend the fewest shots, subject to the confident-region
//! constraint. This module implements Eq. 15: it searches variance downwards
//! from `kappa^2` and keeps the largest feasible value.
//!
//! The paper also uses `kappa` ambiguously (a variance threshold in Eqs. 12
//! and 15, a shot count in its Table 1). Here `kappa_sq` is always a variance.

use crate::analytic::{ExactComplex, Pauli, PauliSum};
use crate::{Circuit, Gate, ParameterizedCircuit, RotationAxis, StateVector};

const TWO_PI: f64 = std::f64::consts::TAU;

/// Largest eigenvalue guard for the Cholesky factorization.
const MIN_CHOL_DIAGONAL: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// The VQE kernel (Eq. 10 of the paper):
///
/// `k(x, x') = sigma0^2 * prod_d (gamma^2 + 2 sum_{v=1}^{V_d} cos(v (x_d - x'_d)))
///                              / (gamma^2 + 2 V_d)`
///
/// `counts[d]` is `V_d`, the number of gates parameterized by variable `d`.
/// A variable with `V_d = 0` contributes a factor of 1 and is skipped.
///
/// [`GaussianProcess`] uses the same convention.
pub fn vqe_kernel(x: &[f64], y: &[f64], gamma_sq: f64, sigma0_sq: f64, counts: &[usize]) -> f64 {
    assert_eq!(x.len(), y.len(), "kernel points must have equal dimension");
    assert_eq!(
        x.len(),
        counts.len(),
        "one gate count per parameter is required"
    );
    let mut product = sigma0_sq;
    for d in 0..x.len() {
        let harmonics = counts[d];
        if harmonics == 0 {
            continue;
        }
        let mut sum = 0.0;
        for harmonic in 1..=harmonics {
            sum += (harmonic as f64 * (x[d] - y[d])).cos();
        }
        let denominator = gamma_sq + 2.0 * harmonics as f64;
        product *= (gamma_sq + 2.0 * sum) / denominator;
    }
    product
}

// ---------------------------------------------------------------------------
// Dense linear algebra
// ---------------------------------------------------------------------------

/// Solve `L u = b` for a lower-triangular `L`.
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let mut u = vec![0.0; b.len()];
    for (i, row) in l.iter().enumerate() {
        let sum = b[i]
            - row
                .iter()
                .zip(&u)
                .take(i)
                .map(|(entry, solved)| entry * solved)
                .sum::<f64>();
        u[i] = sum / row[i];
    }
    u
}

/// Solve `L^T x = u` for a lower-triangular `L`.
fn backward_substitution(l: &[Vec<f64>], u: &[f64]) -> Vec<f64> {
    let mut x = vec![0.0; u.len()];
    for i in (0..u.len()).rev() {
        let sum = u[i]
            - l.iter()
                .skip(i + 1)
                .zip(&x[i + 1..])
                .map(|(row, solved)| row[i] * solved)
                .sum::<f64>();
        x[i] = sum / l[i][i];
    }
    x
}

// ---------------------------------------------------------------------------
// Gaussian process
// ---------------------------------------------------------------------------

/// Heteroscedastic GP regression with the VQE kernel.
///
/// Stores observations `(x, y, sigma^2)` and answers posterior mean and
/// variance queries (Eqs. 4 and 5). The Cholesky factor of
/// `K + Diag(sigma^2)` is maintained incrementally, so [`Self::push`] costs
/// `O(n^2)` rather than the `O(n^3)` of a refactorization.
#[derive(Clone, Debug)]
pub struct GaussianProcess {
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    variances: Vec<f64>,
    gamma_sq: f64,
    sigma0_sq: f64,
    counts: Vec<usize>,
    /// Lower Cholesky factor of `K + Diag(variances)`.
    factor: Vec<Vec<f64>>,
    /// `(K + Diag(variances))^{-1} y`.
    alpha: Vec<f64>,
}

impl GaussianProcess {
    /// An empty GP over `counts.len()` parameters.
    ///
    /// `gamma_sq` controls smoothness (`1.0` makes the Fourier basis
    /// orthonormal) and `sigma0_sq` is the prior variance, which the kernel
    /// gives every point: `k(x, x) = sigma0^2`.
    pub fn new(gamma_sq: f64, sigma0_sq: f64, counts: Vec<usize>) -> Self {
        Self {
            points: Vec::new(),
            values: Vec::new(),
            variances: Vec::new(),
            gamma_sq,
            sigma0_sq,
            counts,
            factor: Vec::new(),
            alpha: Vec::new(),
        }
    }

    /// Number of observations stored.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True when no observation has been added.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Kernel-hyperparameter pair `(gamma^2, sigma0^2)`.
    pub fn hyperparameters(&self) -> (f64, f64) {
        (self.gamma_sq, self.sigma0_sq)
    }

    /// Add an observation of value `value` at `x` with noise variance
    /// `variance`.
    ///
    /// # Panics
    ///
    /// Panics when `x` has the wrong dimension or `variance` is not positive.
    pub fn push(&mut self, x: Vec<f64>, value: f64, variance: f64) {
        assert_eq!(x.len(), self.counts.len(), "wrong parameter dimension");
        assert!(variance > 0.0, "observation variance must be positive");
        if self.points.is_empty() {
            let diagonal = self.sigma0_sq + variance;
            self.factor = vec![vec![diagonal.sqrt()]];
        } else {
            // Append one row/column to the Cholesky factor:
            // m = L^{-1} k, s = sqrt(k(x,x) + variance - m^T m).
            let kernel_vector = self.kernel_vector(&x);
            let m = forward_substitution(&self.factor, &kernel_vector);
            let prior = self.sigma0_sq + variance;
            let residual = prior - m.iter().map(|entry| entry * entry).sum::<f64>();
            assert!(
                residual > MIN_CHOL_DIAGONAL,
                "kernel matrix is not positive definite; lower sigma0_sq or raise gamma_sq"
            );
            let mut row = m;
            row.push(residual.sqrt());
            // The new factor is the old one with a zero column appended, plus
            // the new bottom row.
            let mut new_factor: Vec<Vec<f64>> = self
                .factor
                .iter()
                .map(|existing| {
                    let mut widened = existing.clone();
                    widened.push(0.0);
                    widened
                })
                .collect();
            new_factor.push(row);
            self.factor = new_factor;
        }
        self.points.push(x);
        self.values.push(value);
        self.variances.push(variance);
        self.refresh_alpha();
    }

    fn refresh_alpha(&mut self) {
        if self.values.is_empty() {
            self.alpha.clear();
            return;
        }
        let u = forward_substitution(&self.factor, &self.values);
        self.alpha = backward_substitution(&self.factor, &u);
    }

    /// Kernel evaluations between `x` and every stored point.
    pub fn kernel_vector(&self, x: &[f64]) -> Vec<f64> {
        self.points
            .iter()
            .map(|point| vqe_kernel(x, point, self.gamma_sq, self.sigma0_sq, &self.counts))
            .collect()
    }

    /// Posterior mean at `x` (Eq. 4).
    pub fn posterior_mean(&self, x: &[f64]) -> f64 {
        let kernel_vector = self.kernel_vector(x);
        self.alpha
            .iter()
            .zip(kernel_vector)
            .map(|(a, k)| a * k)
            .sum()
    }

    /// Posterior variance at `x` (Eq. 5), clamped against round-off.
    pub fn posterior_variance(&self, x: &[f64]) -> f64 {
        if self.is_empty() {
            return self.sigma0_sq;
        }
        let kernel_vector = self.kernel_vector(x);
        let projected = forward_substitution(&self.factor, &kernel_vector);
        let reduction = projected.iter().map(|entry| entry * entry).sum::<f64>();
        (self.sigma0_sq - reduction).max(0.0)
    }

    /// Posterior mean and variance at `x` in one pass.
    pub fn posterior_mean_and_variance(&self, x: &[f64]) -> (f64, f64) {
        let kernel_vector = self.kernel_vector(x);
        let mean = self
            .alpha
            .iter()
            .zip(&kernel_vector)
            .map(|(a, k)| a * k)
            .sum();
        let projected = forward_substitution(&self.factor, &kernel_vector);
        let reduction = projected.iter().map(|entry| entry * entry).sum::<f64>();
        (mean, (self.sigma0_sq - reduction).max(0.0))
    }
}

// ---------------------------------------------------------------------------
// Subspace geometry
// ---------------------------------------------------------------------------

/// The `1 + 2 V_d` equidistant observation points along `axis`, as used by
/// SubsCoRe: `alpha_w = 2 pi w / (1 + 2 V_d)` for `w` in `0..1 + 2 V_d`.
///
/// Element `0` is the current best point itself (`alpha_0 = 0`), which is why
/// SubsCoRe-Center can spend fewer shots there.
pub fn axis_points(best: &[f64], axis: usize, counts: &[usize]) -> Vec<Vec<f64>> {
    assert!(axis < best.len(), "axis out of range");
    let total = 1 + 2 * counts[axis].max(1);
    (0..total)
        .map(|w| {
            let mut point = best.to_vec();
            point[axis] = (best[axis] + TWO_PI * w as f64 / total as f64) % TWO_PI;
            point
        })
        .collect()
}

/// `grid` points spaced along the axis through `base`, used to check the
/// confident-region constraint over the whole subspace.
fn axis_scan(base: &[f64], axis: usize, grid: usize) -> Vec<Vec<f64>> {
    (0..grid)
        .map(|k| {
            let mut point = base.to_vec();
            point[axis] = TWO_PI * k as f64 / grid as f64;
            point
        })
        .collect()
}

/// Shots required to reach noise variance `variance` given a single-shot
/// variance `eta_sq`: `N = ceil(eta_sq / variance)`, at least one.
pub fn shots_for_variance(eta_sq: f64, variance: f64) -> usize {
    if variance <= 0.0 {
        return 1;
    }
    (eta_sq / variance).ceil().max(1.0) as usize
}

/// Worst posterior variance along `axis` after adding `added` with noise
/// variances `added_variances`.
fn worst_axis_variance(
    gp: &GaussianProcess,
    base: &[f64],
    axis: usize,
    added: &[Vec<f64>],
    added_variances: &[f64],
    grid: usize,
) -> f64 {
    let mut augmented = gp.clone();
    for (point, variance) in added.iter().zip(added_variances) {
        augmented.push(point.clone(), 0.0, *variance);
    }
    axis_scan(base, axis, grid)
        .iter()
        .map(|point| augmented.posterior_variance(point))
        .fold(0.0f64, f64::max)
}

/// Grids used while searching the shot allocation.
#[derive(Clone, Debug)]
pub struct SearchOptions {
    /// Candidate observation variances tried per search.
    pub shot_grid: usize,
    /// Points scanned along an axis when checking the confident region.
    pub scan_grid: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            shot_grid: 16,
            scan_grid: 72,
        }
    }
}

/// SubsCoRe-Bound: spend the same number of shots at every observed point, with
/// observation variance `kappa_sq` (Corollary 3.2 of the paper).
pub fn choose_shots_bound(kappa_sq: f64, eta_sq: f64, point_count: usize) -> Vec<usize> {
    vec![shots_for_variance(eta_sq, kappa_sq); point_count]
}

/// SubsCoRe-Center: maximize the total observation variance — i.e. spend the
/// fewest shots — subject to the whole updated axis staying in the confident
/// region (Eq. 15 of the paper).
///
/// The shifted points share one variance; the center point, which the GP
/// already knows well, is searched separately and typically receives fewer
/// shots. Candidate variances are tried between `kappa_sq` and `eta_sq`; see
/// [`SearchOptions`].
pub fn choose_shots_center(
    gp: &GaussianProcess,
    base: &[f64],
    points: &[Vec<f64>],
    axis: usize,
    kappa_sq: f64,
    eta_sq: f64,
    search: &SearchOptions,
) -> Vec<usize> {
    assert!(!points.is_empty(), "at least one observation point");
    let shifted = &points[1..];

    // Largest variance (fewest shots) for which the shifted points keep the
    // axis inside the confident region. Feasibility is monotone: observing
    // with less noise can only help, so the feasible set is an interval
    // `[kappa_sq, sigma*]` and the scan stops at its upper end.
    let mut shifted_variance = kappa_sq;
    for step in 0..=search.shot_grid.max(1) {
        let candidate =
            kappa_sq + (eta_sq - kappa_sq) * step as f64 / search.shot_grid.max(1) as f64;
        let variances = vec![candidate; points.len()];
        let worst = worst_axis_variance(gp, base, axis, points, &variances, search.scan_grid);
        if worst <= kappa_sq {
            shifted_variance = candidate;
        } else {
            break;
        }
    }

    // Then the center, with the shifted points fixed. It is usually observed
    // more accurately, so it can tolerate a larger variance (fewer shots).
    let mut center_variance = kappa_sq;
    let mut fixed = vec![center_variance];
    fixed.extend(vec![shifted_variance; shifted.len()]);
    for step in 0..=search.shot_grid.max(1) {
        let candidate =
            kappa_sq + (eta_sq - kappa_sq) * step as f64 / search.shot_grid.max(1) as f64;
        fixed[0] = candidate;
        let worst = worst_axis_variance(gp, base, axis, points, &fixed, search.scan_grid);
        if worst <= kappa_sq {
            center_variance = candidate;
        } else {
            break;
        }
    }

    let mut shots = Vec::with_capacity(points.len());
    shots.push(shots_for_variance(eta_sq, center_variance));
    for _ in shifted {
        shots.push(shots_for_variance(eta_sq, shifted_variance));
    }
    shots
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

/// Which shot-allocation rule SubsCoRe uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadoutStrategy {
    /// Same variance at every observed point (loose bound, cheapest to compute).
    Bound,
    /// Separate variance for the center and the shifted points (Eq. 15).
    Center,
}

/// Tuning knobs for [`run_subscore`].
#[derive(Clone, Debug)]
pub struct SubsCoReOptions {
    /// Kernel smoothness `gamma^2`; `1.0` makes the Fourier basis orthonormal.
    pub gamma_sq: f64,
    /// Prior variance `sigma0^2`; the kernel gives every point `k(x, x)`.
    pub sigma0_sq: f64,
    /// Single-shot observation variance `eta^2`. The paper's theory assumes
    /// `sigma^2(N) = eta^2 / N`.
    pub eta_sq: f64,
    /// Shots used for the initial observation.
    pub initial_shots: usize,
    /// Total shot budget; the loop stops once it is exhausted.
    pub total_shot_budget: usize,
    /// Initial confident-region variance threshold. Defaults to
    /// `eta_sq / 512`, the paper's starting point.
    pub initial_core_variance: f64,
    /// Floor `C_0` for the adaptive threshold.
    pub core_min: f64,
    /// Scale `C_1` for the progress-dependent part of the threshold.
    pub core_slope_scale: f64,
    /// Number of recent best values regressed to estimate progress (`T_Ave`).
    pub slope_window: usize,
    /// Grids used by the shot-allocation search.
    pub search: SearchOptions,
    /// Shot-allocation rule.
    pub strategy: ReadoutStrategy,
    /// Seed for the observation noise.
    pub seed: u64,
}

impl Default for SubsCoReOptions {
    fn default() -> Self {
        Self {
            gamma_sq: 2.0,
            sigma0_sq: 1.0,
            eta_sq: 1.0,
            initial_shots: 64,
            total_shot_budget: 20_000,
            initial_core_variance: 1.0 / 512.0,
            core_min: 1.0 / 4096.0,
            core_slope_scale: 1.0,
            slope_window: 8,
            search: SearchOptions::default(),
            strategy: ReadoutStrategy::Center,
            seed: 42,
        }
    }
}

/// Result of [`run_subscore`].
#[derive(Clone, Debug)]
pub struct SubsCoReOutcome {
    /// Best parameter vector found (minimizer of the GP mean along the axes).
    pub best: Vec<f64>,
    /// GP-mean energy at [`Self::best`].
    pub best_value: f64,
    /// Total shots consumed, including the initial observation.
    pub shots_used: usize,
    /// Number of subspace iterations completed.
    pub iterations: usize,
    /// Shots spent per iteration, in order.
    pub shots_per_iteration: Vec<usize>,
    /// Confident-region variance threshold after each iteration.
    pub core_variances: Vec<f64>,
}

/// Least-squares slope of `values` against their index.
fn slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean_x = (n - 1) as f64 / 2.0;
    let mean_y = values.iter().sum::<f64>() / n as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (i, &value) in values.iter().enumerate() {
        let dx = i as f64 - mean_x;
        numerator += dx * (value - mean_y);
        denominator += dx * dx;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Run SubsCoRe on an arbitrary shot-based objective.
///
/// `observe(params, shots, seed)` returns the estimated objective value at
/// `params` using `shots` measurement shots. Its noise variance is assumed to
/// be `eta_sq / shots`, as in the paper.
///
/// `counts[d]` is `V_d`, the number of gates parameterized by variable `d`; it
/// sets how many observations each subspace iteration needs.
///
/// The loop follows Algorithm 1 of the paper: observe the equidistant axis
/// points with an adaptively chosen shot budget, move to the minimizer of the
/// GP posterior mean along that axis, and tighten the confident-region
/// threshold as progress slows.
pub fn run_subscore<F>(
    observe: F,
    initial: &[f64],
    counts: &[usize],
    options: &SubsCoReOptions,
) -> SubsCoReOutcome
where
    F: Fn(&[f64], usize, u64) -> f64,
{
    let dimensions = initial.len();
    assert_eq!(dimensions, counts.len(), "one gate count per parameter");
    assert!(dimensions > 0, "need at least one parameter");
    assert!(options.eta_sq > 0.0, "eta_sq must be positive");
    assert!(options.initial_shots > 0, "initial_shots must be positive");

    let mut gp = GaussianProcess::new(options.gamma_sq, options.sigma0_sq, counts.to_vec());
    let mut best = initial.to_vec();
    let mut best_value = observe(&best, options.initial_shots, options.seed);
    gp.push(
        best.clone(),
        best_value,
        options.eta_sq / options.initial_shots as f64,
    );

    let mut shots_used = options.initial_shots;
    let mut core_variance = options.initial_core_variance;
    let mut history: Vec<f64> = vec![best_value];
    let mut shots_per_iteration = Vec::new();
    let mut core_variances = Vec::new();
    let mut iterations = 0usize;
    let mut axis = 0usize;

    while shots_used < options.total_shot_budget {
        let points = axis_points(&best, axis, counts);
        let shots = match options.strategy {
            ReadoutStrategy::Bound => {
                choose_shots_bound(core_variance, options.eta_sq, points.len())
            }
            ReadoutStrategy::Center => choose_shots_center(
                &gp,
                &best,
                &points,
                axis,
                core_variance,
                options.eta_sq,
                &options.search,
            ),
        };

        let remaining = options.total_shot_budget - shots_used;
        let planned: usize = shots.iter().sum();
        // Scale the iteration to the remaining budget; if it can no longer
        // afford a meaningful observation at every point, stop rather than
        // spending the tail on one-shot measurements.
        if remaining < shots.len() * 2 || planned == 0 {
            break;
        }
        let scale = if planned > remaining {
            remaining as f64 / planned as f64
        } else {
            1.0
        };

        let mut spent = 0usize;
        for (offset, (point, &point_shots)) in points.iter().zip(shots.iter()).enumerate() {
            let room = remaining.saturating_sub(spent);
            if room == 0 {
                break;
            }
            let allocated = (((point_shots as f64) * scale) as usize).clamp(1, room);
            let value = observe(
                point,
                allocated,
                options
                    .seed
                    .wrapping_add(iterations as u64 * 7919 + offset as u64),
            );
            gp.push(point.clone(), value, options.eta_sq / allocated as f64);
            spent += allocated;
        }
        if spent == 0 {
            break;
        }
        shots_used += spent;
        shots_per_iteration.push(spent);

        // Move to the minimizer of the GP posterior mean along this axis.
        let scan = axis_scan(&best, axis, options.search.scan_grid.max(8));
        let (next, next_value) = scan
            .iter()
            .map(|point| {
                let (mean, _) = gp.posterior_mean_and_variance(point);
                (point.clone(), mean)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("scan is not empty");
        best = next;
        best_value = next_value;
        history.push(best_value);

        // Adapt the confident-region threshold to the progress of the
        // optimization: loose while the energy drops, tight near convergence.
        if history.len() > options.slope_window.max(1) {
            let window = &history[history.len() - options.slope_window.max(1)..];
            let progress = slope(window);
            core_variance = options
                .core_min
                .max(-options.core_slope_scale * progress)
                .clamp(options.eta_sq / 1.0e9, options.eta_sq);
        }
        core_variances.push(core_variance);

        iterations += 1;
        axis = (axis + 1) % dimensions;
    }

    SubsCoReOutcome {
        best,
        best_value,
        shots_used,
        iterations,
        shots_per_iteration,
        core_variances,
    }
}

// ---------------------------------------------------------------------------
// Shot-based Hamiltonian objective
// ---------------------------------------------------------------------------

/// A shot-based energy objective for a parameterized ansatz and a Pauli-sum
/// Hamiltonian.
///
/// Each Pauli word is estimated by rotating into its eigenbasis, sampling
/// `shots` computational-basis outcomes with this crate's seeded RNG, and
/// averaging the parity. Non-identity factors only contribute to the parity;
/// identity factors need no rotation.
pub struct ShotHamiltonian {
    /// The parameterized ansatz.
    pub ansatz: ParameterizedCircuit,
    /// Hamiltonian as a Pauli sum.
    pub hamiltonian: PauliSum,
    /// Number of qubits.
    pub qubits: usize,
}

impl ShotHamiltonian {
    /// Build an objective for `ansatz` and `hamiltonian` on `qubits` qubits.
    ///
    /// # Panics
    ///
    /// Panics when the Hamiltonian acts on a different number of qubits, or the
    /// ansatz uses encoded inputs (the ideal shot estimator has no input
    /// register).
    pub fn new(ansatz: ParameterizedCircuit, hamiltonian: PauliSum, qubits: usize) -> Self {
        assert_eq!(
            hamiltonian.qubits(),
            qubits,
            "the Hamiltonian must act on {qubits} qubits"
        );
        assert_eq!(
            ansatz.input_count(),
            0,
            "ShotHamiltonian does not support input-encoded ansätze"
        );
        Self {
            ansatz,
            hamiltonian,
            qubits,
        }
    }

    /// `V_d`: how many gates each parameter controls.
    pub fn parameter_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.ansatz.parameter_count()];
        for op in self.ansatz.operations() {
            if let crate::ParameterizedOp::Rotation { parameter, .. } = op {
                if *parameter < counts.len() {
                    counts[*parameter] += 1;
                }
            }
        }
        counts
    }

    /// Materialize the ansatz at `params` into a concrete circuit.
    fn materialize(&self, params: &[f64]) -> Circuit {
        let mut circuit = Circuit::new();
        for op in self.ansatz.operations() {
            match *op {
                crate::ParameterizedOp::Fixed { gate, target } => circuit.push(gate, target),
                crate::ParameterizedOp::Rotation {
                    axis,
                    target,
                    parameter,
                } => {
                    let angle = params.get(parameter).copied().unwrap_or(0.0);
                    circuit.push(rotation_gate(axis, angle), target);
                }
                crate::ParameterizedOp::Cnot { control, target } => circuit.cnot(control, target),
                crate::ParameterizedOp::Cz { first, second } => circuit.cz(first, second),
                crate::ParameterizedOp::Swap { first, second } => circuit.swap(first, second),
                crate::ParameterizedOp::EncodingRotation { .. }
                | crate::ParameterizedOp::Channel { .. } => {}
            }
        }
        circuit
    }

    /// Estimate `<H>` at `params` with `shots_per_word` shots per Pauli word.
    ///
    /// Returns the energy estimate; its noise variance is approximately
    /// `sum_w c_w^2 (1 - <W_w>^2) / shots_per_word`.
    pub fn observe(&self, params: &[f64], shots_per_word: usize, seed: u64) -> f64 {
        assert!(shots_per_word > 0, "shots_per_word must be positive");
        let base = self.materialize(params);
        let mut energy = 0.0;
        let mut word_index = 0usize;
        for (word, coefficient) in self.hamiltonian.terms() {
            let coefficient = complex_to_f64(coefficient);
            if coefficient == 0.0 {
                word_index += 1;
                continue;
            }
            let mut circuit = base.clone();
            let mut watched: Vec<usize> = Vec::new();
            for (qubit, pauli) in word.factors().iter().enumerate() {
                match pauli {
                    Pauli::X => {
                        circuit.push(Gate::H, qubit);
                        watched.push(qubit);
                    }
                    Pauli::Y => {
                        circuit.push(Gate::Sdg, qubit);
                        circuit.push(Gate::H, qubit);
                        watched.push(qubit);
                    }
                    Pauli::Z => watched.push(qubit),
                    Pauli::I => {}
                }
            }
            if watched.is_empty() {
                // Identity term: contributes its coefficient exactly.
                energy += coefficient;
                word_index += 1;
                continue;
            }
            // `enumerate` yields (classical index, qubit); the classical
            // register position is the position within `watched`.
            for (classical, qubit) in watched.iter().copied().enumerate() {
                circuit.measure(qubit, classical);
            }
            let initial = StateVector::zero(self.qubits).expect("valid state");
            let result = circuit
                .run_shots(
                    &initial,
                    shots_per_word,
                    seed.wrapping_add(word_index as u64 * 1_000_003),
                )
                .expect("shot execution");
            // Classical bit `j` holds the outcome of `watched[j]`.
            let mut parity_sum = 0i64;
            for (bits, count) in result.counts {
                let parity: u8 = (0..watched.len())
                    .map(|classical| bits.get(classical).copied().unwrap_or(0))
                    .fold(0u8, |acc, bit| acc ^ bit);
                parity_sum += if parity == 0 {
                    count as i64
                } else {
                    -(count as i64)
                };
            }
            energy += coefficient * parity_sum as f64 / shots_per_word as f64;
            word_index += 1;
        }
        energy
    }

    /// Exact energy at `params`, for reference.
    ///
    /// This is the noise-free value the shot-based [`Self::observe`] approximates.
    pub fn exact_energy(&self, params: &[f64]) -> Result<f64, String> {
        let state = self
            .ansatz
            .run(params, self.qubits)
            .map_err(|error| error.to_string())?;
        let ket = state.to_ket().map_err(|error| error.to_string())?;
        self.hamiltonian
            .expectation_ket(&ket)
            .map(|value| value.re)
            .map_err(|error| error.to_string())
    }
}

fn rotation_gate(axis: RotationAxis, angle: f64) -> Gate {
    match axis {
        RotationAxis::X => Gate::Rx(angle),
        RotationAxis::Y => Gate::Ry(angle),
        RotationAxis::Z => Gate::Rz(angle),
    }
}

fn complex_to_f64(value: &ExactComplex) -> f64 {
    value.as_complex64().re
}
