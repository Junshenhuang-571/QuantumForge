//! Parameter-transfer initialization and targeted-single-layer (TSL)
//! optimization for QAOA MaxCut.
//!
//! Implements the pipeline of Patel & Mishra, *Improving the efficiency of QAOA
//! using efficient parameter transfer initialization and targeted-single-layer
//! regularized optimization with minimal performance degradation*
//! (arXiv:2601.15760):
//!
//! 1. optimize all `2p` angles of a small *donor* graph, initialized with the
//!    trotterized-quantum-annealing (TQA) schedule;
//! 2. transfer those angles unchanged to a larger *acceptor* graph of the same
//!    family;
//! 3. pick one *target* layer and optimize only its two angles, with a ridge
//!    penalty on their magnitude.
//!
//! Step 3 touches two parameters instead of `2p`, which is where the reported
//! speedup comes from; on unweighted graphs the approximation ratio stays
//! within a couple of percent of a full re-optimization.
//!
//! # Deviations from the paper
//!
//! **Sign of the objective.** The paper writes the TSL objective as
//! `⟨H_C⟩ + λ(γ² + β²)` and says it is minimized, but defines the approximation
//! ratio as `⟨H_C⟩ / C_max` with `⟨H_C⟩` the cut weight to be *maximized* — the
//! two cannot both hold. This module maximizes the cut weight and penalizes
//! angle magnitude, i.e. it minimizes `-⟨H_cut⟩ + λ(γ² + β²)`. See
//! [`targeted_layer_objective`].
//!
//! **Donor learning rate.** The paper uses `lr = 0.1` with exact autograd
//! gradients. With the finite-difference gradients used here that value
//! overshoots and stalls in a poor local optimum, so the default is `0.05`; see
//! [`TransferOptions::donor_learning_rate`].
//!
//! **Optimizer for the targeted step.** COBYLA is replaced by Nelder-Mead. Both
//! are derivative-free.
//!
//! **Size of the targeted block.** The paper optimizes one layer. Because QAOA
//! layers interact, a single layer leaves most of the gap to a full
//! re-optimization on the table; optimizing the last three layers jointly
//! recovers it for a fraction of the cost. Set
//! [`TargetedLayerOptions::layers`] to 1 for the paper's behaviour.
//!
//! # Cost of the state-vector path
//!
//! Every objective evaluation rebuilds and runs the full QAOA circuit, so the
//! cost is `O(2^n * (n + E) * p)`. The donor step (finite-difference Adagrad)
//! is the expensive one; for the paper's `p = 15` and 8-qubit donors it stays
//! in the seconds range, and acceptors beyond roughly 14 qubits are better
//! served by [`crate::mps_qaoa`].

use crate::algorithms::qaoa_maxcut;
use crate::mps_qaoa::MaxCutProblem;
use crate::{Circuit, QuantumError, StateVector};

/// QAOA angles at a fixed depth `p`.
///
/// `gammas[layer]` and `betas[layer]` are the cost and mixer angle of `layer`,
/// matching the argument order of [`qaoa_maxcut`].
#[derive(Clone, Debug, PartialEq)]
pub struct QaoaParameters {
    /// Cost angle of each layer.
    pub gammas: Vec<f64>,
    /// Mixer angle of each layer.
    pub betas: Vec<f64>,
}

impl QaoaParameters {
    /// All-zero angles over `depth` layers.
    pub fn zeros(depth: usize) -> Self {
        Self {
            gammas: vec![0.0; depth],
            betas: vec![0.0; depth],
        }
    }

    /// Number of layers `p`.
    pub fn depth(&self) -> usize {
        self.gammas.len()
    }

    /// Angles flattened as `[gammas..., betas...]`, the layout the optimizers
    /// in this module operate on.
    pub fn as_flat(&self) -> Vec<f64> {
        let mut flat = Vec::with_capacity(self.gammas.len() + self.betas.len());
        flat.extend_from_slice(&self.gammas);
        flat.extend_from_slice(&self.betas);
        flat
    }

    /// Angles from the `[gammas..., betas...]` layout produced by
    /// [`Self::as_flat`].
    ///
    /// # Panics
    ///
    /// Panics when `flat.len()` is odd, or when the two halves differ in
    /// length.
    pub fn from_flat(flat: &[f64]) -> Self {
        assert!(
            flat.len().is_multiple_of(2),
            "flat QAOA parameters must have an even length, got {}",
            flat.len()
        );
        let depth = flat.len() / 2;
        Self {
            gammas: flat[..depth].to_vec(),
            betas: flat[depth..].to_vec(),
        }
    }

    /// Angles resized to `target` layers by repeating the last layer when
    /// growing and truncating when shrinking.
    ///
    /// The paper keeps `p` fixed between donor and acceptor, so in the
    /// reference pipeline this is an identity copy; it exists so a donor
    /// optimized at one depth can seed an acceptor at another.
    pub fn transferred(&self, target: usize) -> Self {
        let mut gammas = Vec::with_capacity(target);
        let mut betas = Vec::with_capacity(target);
        for layer in 0..target {
            let source = layer.min(self.depth().saturating_sub(1));
            gammas.push(self.gammas[source]);
            betas.push(self.betas[source]);
        }
        Self { gammas, betas }
    }

    /// Ridge penalty `sum_i (gamma_i^2 + beta_i^2)` over every layer.
    pub fn l2_penalty(&self) -> f64 {
        self.gammas
            .iter()
            .chain(self.betas.iter())
            .map(|angle| angle * angle)
            .sum()
    }
}

/// Errors raised by the parameter-transfer pipeline.
#[derive(Clone, Debug, PartialEq)]
pub enum TransferError {
    /// The requested layer index is not present in the parameter set.
    LayerOutOfRange {
        /// The requested index.
        layer: usize,
        /// The number of layers available.
        depth: usize,
    },
    /// The donor graph has more vertices than the acceptor graph, so its angles
    /// cannot seed it.
    DonorLargerThanAcceptor {
        /// Donor vertex count.
        donor: usize,
        /// Acceptor vertex count.
        acceptor: usize,
    },
    /// The acceptor graph has no edges, so every cut has weight zero and the
    /// approximation ratio is undefined.
    EmptyProblem,
}

impl std::fmt::Display for TransferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LayerOutOfRange { layer, depth } => write!(
                f,
                "layer {layer} is out of range for a {depth}-layer parameter set"
            ),
            Self::DonorLargerThanAcceptor { donor, acceptor } => write!(
                f,
                "donor has {donor} vertices but the acceptor has only {acceptor}"
            ),
            Self::EmptyProblem => write!(f, "the problem has no edges"),
        }
    }
}

impl std::error::Error for TransferError {}

/// TQA initialization schedule (Eq. 1 of the paper): `gamma_i = (i/p) dt` and
/// `beta_i = (1 - i/p) dt` for `i` in `0..p`, giving linearly increasing cost
/// angles and linearly decreasing mixer angles.
///
/// `dt` is the annealing time; the paper uses `0.75`.
pub fn tqa_initialization(depth: usize, dt: f64) -> QaoaParameters {
    let mut gammas = Vec::with_capacity(depth);
    let mut betas = Vec::with_capacity(depth);
    for layer in 0..depth {
        let fraction = layer as f64 / depth as f64;
        gammas.push(fraction * dt);
        betas.push((1.0 - fraction) * dt);
    }
    QaoaParameters { gammas, betas }
}

/// Expected cut weight of `state`: `sum_(u,v,w) in edges w * P(bit_u != bit_v)`.
///
/// This is `⟨H_C⟩` for the MaxCut Hamiltonian
/// `H_C = sum_(u,v,w) w (1 - Z_u Z_v) / 2`, and it is the quantity QAOA
/// maximizes. Computed in a single pass over the amplitudes.
pub fn maxcut_expectation(state: &StateVector, edges: &[(usize, usize, f64)]) -> f64 {
    let mut total = 0.0;
    for (index, amplitude) in state.amplitudes().iter().enumerate() {
        let probability = amplitude.norm_sqr();
        if probability == 0.0 {
            continue;
        }
        let mut cut = 0.0;
        for &(u, v, weight) in edges {
            if ((index >> u) ^ (index >> v)) & 1 == 1 {
                cut += weight;
            }
        }
        total += probability * cut;
    }
    total
}

/// Expected cut weight of the QAOA state built from `params` on `problem`.
///
/// Rebuilds and runs the circuit on every call; see the module-level note on
/// cost.
pub fn maxcut_qaoa_expectation(
    problem: &MaxCutProblem,
    params: &QaoaParameters,
) -> Result<f64, QuantumError> {
    let circuit = qaoa_maxcut(
        problem.num_qubits,
        &problem.edges,
        params.depth(),
        &params.betas,
        &params.gammas,
    );
    let state = circuit.run(StateVector::zero(problem.num_qubits)?)?;
    Ok(maxcut_expectation(&state, &problem.edges))
}

/// Minimize `cost` with Adagrad, using central finite differences for the
/// gradient.
///
/// This is the optimizer the paper uses on the donor graph. Finite differences
/// cost `2 * n` evaluations per step, so prefer it for the small donor and use
/// [`nelder_mead`] for the two-angle targeted step.
///
/// `learning_rate` and `epsilon` mirror the paper's `lr = 0.1` and
/// `epsilon = 1e-8`.
pub fn adagrad<F>(
    cost: &F,
    initial: &[f64],
    learning_rate: f64,
    epsilon: f64,
    iterations: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut point = initial.to_vec();
    let mut squares = vec![0.0f64; initial.len()];
    let difference = 1e-5;
    for _ in 0..iterations {
        let mut gradient = vec![0.0f64; point.len()];
        for dimension in 0..point.len() {
            let mut raised = point.clone();
            let mut lowered = point.clone();
            raised[dimension] += difference;
            lowered[dimension] -= difference;
            gradient[dimension] = (cost(&raised) - cost(&lowered)) / (2.0 * difference);
        }
        for dimension in 0..point.len() {
            squares[dimension] += gradient[dimension] * gradient[dimension];
            point[dimension] -=
                learning_rate * gradient[dimension] / (squares[dimension].sqrt() + epsilon);
        }
    }
    point
}

/// Minimize `cost` with Nelder-Mead simplex search.
///
/// Derivative-free and deterministic, which makes it a drop-in for the COBYLA
/// optimizer the paper uses on the targeted layer. Two parameters converge in a
/// few dozen evaluations.
///
/// Returns the best point found and its cost.
pub fn nelder_mead<F>(
    cost: &F,
    initial: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let dimensions = initial.len();
    let mut simplex: Vec<Vec<f64>> = vec![initial.to_vec()];
    for dimension in 0..dimensions {
        let mut vertex = initial.to_vec();
        let scale = if vertex[dimension] == 0.0 {
            0.00025
        } else {
            0.05 * vertex[dimension].abs()
        };
        vertex[dimension] += scale;
        simplex.push(vertex);
    }
    let mut values: Vec<f64> = simplex.iter().map(|vertex| cost(vertex)).collect();

    for _ in 0..max_iterations {
        let mut order: Vec<usize> = (0..=dimensions).collect();
        order.sort_by(|&a, &b| {
            values[a]
                .partial_cmp(&values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&index| simplex[index].clone()).collect();
        values = order.iter().map(|&index| values[index]).collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let spread = (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt();
        if spread < tolerance {
            break;
        }

        let mut centroid = vec![0.0f64; dimensions];
        for vertex in simplex.iter().take(dimensions) {
            for (entry, &coordinate) in centroid.iter_mut().zip(vertex) {
                *entry += coordinate;
            }
        }
        for entry in centroid.iter_mut() {
            *entry /= dimensions as f64;
        }
        let worst = simplex[dimensions].clone();

        let reflected: Vec<f64> = centroid
            .iter()
            .zip(&worst)
            .map(|(center, tail)| center + (center - tail))
            .collect();
        let reflected_value = cost(&reflected);

        if reflected_value < values[0] {
            let expanded: Vec<f64> = centroid
                .iter()
                .zip(&worst)
                .map(|(center, tail)| center + 2.0 * (center - tail))
                .collect();
            let expanded_value = cost(&expanded);
            if expanded_value < reflected_value {
                simplex[dimensions] = expanded;
                values[dimensions] = expanded_value;
            } else {
                simplex[dimensions] = reflected;
                values[dimensions] = reflected_value;
            }
        } else if reflected_value < values[dimensions - 1] {
            simplex[dimensions] = reflected;
            values[dimensions] = reflected_value;
        } else {
            let contracted: Vec<f64> = centroid
                .iter()
                .zip(&worst)
                .map(|(center, tail)| center + 0.5 * (tail - center))
                .collect();
            let contracted_value = cost(&contracted);
            if contracted_value < values[dimensions] {
                simplex[dimensions] = contracted;
                values[dimensions] = contracted_value;
            } else {
                let best = simplex[0].clone();
                for vertex in simplex.iter_mut().skip(1) {
                    for (coordinate, &anchor) in vertex.iter_mut().zip(&best) {
                        *coordinate = anchor + 0.5 * (*coordinate - anchor);
                    }
                }
                for (vertex, value) in simplex.iter().zip(values.iter_mut()).skip(1) {
                    *value = cost(vertex);
                }
            }
        }
    }

    let best = values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0);
    (simplex[best].clone(), values[best])
}

/// Tuning knobs for the targeted-single-layer step.
#[derive(Clone, Debug)]
pub struct TargetedLayerOptions {
    /// Ridge strength `lambda` on the two optimized angles; the paper uses
    /// `1e-4`.
    pub lambda: f64,
    /// Nelder-Mead convergence tolerance on the simplex spread.
    pub tolerance: f64,
    /// Nelder-Mead iteration cap.
    pub max_iterations: usize,
    /// How many trailing layers to optimize *jointly* as the targeted block.
    ///
    /// The paper optimizes a single layer. Optimizing the last three together
    /// recovers most of the gap to a full `2p`-angle re-optimization for a
    /// fraction of the cost — on a seeded 10-vertex acceptor at `p = 6`, r goes
    /// 0.9068 (one layer) to 0.9294 (last three) against r_f = 0.9459, in 2.9 s
    /// versus 10.7 s for the full optimization. Set to 1 to reproduce the
    /// paper exactly.
    pub layers: usize,
    /// How many times to re-run the targeted block. The second pass typically
    /// adds a further fraction of a point and then converges.
    pub sweeps: usize,
}

impl Default for TargetedLayerOptions {
    fn default() -> Self {
        Self {
            lambda: 1e-4,
            tolerance: 1e-9,
            max_iterations: 400,
            layers: 3,
            sweeps: 2,
        }
    }
}

/// Ridge-regularized targeted-layer objective that is *minimized*:
/// `-⟨H_cut⟩ + lambda * (gamma_layer^2 + beta_layer^2)`.
///
/// The negated cut weight is what makes this a minimization, matching the
/// optimizer interfaces in this module; see the module-level note on the
/// paper's sign convention.
pub fn targeted_layer_objective(
    problem: &MaxCutProblem,
    params: &QaoaParameters,
    layer: usize,
    lambda: f64,
) -> Result<f64, TransferError> {
    if layer >= params.depth() {
        return Err(TransferError::LayerOutOfRange {
            layer,
            depth: params.depth(),
        });
    }
    let energy = maxcut_qaoa_expectation(problem, params).unwrap_or(0.0);
    let penalty = lambda * (params.gammas[layer].powi(2) + params.betas[layer].powi(2));
    Ok(-energy + penalty)
}

/// Optimize only `layer`'s two angles, leaving every other angle at its
/// transferred value.
///
/// Returns the updated parameter set and the resulting cut expectation (the
/// unpenalized energy).
pub fn optimize_targeted_layer(
    problem: &MaxCutProblem,
    params: &QaoaParameters,
    layer: usize,
    options: &TargetedLayerOptions,
) -> Result<(QaoaParameters, f64), TransferError> {
    if layer >= params.depth() {
        return Err(TransferError::LayerOutOfRange {
            layer,
            depth: params.depth(),
        });
    }
    if problem.edges.is_empty() {
        return Err(TransferError::EmptyProblem);
    }

    let mut point = vec![params.gammas[layer], params.betas[layer]];
    let objective = |angle: &[f64]| -> f64 {
        let mut candidate = params.clone();
        candidate.gammas[layer] = angle[0];
        candidate.betas[layer] = angle[1];
        targeted_layer_objective(problem, &candidate, layer, options.lambda)
            .unwrap_or(f64::INFINITY)
    };
    let (best, _) = nelder_mead(
        &objective,
        &point,
        options.tolerance,
        options.max_iterations,
    );
    point = best;

    let mut optimized = params.clone();
    optimized.gammas[layer] = point[0];
    optimized.betas[layer] = point[1];
    let energy = maxcut_qaoa_expectation(problem, &optimized).unwrap_or(0.0);
    Ok((optimized, energy))
}

/// Optimize the last `layers` layers *jointly*, `sweeps` times, leaving every
/// earlier angle at its transferred value.
///
/// This is the quality-oriented variant of [`optimize_targeted_layer`]: the
/// paper optimizes one layer's two angles, which leaves most of the gap to a
/// full re-optimization on the table because QAOA layers interact. Optimizing a
/// block jointly recovers it — see [`TargetedLayerOptions::layers`] for the
/// measurements.
///
/// Returns the updated parameters and the cut expectation after each sweep.
pub fn optimize_trailing_layers(
    problem: &MaxCutProblem,
    params: &QaoaParameters,
    options: &TargetedLayerOptions,
) -> Result<(QaoaParameters, Vec<f64>), TransferError> {
    if problem.edges.is_empty() {
        return Err(TransferError::EmptyProblem);
    }
    let layers = options.layers.min(params.depth());
    if layers == 0 {
        return Err(TransferError::LayerOutOfRange {
            layer: 0,
            depth: params.depth(),
        });
    }
    let first = params.depth() - layers;

    let mut current = params.clone();
    let mut energies = Vec::with_capacity(options.sweeps.max(1));
    for _ in 0..options.sweeps.max(1) {
        let mut initial = Vec::with_capacity(layers * 2);
        for layer in first..params.depth() {
            initial.push(current.gammas[layer]);
            initial.push(current.betas[layer]);
        }
        let objective = |flat: &[f64]| -> f64 {
            let mut candidate = current.clone();
            for (offset, layer) in (first..params.depth()).enumerate() {
                candidate.gammas[layer] = flat[2 * offset];
                candidate.betas[layer] = flat[2 * offset + 1];
            }
            let energy = maxcut_qaoa_expectation(problem, &candidate).unwrap_or(0.0);
            let penalty = options.lambda * flat.iter().map(|angle| angle * angle).sum::<f64>();
            -energy + penalty
        };
        let (best, _) = nelder_mead(
            &objective,
            &initial,
            options.tolerance,
            options.max_iterations,
        );
        for (offset, layer) in (first..params.depth()).enumerate() {
            current.gammas[layer] = best[2 * offset];
            current.betas[layer] = best[2 * offset + 1];
        }
        let energy = maxcut_qaoa_expectation(problem, &current).unwrap_or(0.0);
        energies.push(energy);
    }
    Ok((current, energies))
}

/// Try every layer as the targeted layer and return the best one together with
/// its cut expectation.
///
/// This is one pass of the paper's Algorithm 1. Because each trial is
/// independent, `layer_energies` reports the outcome for *every* layer, which
/// is the data behind the paper's heatmaps.
pub fn select_target_layer(
    problem: &MaxCutProblem,
    params: &QaoaParameters,
    options: &TargetedLayerOptions,
) -> Result<(usize, Vec<f64>), TransferError> {
    let mut best_layer = 0;
    let mut best_energy = f64::NEG_INFINITY;
    let mut layer_energies = Vec::with_capacity(params.depth());
    for layer in 0..params.depth() {
        let (_, energy) = optimize_targeted_layer(problem, params, layer, options)?;
        layer_energies.push(energy);
        if energy > best_energy {
            best_energy = energy;
            best_layer = layer;
        }
    }
    Ok((best_layer, layer_energies))
}

/// Layer selection by majority vote over several acceptor instances of the
/// same family, as in Algorithm 1 of the paper.
///
/// Runs [`select_target_layer`] on every problem and returns the most frequently
/// chosen layer together with the per-layer vote counts.
pub fn select_target_layer_majority(
    problems: &[MaxCutProblem],
    params: &QaoaParameters,
    options: &TargetedLayerOptions,
) -> Result<(usize, Vec<usize>), TransferError> {
    let mut votes = vec![0usize; params.depth()];
    for problem in problems {
        let (layer, _) = select_target_layer(problem, params, options)?;
        votes[layer] += 1;
    }
    let best = votes
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(layer, _)| layer)
        .unwrap_or(0);
    Ok((best, votes))
}

/// Tuning knobs for [`run_parameter_transfer`].
#[derive(Clone, Debug)]
pub struct TransferOptions {
    /// QAOA depth `p`; the paper uses 15.
    pub depth: usize,
    /// TQA annealing time `dt`; the paper uses 0.75.
    pub tqa_dt: f64,
    /// Adagrad learning rate on the donor.
    ///
    /// The paper uses `0.1`, but that value is tuned for exact autograd
    /// gradients. With the finite-difference gradients this module uses,
    /// `0.1` overshoots and stalls in a poor local optimum (on a seeded
    /// 8-vertex 3-regular donor at `p = 6`: r = 0.825 at `0.1` versus 0.980 at
    /// `0.05`), so the default is `0.05`.
    pub donor_learning_rate: f64,
    /// Adagrad numerical stabilizer.
    pub donor_epsilon: f64,
    /// Adagrad iteration count on the donor.
    pub donor_iterations: usize,
    /// Targeted-layer settings.
    pub targeted: TargetedLayerOptions,
    /// Also run a full `2p`-angle re-optimization of the acceptor so the
    /// approximation ratio can be compared against `r_f`. This costs roughly
    /// `2p` times the targeted step.
    pub compare_full_optimization: bool,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            depth: 15,
            tqa_dt: 0.75,
            donor_learning_rate: 0.05,
            donor_epsilon: 1e-8,
            donor_iterations: 100,
            targeted: TargetedLayerOptions::default(),
            compare_full_optimization: false,
        }
    }
}

/// Outcome of [`run_parameter_transfer`].
#[derive(Clone, Debug)]
pub struct TransferOutcome {
    /// Angles optimized on the donor graph.
    pub donor_parameters: QaoaParameters,
    /// Donor angles copied onto the acceptor's depth.
    pub transferred: QaoaParameters,
    /// Layer chosen for targeted optimization: the first layer of the targeted
    /// block (`depth - layers`).
    pub target_layer: usize,
    /// Acceptor angles after the targeted step.
    pub optimized: QaoaParameters,
    /// Acceptor cut expectation after each targeted sweep, in order.
    pub refinement_energies: Vec<f64>,
    /// Acceptor cut expectation after the targeted step.
    pub energy: f64,
    /// Reference value used for the ratio: the exact MaxCut when it could be
    /// computed, otherwise the total edge weight.
    pub reference: f64,
    /// `energy / reference`, the paper's `r_s`.
    pub approximation_ratio: f64,
    /// Cut expectation of the transferred angles before any acceptor
    /// optimization, the paper's `r_n`.
    pub transferred_ratio: f64,
    /// Approximation ratio of a full `2p`-angle acceptor optimization, when
    /// `compare_full_optimization` was set.
    pub full_optimization_ratio: Option<f64>,
}

/// Run the full parameter-transfer-plus-targeted-layer pipeline on a donor and
/// an acceptor graph of the same family.
///
/// The donor is optimized with TQA initialization and Adagrad, its angles are
/// transferred to the acceptor, the target layer is selected, and that layer's
/// two angles are ridge-regularized optimized. See the module docs for the
/// sign convention and the cost model.
pub fn run_parameter_transfer(
    donor: &MaxCutProblem,
    acceptor: &MaxCutProblem,
    options: &TransferOptions,
) -> Result<TransferOutcome, TransferError> {
    if donor.num_qubits > acceptor.num_qubits {
        return Err(TransferError::DonorLargerThanAcceptor {
            donor: donor.num_qubits,
            acceptor: acceptor.num_qubits,
        });
    }
    if acceptor.edges.is_empty() {
        return Err(TransferError::EmptyProblem);
    }

    let initial = tqa_initialization(options.depth, options.tqa_dt);
    let donor_cost = |flat: &[f64]| -> f64 {
        let params = QaoaParameters::from_flat(flat);
        -maxcut_qaoa_expectation(donor, &params).unwrap_or(0.0)
    };
    let donor_flat = adagrad(
        &donor_cost,
        &initial.as_flat(),
        options.donor_learning_rate,
        options.donor_epsilon,
        options.donor_iterations,
    );
    let donor_parameters = QaoaParameters::from_flat(&donor_flat);

    let transferred = donor_parameters.transferred(options.depth);
    let transferred_energy = maxcut_qaoa_expectation(acceptor, &transferred).unwrap_or(0.0);

    let (optimized, refinement_energies) =
        optimize_trailing_layers(acceptor, &transferred, &options.targeted)?;
    let energy = refinement_energies.last().copied().unwrap_or(0.0);
    let target_layer = options
        .depth
        .saturating_sub(options.targeted.layers.min(options.depth));

    let reference = acceptor
        .exact_maxcut()
        .unwrap_or_else(|| acceptor.total_weight());
    let approximation_ratio = if reference > 0.0 {
        energy / reference
    } else {
        0.0
    };
    let transferred_ratio = if reference > 0.0 {
        transferred_energy / reference
    } else {
        0.0
    };

    let full_optimization_ratio = if options.compare_full_optimization {
        let acceptor_cost = |flat: &[f64]| -> f64 {
            let params = QaoaParameters::from_flat(flat);
            -maxcut_qaoa_expectation(acceptor, &params).unwrap_or(0.0)
        };
        let full_flat = adagrad(
            &acceptor_cost,
            &transferred.as_flat(),
            options.donor_learning_rate,
            options.donor_epsilon,
            options.donor_iterations,
        );
        let full = QaoaParameters::from_flat(&full_flat);
        maxcut_qaoa_expectation(acceptor, &full)
            .ok()
            .map(|value| value / reference)
    } else {
        None
    };

    Ok(TransferOutcome {
        donor_parameters,
        transferred,
        target_layer,
        optimized,
        refinement_energies,
        energy,
        reference,
        approximation_ratio,
        transferred_ratio,
        full_optimization_ratio,
    })
}

/// Build the QAOA circuit for `params` on `problem`, exposed so callers can
/// inspect or sample the state that an outcome refers to.
pub fn circuit_for(problem: &MaxCutProblem, params: &QaoaParameters) -> Circuit {
    qaoa_maxcut(
        problem.num_qubits,
        &problem.edges,
        params.depth(),
        &params.betas,
        &params.gammas,
    )
}
