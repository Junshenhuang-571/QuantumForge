//! Noise-driven optimization: annealed Langevin steps.
//!
//! A gradient step with an annealed thermal kick,
//! `x <- x - lr * grad f(x) + sqrt(2 lr T) xi`, is the Langevin discretization
//! of `dx = -grad f(x) dt + sqrt(2 T) dW`. The kick satisfies
//! fluctuation-dissipation, so at fixed `T` the iterate samples the Boltzmann
//! distribution `exp(-f/T)`; annealing `T -> 0` turns it into an optimizer that
//! can cross barriers a plain gradient step cannot.
//!
//! The useful regime is a **barrier-limited** objective: the plain step is
//! trapped and the thermal kick escapes. On a smooth convex objective the kick
//! only adds variance, so this is a tool to reach for when the landscape is
//! rugged, not a default.

use crate::SmallRng;

/// Options for [`annealed_langevin`].
#[derive(Clone, Debug)]
pub struct LangevinOptions {
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// Number of Langevin steps.
    pub steps: usize,
    /// Temperature at the first step.
    pub initial_temperature: f64,
    /// Temperature at the last step.
    pub final_temperature: f64,
    /// Seed for the thermal kicks.
    pub seed: u64,
}

impl Default for LangevinOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            steps: 2_000,
            initial_temperature: 1.0,
            final_temperature: 0.01,
            seed: 42,
        }
    }
}

/// Result of an annealed Langevin run.
#[derive(Clone, Debug)]
pub struct LangevinOutcome {
    /// Best point visited.
    pub best: Vec<f64>,
    /// Objective value at `best`.
    pub best_value: f64,
    /// Objective value at the final iterate.
    pub final_value: f64,
    /// Temperature at each step, in order.
    pub temperatures: Vec<f64>,
}

/// Standard normal draw from two uniforms (Box-Muller).
fn standard_normal(rng: &mut SmallRng) -> f64 {
    let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Run annealed Langevin dynamics on `objective`.
///
/// `gradient(x, out)` must write `df/dx_i` into `out`. The temperature is
/// annealed geometrically from `initial_temperature` to `final_temperature`.
pub fn annealed_langevin<F, G>(
    objective: F,
    gradient: G,
    initial: &[f64],
    options: &LangevinOptions,
) -> LangevinOutcome
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64], &mut [f64]),
{
    let mut rng = SmallRng::seeded(options.seed);
    let mut x = initial.to_vec();
    let mut gradient_buffer = vec![0.0f64; x.len()];
    let mut best = x.clone();
    let mut best_value = objective(&x);
    let mut temperatures = Vec::with_capacity(options.steps);
    let ratio = (options.final_temperature / options.initial_temperature)
        .powf(1.0 / options.steps.max(1) as f64);
    let mut temperature = options.initial_temperature;
    for _ in 0..options.steps {
        gradient(&x, &mut gradient_buffer);
        let noise_scale = (2.0 * options.learning_rate * temperature).sqrt();
        for (x_i, g_i) in x.iter_mut().zip(gradient_buffer.iter()) {
            *x_i -= options.learning_rate * g_i;
            *x_i += noise_scale * standard_normal(&mut rng);
        }
        let value = objective(&x);
        if value < best_value {
            best_value = value;
            best = x.clone();
        }
        temperatures.push(temperature);
        temperature *= ratio;
    }
    LangevinOutcome {
        best,
        best_value,
        final_value: objective(&x),
        temperatures,
    }
}

/// Plain gradient descent, for comparison.
pub fn gradient_descent<F, G>(
    objective: F,
    gradient: G,
    initial: &[f64],
    learning_rate: f64,
    steps: usize,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64], &mut [f64]),
{
    let mut x = initial.to_vec();
    let mut gradient_buffer = vec![0.0f64; x.len()];
    let mut best = x.clone();
    let mut best_value = objective(&x);
    for _ in 0..steps {
        gradient(&x, &mut gradient_buffer);
        for (x_i, g_i) in x.iter_mut().zip(gradient_buffer.iter()) {
            *x_i -= learning_rate * g_i;
        }
        let value = objective(&x);
        if value < best_value {
            best_value = value;
            best = x.clone();
        }
    }
    (best, best_value)
}

/// Central finite-difference gradient, for objectives without an analytic one.
pub fn finite_difference_gradient<F>(objective: F, x: &[f64], step: f64, out: &mut [f64])
where
    F: Fn(&[f64]) -> f64,
{
    let mut shifted = x.to_vec();
    for (i, out_i) in out.iter_mut().enumerate() {
        let original = shifted[i];
        shifted[i] = original + step;
        let plus = objective(&shifted);
        shifted[i] = original - step;
        let minus = objective(&shifted);
        shifted[i] = original;
        *out_i = (plus - minus) / (2.0 * step);
    }
}

// ---------------------------------------------------------------------------
// Noise-driven VQE
// ---------------------------------------------------------------------------

use crate::analytic::PauliSum;
use crate::{ParameterizedCircuit, QuantumError};

/// Energy standard deviation of a shot estimator for `hamiltonian` at `shots`
/// shots per point: `sqrt(sum_w |c_w|^2 / shots)`.
///
/// This is the natural energy scale for the Langevin temperature. It is
/// deliberately NOT `T = eta^2 / N`: the shot noise is a variance of the energy
/// estimator, while `exp(-f/T)` needs `T` in energy units, so the temperature
/// is the estimator's standard deviation.
pub fn shot_noise_temperature(hamiltonian: &PauliSum, shots: usize) -> Result<f64, QuantumError> {
    if shots == 0 {
        return Err(QuantumError::InvalidShotCount);
    }
    let mut variance = 0.0f64;
    for coefficient in hamiltonian.terms().values() {
        variance += coefficient.as_complex64().norm_sqr();
    }
    let value = (variance / shots as f64).sqrt();
    if !value.is_finite() {
        return Err(QuantumError::InvalidProbability(value));
    }
    Ok(value)
}

/// Langevin dynamics at a FIXED temperature; returns every iterate after the
/// first. Used to check the fluctuation-dissipation claim that the stationary
/// distribution is `exp(-f/T)`.
pub fn langevin_samples<G>(
    gradient: G,
    initial: &[f64],
    temperature: f64,
    steps: usize,
    learning_rate: f64,
    seed: u64,
) -> Vec<Vec<f64>>
where
    G: Fn(&[f64], &mut [f64]),
{
    let mut rng = SmallRng::seeded(seed);
    let mut x = initial.to_vec();
    let mut buffer = vec![0.0f64; x.len()];
    let noise_scale = (2.0 * learning_rate * temperature).sqrt();
    let mut samples = Vec::with_capacity(steps);
    for _ in 0..steps {
        gradient(&x, &mut buffer);
        for (value, slope) in x.iter_mut().zip(buffer.iter()) {
            *value -= learning_rate * slope;
            *value += noise_scale * standard_normal(&mut rng);
        }
        samples.push(x.clone());
    }
    samples
}

/// Options for [`annealed_langevin_vqe`].
#[derive(Clone, Debug)]
pub struct LangevinVqeOptions {
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// Number of Langevin steps.
    pub steps: usize,
    /// Temperature at the first step, in energy units.
    pub initial_temperature: f64,
    /// Temperature at the last step.
    pub final_temperature: f64,
    /// Seed for the thermal kicks.
    pub seed: u64,
}

impl Default for LangevinVqeOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            steps: 800,
            initial_temperature: 0.5,
            final_temperature: 0.005,
            seed: 42,
        }
    }
}

/// Result of [`annealed_langevin_vqe`].
#[derive(Clone, Debug)]
pub struct LangevinVqeOutcome {
    /// Best parameters visited.
    pub best: Vec<f64>,
    /// Energy at `best`.
    pub best_energy: f64,
    /// Energy at the final iterate.
    pub final_energy: f64,
    /// Best energy after each step.
    pub energies: Vec<f64>,
    /// Exact ground-state energy of the Hamiltonian, when it is available.
    pub exact_energy: Option<f64>,
}

/// Annealed Langevin on a VQE objective: exact parameter-shift gradient plus a
/// fluctuation-dissipation kick `sqrt(2 lr T)`.
///
/// The kick is a resource only when the ansatz landscape is barrier-limited;
/// on a benign landscape it adds variance and loses to plain gradient descent.
pub fn annealed_langevin_vqe(
    circuit: &ParameterizedCircuit,
    hamiltonian: &PauliSum,
    qubits: usize,
    initial: &[f64],
    options: &LangevinVqeOptions,
) -> Result<LangevinVqeOutcome, QuantumError> {
    let operator = hamiltonian
        .to_sparse_operator()
        .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
    let energy = |parameters: &[f64]| -> Result<f64, QuantumError> {
        let state = circuit.run(parameters, qubits)?;
        let ket = state
            .to_ket()
            .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
        Ok(operator
            .expectation(&ket)
            .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?
            .re)
    };

    let mut rng = SmallRng::seeded(options.seed);
    let mut parameters = initial.to_vec();
    let mut best = parameters.clone();
    let mut best_energy = energy(&parameters)?;
    let mut energies = Vec::with_capacity(options.steps);
    let half_pi = std::f64::consts::FRAC_PI_2;
    let ratio = (options.final_temperature / options.initial_temperature)
        .powf(1.0 / options.steps.max(1) as f64);
    let mut temperature = options.initial_temperature;
    for _ in 0..options.steps {
        let mut gradient = vec![0.0f64; parameters.len()];
        for (index, slope) in gradient.iter_mut().enumerate() {
            let mut plus = parameters.clone();
            plus[index] += half_pi;
            let mut minus = parameters.clone();
            minus[index] -= half_pi;
            *slope = 0.5 * (energy(&plus)? - energy(&minus)?);
        }
        let noise_scale = (2.0 * options.learning_rate * temperature).sqrt();
        for (value, slope) in parameters.iter_mut().zip(gradient.iter()) {
            *value -= options.learning_rate * slope;
            *value += noise_scale * standard_normal(&mut rng);
        }
        let value = energy(&parameters)?;
        if value < best_energy {
            best_energy = value;
            best = parameters.clone();
        }
        energies.push(best_energy);
        temperature *= ratio;
    }
    let exact_energy = operator
        .lowest_eigenpairs(1, 1e-10, 200)
        .ok()
        .map(|system| system.eigenvalues()[0]);
    Ok(LangevinVqeOutcome {
        best,
        best_energy,
        final_energy: energy(&parameters)?,
        energies,
        exact_energy,
    })
}

/// Plain gradient descent on the same VQE objective, for comparison.
pub fn gradient_descent_vqe(
    circuit: &ParameterizedCircuit,
    hamiltonian: &PauliSum,
    qubits: usize,
    initial: &[f64],
    learning_rate: f64,
    steps: usize,
) -> Result<(Vec<f64>, f64), QuantumError> {
    let operator = hamiltonian
        .to_sparse_operator()
        .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
    let energy = |parameters: &[f64]| -> Result<f64, QuantumError> {
        let state = circuit.run(parameters, qubits)?;
        let ket = state
            .to_ket()
            .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?;
        Ok(operator
            .expectation(&ket)
            .map_err(|error| QuantumError::UnsupportedOperation(error.to_string()))?
            .re)
    };
    let mut parameters = initial.to_vec();
    let mut best = parameters.clone();
    let mut best_energy = energy(&parameters)?;
    let half_pi = std::f64::consts::FRAC_PI_2;
    for _ in 0..steps {
        let mut gradient = vec![0.0f64; parameters.len()];
        for (index, slope) in gradient.iter_mut().enumerate() {
            let mut plus = parameters.clone();
            plus[index] += half_pi;
            let mut minus = parameters.clone();
            minus[index] -= half_pi;
            *slope = 0.5 * (energy(&plus)? - energy(&minus)?);
        }
        for (value, slope) in parameters.iter_mut().zip(gradient.iter()) {
            *value -= learning_rate * slope;
        }
        let value = energy(&parameters)?;
        if value < best_energy {
            best_energy = value;
            best = parameters.clone();
        }
    }
    Ok((best, best_energy))
}
