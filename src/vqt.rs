//! Variational quantum thermalization (VQT).
//!
//! A thermal state `rho_beta = exp(-beta H)/Z` is mixed, so a pure-state ansatz
//! cannot represent it directly. VQT purifies it: a parameterized unitary acts
//! on a system register plus an ancilla register prepared in the maximally
//! entangled state, and the ancilla is traced out. The parameters minimize the
//! free energy of the reduced state,
//!
//! ```text
//! F(theta) = Tr[rho_S(theta) H] - T S(rho_S(theta)).
//! ```
//!
//! At the exact solution `F = -ln Z / beta`, which is the reference the example
//! and the tests compare against.

use crate::analytic::PauliSum;
use crate::thermal::expectation;
use crate::{DensityMatrix, Gate, ParameterizedCircuit, QuantumError, RotationAxis};

/// Options for [`variational_thermalize`].
#[derive(Clone, Debug)]
pub struct VqtOptions {
    /// System qubits (the ancilla register is the same size).
    pub system_qubits: usize,
    /// Hardware-efficient layers on the system+ancilla register.
    pub layers: usize,
    /// Temperature `T` (the inverse temperature is `1/T`).
    pub temperature: f64,
    /// Gradient-descent learning rate.
    pub learning_rate: f64,
    /// Optimization steps.
    pub steps: usize,
    /// Finite-difference step for the free-energy gradient.
    pub finite_difference_step: f64,
}

impl Default for VqtOptions {
    fn default() -> Self {
        Self {
            system_qubits: 2,
            layers: 3,
            temperature: 1.0,
            learning_rate: 0.05,
            steps: 200,
            finite_difference_step: 1e-3,
        }
    }
}

/// Result of [`variational_thermalize`].
#[derive(Clone, Debug)]
pub struct VqtOutcome {
    /// Optimized ansatz parameters.
    pub parameters: Vec<f64>,
    /// Free energy of the prepared state.
    pub free_energy: f64,
    /// Free energy of the exact Gibbs state.
    pub exact_free_energy: f64,
    /// The prepared reduced state.
    pub state: DensityMatrix,
    /// The exact Gibbs state at the same temperature.
    pub exact_state: DensityMatrix,
    /// Trace distance `1/2 Tr|rho - sigma|` between the two.
    pub trace_distance: f64,
    /// Free energy at every optimization step.
    pub history: Vec<f64>,
}

/// System+ancilla ansatz: a maximally entangled preparation followed by
/// hardware-efficient layers of `Ry` and a CNOT ring.
fn build_ansatz(system_qubits: usize, layers: usize) -> ParameterizedCircuit {
    let total_qubits = system_qubits * 2;
    let parameter_count = total_qubits * layers;
    let mut circuit = ParameterizedCircuit::new(parameter_count);
    for qubit in 0..system_qubits {
        circuit.push(Gate::H, qubit);
    }
    for qubit in 0..system_qubits {
        circuit.cnot(qubit, system_qubits + qubit);
    }
    let mut index = 0;
    for _ in 0..layers {
        for qubit in 0..total_qubits {
            circuit.rotation(RotationAxis::Y, qubit, index);
            index += 1;
        }
        for qubit in 0..total_qubits {
            circuit.cnot(qubit, (qubit + 1) % total_qubits);
        }
    }
    circuit
}

/// Reduced system state of the purified ansatz.
fn reduced_state(
    circuit: &ParameterizedCircuit,
    parameters: &[f64],
    system_qubits: usize,
) -> Result<DensityMatrix, QuantumError> {
    let total_qubits = system_qubits * 2;
    let pure = circuit.run(parameters, total_qubits)?;
    let keep: Vec<usize> = (0..system_qubits).collect();
    DensityMatrix::from_state(&pure).partial_trace(&keep)
}

/// Free energy `Tr[rho H] - T S(rho)` of a mixed state.
pub fn free_energy(
    rho: &DensityMatrix,
    hamiltonian: &PauliSum,
    temperature: f64,
) -> Result<f64, QuantumError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(QuantumError::InvalidProbability(temperature));
    }
    let energy = expectation(rho, hamiltonian)?;
    let entropy = rho.von_neumann_entropy()?;
    Ok(energy - temperature * entropy)
}

/// Trace distance `1/2 Tr|rho - sigma|` between two density matrices.
pub fn trace_distance(first: &DensityMatrix, second: &DensityMatrix) -> Result<f64, QuantumError> {
    let dimension = first.dimension();
    if second.dimension() != dimension || first.qubits() != second.qubits() {
        return Err(QuantumError::InvalidSubsystemSelection);
    }
    use faer::c64;
    use faer::diag::Diag;
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::evd::{ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
    use faer::{Mat, Par};
    let matrix = Mat::from_fn(dimension, dimension, |row, col| {
        let difference =
            first.entries()[row * dimension + col] - second.entries()[row * dimension + col];
        c64::new(difference.re, difference.im)
    });
    let par = Par::Seq;
    let params = Default::default();
    let mut eigenvalues = Diag::<c64>::zeros(dimension);
    let scratch = self_adjoint_evd_scratch::<c64>(dimension, ComputeEigenvectors::No, par, params);
    let mut memory = MemBuffer::new(scratch);
    let stack = MemStack::new(&mut memory);
    let _ = self_adjoint_evd(
        matrix.as_ref(),
        eigenvalues.as_mut(),
        None,
        par,
        stack,
        params,
    );
    let total: f64 = (0..dimension)
        .map(|index| eigenvalues[index].re.abs())
        .sum();
    Ok(0.5 * total)
}

/// Minimize the free energy of the purified state by finite-difference
/// gradient descent.
pub fn variational_thermalize(
    hamiltonian: &PauliSum,
    options: &VqtOptions,
) -> Result<VqtOutcome, QuantumError> {
    if options.system_qubits == 0 {
        return Err(QuantumError::ZeroQubits);
    }
    let inverse_temperature = 1.0 / options.temperature;
    let exact_state = crate::thermal::gibbs_state(hamiltonian, inverse_temperature)?;
    let exact_free_energy = free_energy(&exact_state, hamiltonian, options.temperature)?;

    let circuit = build_ansatz(options.system_qubits, options.layers);
    let parameter_count = circuit.parameter_count();
    let mut parameters = vec![0.0f64; parameter_count];

    let objective = |values: &[f64]| -> Result<f64, QuantumError> {
        let rho = reduced_state(&circuit, values, options.system_qubits)?;
        free_energy(&rho, hamiltonian, options.temperature)
    };

    let mut history = Vec::with_capacity(options.steps);
    let step_size = options.finite_difference_step;
    for _ in 0..options.steps {
        let current = objective(&parameters)?;
        history.push(current);
        let mut gradient = vec![0.0f64; parameter_count];
        for index in 0..parameter_count {
            let mut plus = parameters.clone();
            plus[index] += step_size;
            let mut minus = parameters.clone();
            minus[index] -= step_size;
            gradient[index] = (objective(&plus)? - objective(&minus)?) / (2.0 * step_size);
        }
        for (value, slope) in parameters.iter_mut().zip(gradient.iter()) {
            *value -= options.learning_rate * slope;
        }
    }

    let state = reduced_state(&circuit, &parameters, options.system_qubits)?;
    let prepared_free_energy = free_energy(&state, hamiltonian, options.temperature)?;
    let distance = trace_distance(&state, &exact_state)?;
    Ok(VqtOutcome {
        parameters,
        free_energy: prepared_free_energy,
        exact_free_energy,
        state,
        exact_state,
        trace_distance: distance,
        history,
    })
}

/// Convenience: the exact Gibbs free energy `-ln Z / beta`.
pub fn exact_free_energy(hamiltonian: &PauliSum, temperature: f64) -> Result<f64, QuantumError> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(QuantumError::InvalidProbability(temperature));
    }
    crate::thermal::free_energy(hamiltonian, 1.0 / temperature)
}
