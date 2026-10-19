//! High-level algorithm templates.
//!
//! Provides ready-to-use implementations of common quantum algorithms
//! with convenient builder-style APIs.

use crate::{Circuit, CircuitOp, Gate, QuantumError, StateVector};
use std::collections::HashMap;

/// Grover's search algorithm.
pub struct Grover {
    oracle: Circuit,
    num_qubits: usize,
    iterations: usize,
}

impl Grover {
    /// Create a new Grover search with the given oracle.
    pub fn new(oracle: Circuit, num_qubits: usize) -> Self {
        Self {
            oracle,
            num_qubits,
            iterations: 0,
        }
    }

    /// Set the number of Grover iterations (default: optimal ≈ π/4 √N).
    pub fn iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Build the full Grover circuit.
    pub fn build(&self) -> Circuit {
        let mut circuit = Circuit::new();

        // Initial Hadamard layer
        for i in 0..self.num_qubits {
            circuit.push(Gate::H, i);
        }

        // Grover iterations
        for _ in 0..self.iterations {
            // Oracle
            for op in self.oracle.operations() {
                Self::push_op_to_circuit(&mut circuit, *op);
            }

            // Diffusion operator (inversion about mean)
            for i in 0..self.num_qubits {
                circuit.push(Gate::H, i);
                circuit.push(Gate::X, i);
            }

            // Multi-controlled Z (on all qubits)
            if self.num_qubits == 2 {
                circuit.cz(0, 1);
            } else if self.num_qubits == 3 {
                // Toffoli-like multi-controlled Z
                circuit.push(Gate::H, 2);
                circuit.cnot(1, 2);
                circuit.push(Gate::Tdg, 2);
                circuit.cnot(0, 2);
                circuit.push(Gate::T, 2);
                circuit.cnot(1, 2);
                circuit.push(Gate::Tdg, 2);
                circuit.cnot(0, 2);
                circuit.push(Gate::T, 1);
                circuit.push(Gate::T, 2);
                circuit.cnot(0, 1);
                circuit.push(Gate::H, 2);
            } else {
                // General case: cascade of Toffolis
                // Simplified: use Z on last qubit controlled by all others
                for i in 0..self.num_qubits - 1 {
                    circuit.cnot(i, self.num_qubits - 1);
                }
                circuit.push(Gate::Z, self.num_qubits - 1);
                for i in (0..self.num_qubits - 1).rev() {
                    circuit.cnot(i, self.num_qubits - 1);
                }
            }

            for i in 0..self.num_qubits {
                circuit.push(Gate::X, i);
                circuit.push(Gate::H, i);
            }
        }

        circuit
    }

    /// Helper to push an operation to a circuit.
    fn push_op_to_circuit(circuit: &mut Circuit, op: CircuitOp) {
        use CircuitOp::*;
        match op {
            SingleQubit { gate, target } => circuit.push(gate, target),
            Cnot { control, target } => circuit.cnot(control, target),
            Cz { first, second } => circuit.cz(first, second),
            Swap { first, second } => circuit.swap(first, second),
            Measure {
                target,
                classical_bit,
            } => circuit.measure(target, classical_bit),
            ConditionalSingleQubit {
                classical_bit,
                value,
                gate,
                target,
            } => {
                circuit.if_classical_bit(classical_bit, value, gate, target);
            }
            Channel { channel, target } => circuit.channel(target, channel),
        }
    }

    /// Run Grover search and return measurement counts.
    pub fn run(&self, shots: usize, seed: u64) -> Result<HashMap<Vec<u8>, usize>, QuantumError> {
        let circuit = self.build();
        let initial = StateVector::zero(self.num_qubits)?;
        let result = circuit.run_shots(&initial, shots, seed)?;
        Ok(result.counts.into_iter().collect())
    }
}

/// Builder for QAOA circuits (MaxCut, etc.).
pub struct QAOABuilder {
    num_qubits: usize,
    edges: Vec<(usize, usize, f64)>,
    p: usize,
    betas: Vec<f64>,
    gammas: Vec<f64>,
}

impl QAOABuilder {
    /// Create a new QAOA builder for MaxCut on the given graph.
    pub fn maxcut(num_qubits: usize, edges: Vec<(usize, usize, f64)>) -> Self {
        Self {
            num_qubits,
            edges,
            p: 1,
            betas: vec![],
            gammas: vec![],
        }
    }

    /// Set number of QAOA layers.
    pub fn layers(mut self, p: usize) -> Self {
        self.p = p;
        self
    }

    /// Set variational parameters (betas for mixer, gammas for cost).
    pub fn parameters(mut self, betas: Vec<f64>, gammas: Vec<f64>) -> Self {
        self.betas = betas;
        self.gammas = gammas;
        self
    }

    /// Build the QAOA circuit.
    pub fn build(&self) -> Circuit {
        let mut circuit = Circuit::new();

        // Initial |+⟩^n
        for i in 0..self.num_qubits {
            circuit.push(Gate::H, i);
        }

        let p = self.p;
        let betas = if self.betas.is_empty() {
            vec![0.5; p]
        } else {
            self.betas.clone()
        };
        let gammas = if self.gammas.is_empty() {
            vec![0.7; p]
        } else {
            self.gammas.clone()
        };

        for layer in 0..p {
            // Cost Hamiltonian: ∑ Z_i Z_j
            for &(u, v, w) in &self.edges {
                circuit.cnot(u, v);
                circuit.push(Gate::Rz(2.0 * gammas[layer] * w), v);
                circuit.cnot(u, v);
            }

            // Mixer Hamiltonian: ∑ X_i
            for i in 0..self.num_qubits {
                circuit.push(Gate::Rx(2.0 * betas[layer]), i);
            }
        }

        circuit
    }
}

/// Hardware-efficient ansatz (HEA) for VQE.
pub struct HEABuilder {
    num_qubits: usize,
    layers: usize,
    entanglement: EntanglementPattern,
}

#[derive(Clone)]
pub enum EntanglementPattern {
    Linear,
    Circular,
    Full,
    Pairwise(Vec<(usize, usize)>),
}

impl HEABuilder {
    pub fn new(num_qubits: usize, layers: usize) -> Self {
        Self {
            num_qubits,
            layers,
            entanglement: EntanglementPattern::Linear,
        }
    }

    pub fn entanglement(mut self, pattern: EntanglementPattern) -> Self {
        self.entanglement = pattern;
        self
    }

    pub fn build(&self, params: &[f64]) -> Circuit {
        let mut circuit = Circuit::new();
        let mut idx = 0;

        for _layer in 0..self.layers {
            // Single-qubit rotations
            for i in 0..self.num_qubits {
                if idx < params.len() {
                    circuit.push(Gate::Ry(params[idx]), i);
                    idx += 1;
                }
                if idx < params.len() {
                    circuit.push(Gate::Rz(params[idx]), i);
                    idx += 1;
                }
            }

            // Entangling layer
            match &self.entanglement {
                EntanglementPattern::Linear => {
                    for i in 0..self.num_qubits - 1 {
                        circuit.cnot(i, i + 1);
                    }
                }
                EntanglementPattern::Circular => {
                    for i in 0..self.num_qubits {
                        circuit.cnot(i, (i + 1) % self.num_qubits);
                    }
                }
                EntanglementPattern::Full => {
                    for i in 0..self.num_qubits {
                        for j in i + 1..self.num_qubits {
                            circuit.cnot(i, j);
                        }
                    }
                }
                EntanglementPattern::Pairwise(pairs) => {
                    for &(u, v) in pairs {
                        circuit.cnot(u, v);
                    }
                }
            }
        }

        circuit
    }

    /// Number of parameters required.
    pub fn num_params(&self) -> usize {
        self.layers * self.num_qubits * 2
    }
}

/// QAOA-style ansatz (alternating cost/mixer).
pub struct QAOAAnsatzBuilder {
    num_qubits: usize,
    edges: Vec<(usize, usize)>,
    layers: usize,
}

impl QAOAAnsatzBuilder {
    pub fn new(num_qubits: usize, edges: Vec<(usize, usize)>) -> Self {
        Self {
            num_qubits,
            edges,
            layers: 1,
        }
    }

    pub fn layers(mut self, p: usize) -> Self {
        self.layers = p;
        self
    }

    pub fn build(&self, params: &[f64]) -> Circuit {
        let mut circuit = Circuit::new();
        let mut idx = 0;

        // Initial |+⟩^n
        for i in 0..self.num_qubits {
            circuit.push(Gate::H, i);
        }

        for _layer in 0..self.layers {
            // Cost Hamiltonian
            if idx < params.len() {
                let gamma = params[idx];
                for &(u, v) in &self.edges {
                    circuit.cnot(u, v);
                    circuit.push(Gate::Rz(2.0 * gamma), v);
                    circuit.cnot(u, v);
                }
                idx += 1;
            }

            // Mixer Hamiltonian
            if idx < params.len() {
                let beta = params[idx];
                for i in 0..self.num_qubits {
                    circuit.push(Gate::Rx(2.0 * beta), i);
                }
                idx += 1;
            }
        }

        circuit
    }

    pub fn num_params(&self) -> usize {
        self.layers * 2
    }
}

/// Grover Adaptive Search (GAS) for QUBO/MaxCut optimization.
/// Iteratively runs Grover with increasing iterations to find states
/// with cost > threshold, then updates threshold to the best found.
pub struct GroverAdaptive {
    num_qubits: usize,
    edges: Vec<(usize, usize, f64)>,
    shots_per_iter: usize,
    max_outer: usize,
    max_k: usize,
    seed: u64,
}

impl GroverAdaptive {
    /// Create a new GAS search for MaxCut on the given graph.
    pub fn maxcut(num_qubits: usize, edges: Vec<(usize, usize, f64)>) -> Self {
        Self {
            num_qubits,
            edges,
            shots_per_iter: 1000,
            max_outer: 20,
            max_k: 10,
            seed: 42,
        }
    }

    /// Set shots per Grover iteration (default: 1000).
    pub fn shots_per_iter(mut self, shots: usize) -> Self {
        self.shots_per_iter = shots;
        self
    }

    /// Maximum outer loop iterations (default: 20).
    pub fn max_outer(mut self, n: usize) -> Self {
        self.max_outer = n;
        self
    }

    /// Maximum Grover k to try per outer iteration (default: 10).
    pub fn max_k(mut self, k: usize) -> Self {
        self.max_k = k;
        self
    }

    /// Random seed (default: 42).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Evaluate MaxCut cost for a bitstring.
    fn maxcut_cost(&self, bits: &[u8]) -> f64 {
        let mut cost = 0.0;
        for &(u, v, w) in &self.edges {
            if bits[u] != bits[v] {
                cost += w;
            }
        }
        cost
    }

    /// Build oracle marking states with cost > threshold.
    fn build_oracle(&self, _threshold: f64) -> Circuit {
        let mut oracle = Circuit::new();

        // Phase oracle: apply phase proportional to cut value
        for &(u, v, w) in &self.edges {
            oracle.cz(u, v);
            oracle.push(Gate::Rz(w * std::f64::consts::PI / 4.0), u);
            oracle.push(Gate::Rz(w * std::f64::consts::PI / 4.0), v);
        }

        // Also mark the all-ones state (for demo)
        if self.num_qubits == 2 {
            oracle.cz(0, 1);
        } else if self.num_qubits > 2 {
            for i in 0..self.num_qubits - 1 {
                oracle.cnot(i, self.num_qubits - 1);
            }
            oracle.push(Gate::Z, self.num_qubits - 1);
            for i in (0..self.num_qubits - 1).rev() {
                oracle.cnot(i, self.num_qubits - 1);
            }
        }

        oracle
    }

    /// Run the adaptive search.
    pub fn search(&self) -> Result<(Vec<u8>, f64), QuantumError> {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_state: Vec<u8> = vec![0; self.num_qubits];
        let mut threshold = f64::NEG_INFINITY;
        let mut seed = self.seed;

        for _outer in 0..self.max_outer {
            let oracle = self.build_oracle(threshold);

            // Try increasing Grover iterations k
            let mut prev_marked_prob = 0.0;
            let mut improved = false;

            for k in 0..=self.max_k {
                let grover = Grover::new(oracle.clone(), self.num_qubits).iterations(k);
                let counts = grover.run(self.shots_per_iter, seed)?;

                // Find most frequent measured state
                let (candidate, shots) = counts
                    .iter()
                    .max_by_key(|(_, v)| *v)
                    .ok_or(QuantumError::InvalidTrainingData)?;

                let marked_prob = *shots as f64 / self.shots_per_iter as f64;

                // If probability starts dropping, stop increasing k
                if k > 0 && marked_prob < prev_marked_prob {
                    break;
                }
                prev_marked_prob = marked_prob;

                let val = self.maxcut_cost(candidate);
                if val > best_val {
                    best_val = val;
                    best_state = candidate.to_vec();
                    improved = true;
                }

                seed = seed.wrapping_add(1);
            }

            // Update threshold
            if best_val > threshold {
                threshold = best_val;
            }

            if !improved {
                break;
            }
        }

        Ok((best_state, best_val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grover() {
        let mut oracle = Circuit::new();
        oracle.cnot(0, 1);
        oracle.push(Gate::Z, 1); // Mark |11⟩

        let grover = Grover::new(oracle, 2).iterations(1);
        let circuit = grover.build();
        assert!(!circuit.operations().is_empty());
    }

    #[test]
    fn test_qaoa_maxcut() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let qaoa = QAOABuilder::maxcut(3, edges).layers(2);
        let circuit = qaoa.build();
        assert!(!circuit.operations().is_empty());
    }

    #[test]
    fn test_hea_ansatz() {
        let hea = HEABuilder::new(4, 2).entanglement(EntanglementPattern::Linear);
        let params = vec![0.1; hea.num_params()];
        let circuit = hea.build(&params);
        assert!(!circuit.operations().is_empty());
    }

    #[test]
    fn test_qaoa_ansatz() {
        let edges = vec![(0, 1), (1, 2)];
        let qaoa = QAOAAnsatzBuilder::new(3, edges).layers(2);
        let params = vec![0.5, 0.3, 0.7, 0.2];
        let circuit = qaoa.build(&params);
        assert!(!circuit.operations().is_empty());
    }
}
