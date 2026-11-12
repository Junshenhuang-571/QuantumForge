//! Session state for the REPL.

use quantumforge::{
    Complex64, DensityMatrix, Gate, ParameterizedCircuit, QmlTrainingConfig,
    RotationAxis, SingleQubitObservable, StateVector, VariationalQnn,
};
use quantumforge::analytic::Assumptions;
use quantumforge::tensor_network::{DMRG, MPS, TEBD};
use quantumforge::qec::QecCode;
use std::collections::HashMap;

pub struct Session {
    pub qubits: usize,
    pub circuit: String,
    pub state: StateVector,
    /// When Some, `state`/`probs`/`expect` read from this mixed state instead.
    pub density: Option<DensityMatrix>,
    pub assume: Assumptions,
    pub shots: usize,
    pub seed: u64,
    /// TEBD imaginary time evolution state
    pub tebd: Option<TEBD>,
    /// DMRG variational ground state search state
    pub dmrg: Option<DMRG>,
    /// QEC code currently loaded
    pub qec_code: Option<QecCode>,
}

impl Session {
    pub fn new(qubits: usize) -> Self {
        Self {
            qubits,
            circuit: String::new(),
            state: StateVector::zero(qubits).unwrap(),
            density: None,
            assume: Assumptions::default(),
            shots: 1024,
            seed: 42,
            tebd: None,
            dmrg: None,
            qec_code: None,
        }
    }

    pub fn reset_qubits(&mut self, n: usize) {
        self.qubits = n;
        self.circuit.clear();
        self.state = StateVector::zero(n).unwrap();
        self.density = None;
        self.tebd = None;
        self.dmrg = None;
        self.qec_code = None;
    }

    pub fn clear_circuit(&mut self) {
        self.circuit.clear();
    }
}