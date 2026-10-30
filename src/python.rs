//! Python bindings for QuantumForge using PyO3.
//!
//! Enable with `--features python`. Provides a `quantumforge` Python module
//! with classes for Circuit, StateVector, DensityMatrix, and the REPL commands.

#[cfg(feature = "pyo3")]
pub mod python_bindings {
    use crate::{
        Circuit, DensityMatrix, Gate, ShotResult, StateVector,
        algorithms_templates::{
            EntanglementPattern, Grover, HEABuilder, QAOAAnsatzBuilder, QAOABuilder,
        },
        optimizer::{OptimizeConfig, OptimizeResult},
        qasm::parse_qasm,
        qec::{PauliString, QecCode},
        tensor_network::MPS,
    };
    use pyo3::prelude::*;
    use std::collections::BTreeMap;
    use std::mem;

    /// Python wrapper for Circuit
    #[pyclass]
    #[derive(Clone)]
    struct PyCircuit {
        inner: Circuit,
    }

    #[pymethods]
    impl PyCircuit {
        #[new]
        fn new() -> Self {
            Self {
                inner: Circuit::new(),
            }
        }

        #[staticmethod]
        fn from_qasm(qasm: &str) -> PyResult<Self> {
            parse_qasm(qasm)
                .map(|c| Self { inner: c })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn add_gate(&mut self, gate: &str, target: usize) -> PyResult<()> {
            let g = parse_gate(gate)?;
            self.inner.push(g, target);
            Ok(())
        }

        fn add_cnot(&mut self, control: usize, target: usize) {
            self.inner.cnot(control, target);
        }

        fn add_cz(&mut self, first: usize, second: usize) {
            self.inner.cz(first, second);
        }

        fn add_swap(&mut self, first: usize, second: usize) {
            self.inner.swap(first, second);
        }

        fn add_rx(&mut self, angle: f64, target: usize) {
            self.inner.push(Gate::Rx(angle), target);
        }

        fn add_ry(&mut self, angle: f64, target: usize) {
            self.inner.push(Gate::Ry(angle), target);
        }

        fn add_rz(&mut self, angle: f64, target: usize) {
            self.inner.push(Gate::Rz(angle), target);
        }

        fn measure(&mut self, target: usize, classical_bit: usize) {
            self.inner.measure(target, classical_bit);
        }

        fn run(&self, state: &PyStateVector) -> PyResult<PyStateVector> {
            let result = self
                .inner
                .run(state.inner.clone())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyStateVector { inner: result })
        }

        fn run_shots(
            &self,
            state: &PyStateVector,
            shots: usize,
            seed: u64,
        ) -> PyResult<PyShotResult> {
            let result = self
                .inner
                .run_shots(&state.inner, shots, seed)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(PyShotResult { inner: result })
        }

        fn to_svg(&self) -> String {
            self.inner.to_svg()
        }

        fn draw(&self) -> String {
            self.inner.draw()
        }

        fn num_operations(&self) -> usize {
            self.inner.operations().len()
        }

        fn to_json(&self) -> PyResult<String> {
            self.inner
                .to_json()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        #[staticmethod]
        fn from_json(json: &str) -> PyResult<Self> {
            Circuit::from_json(json)
                .map(|c| Self { inner: c })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }
    }

    /// Python wrapper for StateVector
    #[pyclass]
    #[derive(Clone)]
    struct PyStateVector {
        inner: StateVector,
    }

    #[pymethods]
    impl PyStateVector {
        #[staticmethod]
        fn zero(qubits: usize) -> PyResult<Self> {
            StateVector::zero(qubits)
                .map(|s| Self { inner: s })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn amplitudes(&self) -> PyResult<Vec<(f64, f64)>> {
            Ok(self
                .inner
                .amplitudes()
                .iter()
                .map(|c| (c.re, c.im))
                .collect())
        }

        fn num_qubits(&self) -> usize {
            self.inner.qubits()
        }

        fn probability(&self, index: usize) -> f64 {
            let amp = self
                .inner
                .amplitudes()
                .get(index)
                .copied()
                .unwrap_or(crate::Complex64::ZERO);
            amp.norm_sqr()
        }

        fn probabilities(&self) -> Vec<f64> {
            self.inner.probabilities()
        }

        fn __repr__(&self) -> String {
            format!("StateVector({} qubits)", self.inner.qubits())
        }
    }

    /// Python wrapper for DensityMatrix
    #[pyclass]
    #[derive(Clone)]
    struct PyDensityMatrix {
        inner: DensityMatrix,
    }

    #[pymethods]
    impl PyDensityMatrix {
        #[staticmethod]
        fn maximally_mixed(qubits: usize) -> PyResult<Self> {
            DensityMatrix::maximally_mixed(qubits)
                .map(|d| Self { inner: d })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        #[staticmethod]
        fn from_state(state: &PyStateVector) -> Self {
            Self {
                inner: DensityMatrix::from_state(&state.inner),
            }
        }

        fn apply_gate(&mut self, gate: &str, target: usize) -> PyResult<()> {
            let g = parse_gate(gate)?;
            self.inner
                .apply(g, target)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        fn apply_cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
            self.inner
                .apply_cnot(control, target)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        fn apply_channel(&mut self, channel: &str, qubit: usize, param: f64) -> PyResult<()> {
            let ch = match channel {
                "depolarizing" => crate::NoiseChannel::Depolarizing { probability: param },
                "amplitude_damping" => crate::NoiseChannel::AmplitudeDamping { gamma: param },
                "phase_damping" => crate::NoiseChannel::PhaseDamping { gamma: param },
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unknown channel: {}",
                        channel
                    )));
                }
            };
            self.inner
                .apply_channel(ch, qubit)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        fn partial_trace(&self, keep_qubits: Vec<usize>) -> PyResult<Self> {
            self.inner
                .partial_trace(&keep_qubits)
                .map(|d| Self { inner: d })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        fn purity(&self) -> f64 {
            self.inner.purity()
        }

        fn num_qubits(&self) -> usize {
            self.inner.qubits()
        }

        fn __repr__(&self) -> String {
            format!("DensityMatrix({} qubits)", self.inner.qubits())
        }
    }

    /// Python wrapper for ShotResult
    #[pyclass]
    struct PyShotResult {
        inner: ShotResult,
    }

    #[pymethods]
    impl PyShotResult {
        fn counts(&self) -> PyResult<Vec<(Vec<u8>, usize)>> {
            Ok(self
                .inner
                .counts
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect())
        }

        fn shots(&self) -> usize {
            self.inner.shots
        }

        fn __repr__(&self) -> String {
            format!("ShotResult({} shots)", self.inner.shots)
        }
    }

    /// Python wrapper for PauliString
    #[pyclass]
    #[derive(Clone)]
    struct PyPauliString {
        inner: crate::qec::PauliString,
    }

    #[pymethods]
    impl PyPauliString {
        #[new]
        fn new(n: usize) -> Self {
            Self {
                inner: crate::qec::PauliString::new(n),
            }
        }

        fn set(&mut self, qubit: usize, pauli: u8) {
            self.inner.set(qubit, pauli);
        }

        fn weight(&self) -> usize {
            self.inner.weight()
        }

        fn __str__(&self) -> String {
            self.inner.to_string()
        }

        fn __repr__(&self) -> String {
            format!("PauliString({})", self.inner.to_string())
        }
    }

    /// Python wrapper for QecCode
    #[pyclass]
    struct PyQecCode {
        inner: crate::qec::QecCode,
    }

    #[pymethods]
    impl PyQecCode {
        fn n_physical(&self) -> usize {
            self.inner.n_physical
        }

        fn n_logical(&self) -> usize {
            self.inner.n_logical
        }

        fn stabilizers(&self) -> Vec<String> {
            self.inner
                .stabilizers
                .iter()
                .map(|s| s.to_string())
                .collect()
        }

        fn __repr__(&self) -> String {
            format!(
                "QecCode(n_physical={}, n_logical={}, stabilizers={})",
                self.inner.n_physical,
                self.inner.n_logical,
                self.inner.stabilizers.len()
            )
        }
    }

    /// Grover search
    #[pyclass]
    struct PyGrover {
        inner: Grover,
    }

    #[pymethods]
    impl PyGrover {
        #[new]
        fn new(oracle: PyCircuit, num_qubits: usize) -> Self {
            Self {
                inner: Grover::new(oracle.inner, num_qubits),
            }
        }

        fn iterations(&mut self, n: usize) {
            self.inner =
                mem::replace(&mut self.inner, Grover::new(Circuit::new(), 0)).iterations(n);
        }

        fn build(&self) -> PyCircuit {
            PyCircuit {
                inner: self.inner.build(),
            }
        }

        fn run(&self, shots: usize, seed: u64) -> PyResult<PyShotResult> {
            let counts: std::collections::HashMap<Vec<u8>, usize> = self
                .inner
                .run(shots, seed)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let mut btree: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
            for (k, v) in counts {
                btree.insert(k, v);
            }
            Ok(PyShotResult {
                inner: ShotResult {
                    shots,
                    counts: btree,
                },
            })
        }
    }

    /// Builder for QAOA circuits (MaxCut, etc.)
    #[pyclass]
    struct PyQAOABuilder {
        inner: QAOABuilder,
    }

    #[pymethods]
    impl PyQAOABuilder {
        #[new]
        fn new(num_qubits: usize, edges: Vec<(usize, usize, f64)>) -> Self {
            Self {
                inner: QAOABuilder::maxcut(num_qubits, edges),
            }
        }

        fn layers(&mut self, p: usize) {
            let old = mem::replace(&mut self.inner, QAOABuilder::maxcut(0, vec![]));
            self.inner = old.layers(p);
        }

        fn parameters(&mut self, betas: Vec<f64>, gammas: Vec<f64>) {
            let old = mem::replace(&mut self.inner, QAOABuilder::maxcut(0, vec![]));
            self.inner = old.parameters(betas, gammas);
        }

        fn build(&self) -> PyCircuit {
            PyCircuit {
                inner: self.inner.build(),
            }
        }
    }

    /// Hardware-efficient ansatz (HEA) for VQE.
    #[pyclass]
    struct PyHEABuilder {
        inner: HEABuilder,
    }

    #[pymethods]
    impl PyHEABuilder {
        #[new]
        fn new(num_qubits: usize, layers: usize) -> Self {
            Self {
                inner: HEABuilder::new(num_qubits, layers),
            }
        }

        fn entanglement(&mut self, pattern: &str) -> PyResult<()> {
            let old = mem::replace(&mut self.inner, HEABuilder::new(0, 0));
            self.inner = old.entanglement(match pattern {
                "linear" => EntanglementPattern::Linear,
                "circular" => EntanglementPattern::Circular,
                "full" => EntanglementPattern::Full,
                "pairwise" => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "pairwise not yet supported in Python",
                    ));
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Unknown entanglement pattern",
                    ));
                }
            });
            Ok(())
        }

        fn build(&self, params: Vec<f64>) -> PyResult<PyCircuit> {
            if params.len() != self.inner.num_params() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Expected {} parameters, got {}",
                    self.inner.num_params(),
                    params.len()
                )));
            }
            Ok(PyCircuit {
                inner: self.inner.build(&params),
            })
        }

        fn num_params(&self) -> usize {
            self.inner.num_params()
        }
    }

    /// QAOA-style ansatz (alternating cost/mixer).
    #[pyclass]
    struct PyQAOAAnsatzBuilder {
        inner: QAOAAnsatzBuilder,
    }

    #[pymethods]
    impl PyQAOAAnsatzBuilder {
        #[new]
        fn new(num_qubits: usize, edges: Vec<(usize, usize)>) -> Self {
            Self {
                inner: QAOAAnsatzBuilder::new(num_qubits, edges),
            }
        }

        fn layers(&mut self, p: usize) {
            let old = mem::replace(&mut self.inner, QAOAAnsatzBuilder::new(0, vec![]));
            self.inner = old.layers(p);
        }

        fn build(&self, params: Vec<f64>) -> PyResult<PyCircuit> {
            if params.len() != self.inner.num_params() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Expected {} parameters, got {}",
                    self.inner.num_params(),
                    params.len()
                )));
            }
            Ok(PyCircuit {
                inner: self.inner.build(&params),
            })
        }

        fn num_params(&self) -> usize {
            self.inner.num_params()
        }
    }

    /// Optimizer
    #[pyclass]
    struct PyOptimizeConfig {
        inner: OptimizeConfig,
    }

    #[pymethods]
    impl PyOptimizeConfig {
        #[staticmethod]
        fn all() -> Self {
            Self {
                inner: OptimizeConfig::all(),
            }
        }

        #[staticmethod]
        fn conservative() -> Self {
            Self {
                inner: OptimizeConfig::conservative(),
            }
        }
    }

    #[pyfunction]
    fn optimize_circuit_py(
        circuit: &PyCircuit,
        config: Option<&PyOptimizeConfig>,
    ) -> PyResult<PyCircuit> {
        let config = config
            .map(|c| c.inner.clone())
            .unwrap_or(OptimizeConfig::all());
        let mut circuit = circuit.inner.clone();
        let _result: OptimizeResult = crate::optimizer::optimize_circuit(&mut circuit, &config);
        Ok(PyCircuit { inner: circuit })
    }

    /// QEC codes
    #[pyfunction]
    fn repetition_code_bitflip(n: usize) -> PyResult<PyQecCode> {
        Ok(PyQecCode {
            inner: crate::qec::repetition_code_bitflip(n),
        })
    }

    #[pyfunction]
    fn repetition_code_phaseflip(n: usize) -> PyResult<PyQecCode> {
        Ok(PyQecCode {
            inner: crate::qec::repetition_code_phaseflip(n),
        })
    }

    #[pyfunction]
    fn shor_code() -> PyQecCode {
        PyQecCode {
            inner: crate::qec::shor_code(),
        }
    }

    #[pyfunction]
    fn steane_code() -> PyQecCode {
        PyQecCode {
            inner: crate::qec::steane_code(),
        }
    }

    #[pyfunction]
    fn surface_code_planar(d: usize) -> PyQecCode {
        PyQecCode {
            inner: crate::qec::surface_code_planar(d),
        }
    }

    #[pyfunction]
    fn surface_code_rotated(d: usize) -> PyQecCode {
        PyQecCode {
            inner: crate::qec::surface_code_rotated(d),
        }
    }

    #[pyfunction]
    fn encode_repetition_bitflip(state: &PyStateVector, n: usize) -> PyResult<PyStateVector> {
        let result = crate::qec::encode_repetition_bitflip(&state.inner, n)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(PyStateVector { inner: result })
    }

    /// Tensor networks
    #[pyclass]
    struct PyMPS {
        inner: crate::tensor_network::MPS,
    }

    #[pymethods]
    impl PyMPS {
        #[staticmethod]
        fn zero_state(n_sites: usize, max_bond: usize) -> Self {
            Self {
                inner: crate::tensor_network::MPS::zero_state(n_sites, max_bond),
            }
        }

        #[staticmethod]
        fn random(n_sites: usize, phys_dim: usize, max_bond: usize) -> Self {
            Self {
                inner: crate::tensor_network::MPS::random(n_sites, phys_dim, max_bond),
            }
        }

        fn apply_single_qubit_gate(
            &mut self,
            gate: Vec<Vec<(f64, f64)>>,
            site: usize,
        ) -> PyResult<()> {
            let mut gate_arr = [[crate::Complex64::ZERO; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    let (re, im) = gate[i][j];
                    gate_arr[i][j] = crate::Complex64::new(re, im);
                }
            }
            self.inner
                .apply_single_qubit_gate(&gate_arr, site)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        }

        fn entanglement_entropy(&self, cut: usize) -> f64 {
            self.inner.entanglement_entropy(cut)
        }

        fn num_sites(&self) -> usize {
            self.inner.n_sites
        }
    }

    /// Tensor
    #[pyclass]
    struct PyTensor {
        inner: crate::tensor_network::Tensor,
    }

    #[pymethods]
    impl PyTensor {
        #[staticmethod]
        fn new(shape: [usize; 3]) -> Self {
            Self {
                inner: crate::tensor_network::Tensor::new(shape),
            }
        }

        #[staticmethod]
        fn identity(d: usize) -> Self {
            Self {
                inner: crate::tensor_network::Tensor::identity(d),
            }
        }

        fn shape(&self) -> [usize; 3] {
            self.inner.shape
        }
    }

    /// TEBD
    #[pyclass]
    struct PyTEBD {
        inner: crate::tensor_network::TEBD,
    }

    #[pymethods]
    impl PyTEBD {
        #[new]
        fn new(mps: &PyMPS, dt: f64, max_bond: usize, truncation_tol: f64) -> Self {
            Self {
                inner: crate::tensor_network::TEBD::new(
                    mps.inner.clone(),
                    dt,
                    max_bond,
                    truncation_tol,
                ),
            }
        }

        fn step(&mut self) -> PyResult<()> {
            self.inner
                .step()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        }

        fn find_ground_state(&mut self, max_steps: usize, _energy_tol: f64) -> PyResult<f64> {
            self.inner
                .find_ground_state(max_steps, _energy_tol)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        }
    }

    /// DMRG
    #[pyclass]
    struct PyDMRG {
        inner: crate::tensor_network::DMRG,
    }

    #[pymethods]
    impl PyDMRG {
        #[new]
        fn new(mps: &PyMPS, max_bond: usize, truncation_tol: f64, max_sweeps: usize) -> Self {
            Self {
                inner: crate::tensor_network::DMRG::new(
                    mps.inner.clone(),
                    max_bond,
                    truncation_tol,
                    max_sweeps,
                ),
            }
        }

        fn run(&mut self) -> PyResult<f64> {
            // DMRG::run requires Hamiltonian gates; use empty as placeholder
            let even_h = [[crate::Complex64::ZERO; 4]; 4];
            let odd_h = [[crate::Complex64::ZERO; 4]; 4];
            self.inner
                .run(&even_h, &odd_h)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        }
    }

    /// QASM parsing and export
    #[pyfunction]
    fn parse_qasm_py(qasm: &str) -> PyResult<PyCircuit> {
        crate::qasm::parse_qasm(qasm)
            .map(|c| PyCircuit { inner: c })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[pyfunction]
    fn export_qasm_py(circuit: &PyCircuit) -> PyResult<String> {
        Ok(crate::qasm_export::export_qasm(&circuit.inner))
    }

    #[pyfunction]
    fn parse_circuit_py(expr: &str) -> PyResult<PyCircuit> {
        crate::parse_circuit(expr)
            .map(|c| PyCircuit { inner: c })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Parse a gate string like "H", "X", "Rx(1.5)"
    fn parse_gate(s: &str) -> PyResult<Gate> {
        let s = s.trim();
        if let Some(paren) = s.find('(') {
            let name = &s[..paren];
            let angle_str = &s[paren + 1..s.len() - 1];
            let angle: f64 = angle_str
                .parse()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid angle"))?;
            match name {
                "Rx" => Ok(Gate::Rx(angle)),
                "Ry" => Ok(Gate::Ry(angle)),
                "Rz" => Ok(Gate::Rz(angle)),
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown parameterized gate {}",
                    name
                ))),
            }
        } else {
            match s {
                "I" | "Identity" => Ok(Gate::I),
                "X" => Ok(Gate::X),
                "Y" => Ok(Gate::Y),
                "Z" => Ok(Gate::Z),
                "H" => Ok(Gate::H),
                "S" => Ok(Gate::S),
                "Sdg" => Ok(Gate::Sdg),
                "T" => Ok(Gate::T),
                "Tdg" => Ok(Gate::Tdg),
                _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unknown gate {}",
                    s
                ))),
            }
        }
    }

    /// Parser
    #[pyfunction]
    fn parse_gate_py(s: &str) -> PyResult<String> {
        Ok(s.to_string())
    }

    #[pyfunction]
    fn qft_circuit_py(num_qubits: usize) -> PyCircuit {
        PyCircuit {
            inner: crate::algorithms::qft_circuit(num_qubits),
        }
    }

    #[pyfunction]
    fn hhl_2x2_py(a00: f64, a01: f64, a10: f64, a11: f64, b0: f64, b1: f64) -> PyCircuit {
        PyCircuit {
            inner: crate::algorithms::hhl_2x2([[a00, a01], [a10, a11]], [b0, b1]),
        }
    }

    #[pyfunction]
    fn shor_period_finding_py(n: u64, a: u64) -> PyCircuit {
        PyCircuit {
            inner: crate::algorithms::shor_period_finding(n, a),
        }
    }

    #[pyfunction]
    fn hhl_solve_py(a00: f64, a01: f64, a10: f64, a11: f64, b0: f64, b1: f64) -> (f64, f64) {
        let x = crate::algorithms::hhl_solve([[a00, a01], [a10, a11]], [b0, b1]);
        (x[0], x[1])
    }

    #[pyfunction]
    fn shor_factor_py(n: u64) -> Option<(u64, u64)> {
        crate::algorithms::shor_factor(n)
    }

    #[pyfunction]
fn hhl_solve_general_py(rows: Vec<Vec<f64>>, b: Vec<f64>) -> Vec<f64> {
    crate::algorithms::hhl_solve_general(&rows, &b)
}

#[pyfunction]
fn shor_factor_large_py(n: u64) -> Option<(u64, u64)> {
    crate::algorithms::shor_factor_large(n)
}

/// Python module definition
    #[pymodule]
    pub fn quantumforge(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyCircuit>()?;
        m.add_class::<PyStateVector>()?;
        m.add_class::<PyDensityMatrix>()?;
        m.add_class::<PyShotResult>()?;
        m.add_class::<PyPauliString>()?;
        m.add_class::<PyQecCode>()?;
        m.add_class::<PyGrover>()?;
        m.add_class::<PyQAOABuilder>()?;
        m.add_class::<PyHEABuilder>()?;
        m.add_class::<PyQAOAAnsatzBuilder>()?;
        m.add_class::<PyOptimizeConfig>()?;
        m.add_class::<PyQecCode>()?;
        m.add_class::<PyMPS>()?;
        m.add_class::<PyTensor>()?;
        m.add_class::<PyTEBD>()?;
        m.add_class::<PyDMRG>()?;

        m.add_function(wrap_pyfunction!(qft_circuit_py, m)?)?;
        m.add_function(wrap_pyfunction!(hhl_2x2_py, m)?)?;
        m.add_function(wrap_pyfunction!(shor_period_finding_py, m)?)?;
        m.add_function(wrap_pyfunction!(hhl_solve_py, m)?)?;
        m.add_function(wrap_pyfunction!(shor_factor_py, m)?)?;
        m.add_function(wrap_pyfunction!(hhl_solve_general_py, m)?)?;
        m.add_function(wrap_pyfunction!(shor_factor_large_py, m)?)?;
        m.add_function(wrap_pyfunction!(export_qasm_py, m)?)?;
        m.add_function(wrap_pyfunction!(parse_circuit_py, m)?)?;
        m.add_function(wrap_pyfunction!(optimize_circuit_py, m)?)?;
        m.add_function(wrap_pyfunction!(repetition_code_bitflip, m)?)?;
        m.add_function(wrap_pyfunction!(repetition_code_phaseflip, m)?)?;
        m.add_function(wrap_pyfunction!(shor_code, m)?)?;
        m.add_function(wrap_pyfunction!(steane_code, m)?)?;
        m.add_function(wrap_pyfunction!(surface_code_planar, m)?)?;
        m.add_function(wrap_pyfunction!(surface_code_rotated, m)?)?;
        m.add_function(wrap_pyfunction!(encode_repetition_bitflip, m)?)?;
        m.add_function(wrap_pyfunction!(parse_gate_py, m)?)?;

        Ok(())
    }
}

#[cfg(feature = "pyo3")]
pub use python_bindings::quantumforge;
