//! Python bindings for QuantumForge using PyO3.
//!
//! Enable with `--features python`. Provides a `quantumforge` Python module
//! with classes for Circuit, StateVector, DensityMatrix, and the REPL commands.

#[cfg(feature = "pyo3")]
pub mod python_bindings {
    //! PyO3 classes exposed as the `quantumforge` Python module.
    //!
    //! Each class wraps the corresponding crate type (`Circuit`, `StateVector`,
    //! `DensityMatrix`, `ShotResult`, ...) and mirrors its Rust API, so the
    //! Python surface tracks the documented Rust one.

    use crate::{
        Circuit, DensityMatrix, Gate, ParameterizedCircuit, RotationAxis, ShotResult,
        SingleQubitObservable, StateVector,
        algorithms_templates::{
            EntanglementPattern, Grover, HEABuilder, QAOAAnsatzBuilder, QAOABuilder,
        },
        optimizer::{OptimizeConfig, OptimizeResult},
        qasm::parse_qasm,
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
                        "Unknown channel: {channel}"
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
            format!("PauliString({})", self.inner)
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

    /// Python wrapper for ParameterizedCircuit with training support.
    #[pyclass]
    struct PyParameterizedCircuit {
        inner: ParameterizedCircuit,
    }

    #[pymethods]
    impl PyParameterizedCircuit {
        #[new]
        fn new(parameter_count: usize) -> Self {
            Self {
                inner: ParameterizedCircuit::new(parameter_count),
            }
        }

        #[staticmethod]
        fn with_inputs(parameter_count: usize, input_count: usize) -> Self {
            Self {
                inner: ParameterizedCircuit::with_inputs(parameter_count, input_count),
            }
        }

        fn rotation(&mut self, axis: &str, target: usize, parameter: usize) {
            let ax = match axis {
                "x" | "X" => RotationAxis::X,
                "y" | "Y" => RotationAxis::Y,
                "z" | "Z" => RotationAxis::Z,
                _ => RotationAxis::Y,
            };
            self.inner.rotation(ax, target, parameter);
        }

        fn encode_rotation(&mut self, axis: &str, target: usize, input: usize) {
            let ax = match axis {
                "x" | "X" => RotationAxis::X,
                "y" | "Y" => RotationAxis::Y,
                "z" | "Z" => RotationAxis::Z,
                _ => RotationAxis::Y,
            };
            self.inner.encode_rotation(ax, target, input);
        }

        fn cnot(&mut self, control: usize, target: usize) {
            self.inner.cnot(control, target);
        }

        fn parameter_count(&self) -> usize {
            self.inner.parameter_count()
        }

        fn input_count(&self) -> usize {
            self.inner.input_count()
        }

        fn expectation_z(
            &self,
            params: Vec<f64>,
            qubits: usize,
            observable_qubit: usize,
        ) -> PyResult<f64> {
            self.inner
                .expectation_z(&params, qubits, observable_qubit)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn expectation_z_shots_with_ci(
            &self,
            params: Vec<f64>,
            qubits: usize,
            observable_qubit: usize,
            shots: usize,
            seed: u64,
        ) -> PyResult<(f64, (f64, f64))> {
            let sampling = crate::ShotConfig { shots, seed };
            self.inner
                .expectation_shots_with_ci(
                    &params,
                    qubits,
                    SingleQubitObservable::Z,
                    observable_qubit,
                    sampling,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn mse_loss_and_gradient(
            &self,
            data_x: Vec<f64>,
            data_y: Vec<f64>,
            params: Vec<f64>,
            qubits: usize,
        ) -> PyResult<(f64, Vec<f64>)> {
            let data: Vec<(Vec<f64>, f64)> = data_x
                .into_iter()
                .zip(data_y)
                .map(|(x, y)| (vec![x], y))
                .collect();
            self.inner
                .mse_loss_and_gradient(&data, &params, qubits, SingleQubitObservable::Z, 0)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn parameter_shift_gradient(
            &self,
            inputs: Vec<f64>,
            params: Vec<f64>,
            qubits: usize,
        ) -> PyResult<Vec<f64>> {
            self.inner
                .parameter_shift_gradient_with_inputs(
                    &inputs,
                    &params,
                    qubits,
                    SingleQubitObservable::Z,
                    0,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        /// Facts about this circuit that decide which gradient rule applies.
        ///
        /// `shots` is the shot budget the caller intends for the expectation;
        /// it does not make anything here sample.
        fn profile(&self, shots: usize) -> PyCircuitProfile {
            let mut profile = self.inner.circuit_profile();
            profile.shots = shots;
            PyCircuitProfile { inner: profile }
        }

        /// Gradient of `<observable>` on `observable_qubit` with a chosen rule.
        ///
        /// `strategy` is one of the names from `gradient_strategies()`; `"auto"`
        /// resolves it from [`Self::profile`]. Exact rules return the analytic
        /// gradient, so the result is a reference you can compare a sampled
        /// gradient against.
        #[pyo3(signature = (params, qubits, observable = "Z", observable_qubit = 0,
                            strategy = "auto", finite_difference_step = 1e-3, shots = 0))]
        #[allow(clippy::too_many_arguments)]
        fn gradient(
            &self,
            params: Vec<f64>,
            qubits: usize,
            observable: &str,
            observable_qubit: usize,
            strategy: &str,
            finite_difference_step: f64,
            shots: usize,
        ) -> PyResult<PyGradientResult> {
            let observable = parse_single_qubit_observable(observable)?;
            let strategy =
                crate::gradients::GradientStrategy::parse(strategy).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "unknown gradient strategy '{strategy}'; see gradient_strategies()"
                    ))
                })?;
            let options = crate::gradients::GradientOptions {
                strategy,
                finite_difference_step,
                shots,
            };
            let estimate = self
                .inner
                .gradient_with_inputs(&options, &[], &params, qubits, observable, observable_qubit)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            Ok(PyGradientResult { inner: estimate })
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
    #[pyo3(signature = (circuit, config=None))]
    fn optimize_circuit_py<'a>(
        circuit: &'a PyCircuit,
        config: Option<&'a PyOptimizeConfig>,
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
                .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
        }

        fn find_ground_state(&mut self, max_steps: usize, _energy_tol: f64) -> PyResult<f64> {
            self.inner
                .find_ground_state(max_steps, _energy_tol)
                .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
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
                .map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
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
                    "unknown parameterized gate {name}"
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
                    "unknown gate {s}"
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

    /// STO-3G electron integrals for H and He, returned as nested matrices.
    #[pyclass]
    #[derive(Clone)]
    struct PyElectronIntegrals {
        inner: crate::chemistry::ElectronIntegrals,
    }

    #[pymethods]
    impl PyElectronIntegrals {
        #[getter]
        fn overlap(&self) -> Vec<Vec<f64>> {
            self.inner.overlap.clone()
        }
        #[getter]
        fn kinetic(&self) -> Vec<Vec<f64>> {
            self.inner.kinetic.clone()
        }
        #[getter]
        fn nuclear(&self) -> Vec<Vec<f64>> {
            self.inner.nuclear.clone()
        }
        #[getter]
        fn repulsion(&self) -> Vec<Vec<Vec<Vec<f64>>>> {
            self.inner.repulsion.clone()
        }
        #[getter]
        fn nuclear_repulsion(&self) -> f64 {
            self.inner.nuclear_repulsion
        }
    }

    /// Result of an ADAPT-VQE run.
    #[pyclass]
    struct PyAdaptVqeResult {
        inner: crate::chemistry::adapt_vqe::AdaptVqeResult,
    }

    #[pymethods]
    impl PyAdaptVqeResult {
        #[getter]
        fn circuit(&self) -> PyParameterizedCircuit {
            PyParameterizedCircuit {
                inner: self.inner.circuit.clone(),
            }
        }
        #[getter]
        fn operators(&self) -> Vec<String> {
            self.inner
                .operators
                .iter()
                .map(|word| word.to_string())
                .collect()
        }
        #[getter]
        fn parameters(&self) -> Vec<f64> {
            self.inner.parameters.clone()
        }
        #[getter]
        fn energy(&self) -> f64 {
            self.inner.energy
        }
        #[getter]
        fn gradients(&self) -> Vec<f64> {
            self.inner.gradients.clone()
        }
    }

    #[pyfunction]
    fn qpe_circuit_py(unitary: &PyCircuit, precision_qubits: usize) -> PyCircuit {
        PyCircuit {
            inner: crate::algorithms::qpe_circuit(unitary.inner.clone(), precision_qubits),
        }
    }

    fn chemistry_error(error: crate::chemistry::ChemistryError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string())
    }

    #[pyfunction]
    fn sto3g_integrals_py(
        elements: Vec<String>,
        positions: Vec<Vec<f64>>,
    ) -> PyResult<PyElectronIntegrals> {
        if elements.len() != positions.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "elements and positions must have the same length",
            ));
        }
        let atoms: Vec<crate::chemistry::Atom> = elements
            .iter()
            .zip(positions.iter())
            .map(|(element, position)| {
                let element = match element.as_str() {
                    "H" | "h" => Ok(crate::chemistry::Element::H),
                    "He" | "he" | "HE" => Ok(crate::chemistry::Element::He),
                    other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "unsupported element '{other}'; STO-3G here supports H and He"
                    ))),
                }?;
                let coords: [f64; 3] = position.as_slice().try_into().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "each position must have exactly three coordinates",
                    )
                })?;
                Ok(crate::chemistry::Atom {
                    element,
                    position: coords,
                })
            })
            .collect::<PyResult<_>>()?;
        crate::chemistry::sto3g_integrals(&atoms)
            .map(|inner| PyElectronIntegrals { inner })
            .map_err(chemistry_error)
    }

    #[pyfunction]
    fn molecular_hamiltonian_py(
        integrals: &PyElectronIntegrals,
        mapping: &str,
    ) -> PyResult<PyPauliSum> {
        let mapping = match mapping {
            "jordan-wigner" | "jw" | "JordanWigner" => {
                crate::chemistry::mapping::FermionMapping::JordanWigner
            }
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "unsupported mapping '{other}'; use 'jordan-wigner'"
                )));
            }
        };
        crate::chemistry::mapping::molecular_hamiltonian(&integrals.inner, mapping)
            .map(|inner| PyPauliSum { inner })
            .map_err(chemistry_error)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn run_adapt_vqe_py(
        hamiltonian: &PyPauliSum,
        reference: usize,
        qubits: usize,
        gradient_threshold: f64,
        max_operators: usize,
        learning_rate: f64,
        optimisation_steps: usize,
    ) -> PyResult<PyAdaptVqeResult> {
        let pool = crate::chemistry::adapt_vqe::qubit_excitation_pool(qubits);
        let options = crate::chemistry::adapt_vqe::AdaptVqeOptions {
            gradient_threshold,
            max_operators,
            learning_rate,
            optimisation_steps,
        };
        crate::chemistry::adapt_vqe::run_adapt_vqe(
            &hamiltonian.inner,
            reference,
            qubits,
            &pool,
            &options,
        )
        .map(|inner| PyAdaptVqeResult { inner })
        .map_err(chemistry_error)
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
    fn hhl_solve_quantum_general_py(rows: Vec<Vec<f64>>, b: Vec<f64>) -> Vec<f64> {
        crate::algorithms::hhl_solve_quantum_general(&rows, &b)
    }

    #[pyfunction]
    fn shor_factor_large_py(n: u64) -> Option<(u64, u64)> {
        crate::algorithms::shor_factor(n)
    }

    #[pyfunction]
    fn shor_factor_quantum_py(n: u64) -> Option<(u64, u64)> {
        crate::algorithms::shor_factor_quantum(n)
    }

    // ------------------------------------------------------------------
    // Gradient strategies
    // ------------------------------------------------------------------

    /// A gradient estimate: the gradient, the rule that ran, and how many
    /// state-vector sweeps it consumed.
    #[pyclass]
    struct PyGradientResult {
        inner: crate::gradients::GradientEstimate,
    }

    #[pymethods]
    impl PyGradientResult {
        #[getter]
        fn gradient(&self) -> Vec<f64> {
            self.inner.gradient.clone()
        }

        #[getter]
        fn strategy(&self) -> String {
            self.inner.strategy.name().to_string()
        }

        #[getter]
        fn evaluations(&self) -> usize {
            self.inner.evaluations
        }

        fn __repr__(&self) -> String {
            format!(
                "GradientResult(strategy='{}', evaluations={}, gradient={:?})",
                self.inner.strategy.name(),
                self.inner.evaluations,
                self.inner.gradient
            )
        }
    }

    /// Static facts about a parameterized circuit, used to choose a rule.
    #[pyclass]
    struct PyCircuitProfile {
        inner: crate::gradients::CircuitProfile,
    }

    #[pymethods]
    impl PyCircuitProfile {
        #[getter]
        fn operations(&self) -> usize {
            self.inner.operations
        }

        #[getter]
        fn parameters(&self) -> usize {
            self.inner.parameters
        }

        #[getter]
        fn parameterized_operations(&self) -> usize {
            self.inner.parameterized_operations
        }

        #[getter]
        fn has_channel(&self) -> bool {
            self.inner.has_channel
        }

        #[getter]
        fn shots(&self) -> usize {
            self.inner.shots
        }

        fn __repr__(&self) -> String {
            format!(
                "CircuitProfile(operations={}, parameters={}, parameterized_operations={}, has_channel={}, shots={})",
                self.inner.operations,
                self.inner.parameters,
                self.inner.parameterized_operations,
                self.inner.has_channel,
                self.inner.shots
            )
        }
    }

    fn parse_single_qubit_observable(name: &str) -> PyResult<SingleQubitObservable> {
        match name {
            "X" => Ok(SingleQubitObservable::X),
            "Y" => Ok(SingleQubitObservable::Y),
            "Z" => Ok(SingleQubitObservable::Z),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown observable '{other}'; expected X, Y or Z"
            ))),
        }
    }

    /// The gradient rule names accepted by `ParameterizedCircuit.gradient`,
    /// including `"auto"`.
    #[pyfunction]
    fn gradient_strategies() -> Vec<&'static str> {
        [
            "auto",
            "parameter-shift",
            "adjoint",
            "backprop",
            "finite-difference",
        ]
        .to_vec()
    }

    // ------------------------------------------------------------------
    // Exact-rational analytic core
    // ------------------------------------------------------------------

    /// An exact rational number `numerator / denominator`, reduced to lowest
    /// terms. This is the arithmetic the analytic core uses: no floating point,
    /// so `1/3 + 1/6` is exactly `1/2`.
    #[pyclass]
    #[derive(Clone)]
    struct PyRational {
        inner: crate::analytic::Rational,
    }

    fn rational_binary(
        op: impl Fn(
            crate::analytic::Rational,
            crate::analytic::Rational,
        ) -> Result<crate::analytic::Rational, crate::analytic::AnalyticError>,
        left: &PyRational,
        right: &PyRational,
    ) -> PyResult<PyRational> {
        op(left.inner, right.inner)
            .map(|inner| PyRational { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[pymethods]
    impl PyRational {
        #[new]
        fn new(numerator: i128, denominator: i128) -> PyResult<Self> {
            Ok(Self {
                inner: crate::analytic::Rational::new(numerator, denominator)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            })
        }

        #[getter]
        fn numerator(&self) -> i128 {
            self.inner.numerator()
        }

        #[getter]
        fn denominator(&self) -> i128 {
            self.inner.denominator()
        }

        /// The value as a float. Lossy for anything that is not dyadic.
        fn to_float(&self) -> f64 {
            self.inner.as_f64()
        }

        fn is_zero(&self) -> bool {
            self.inner.is_zero()
        }

        fn __str__(&self) -> String {
            self.inner.to_string()
        }

        fn __repr__(&self) -> String {
            format!(
                "Rational({}, {})",
                self.inner.numerator(),
                self.inner.denominator()
            )
        }

        fn __add__(&self, other: &PyRational) -> PyResult<PyRational> {
            rational_binary(|a, b| a.checked_add(b), self, other)
        }

        fn __sub__(&self, other: &PyRational) -> PyResult<PyRational> {
            rational_binary(|a, b| a.checked_sub(b), self, other)
        }

        fn __mul__(&self, other: &PyRational) -> PyResult<PyRational> {
            rational_binary(|a, b| a.checked_mul(b), self, other)
        }

        fn __truediv__(&self, other: &PyRational) -> PyResult<PyRational> {
            rational_binary(|a, b| a.checked_div(b), self, other)
        }

        fn __neg__(&self) -> PyResult<PyRational> {
            self.inner
                .checked_neg()
                .map(|inner| PyRational { inner })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn __eq__(&self, other: &PyRational) -> bool {
            self.inner == other.inner
        }
    }

    /// An exact complex number with rational real and imaginary parts.
    #[pyclass]
    #[derive(Clone)]
    struct PyExactComplex {
        inner: crate::analytic::ExactComplex,
    }

    fn exact_complex_binary(
        op: impl Fn(
            crate::analytic::ExactComplex,
            crate::analytic::ExactComplex,
        ) -> Result<crate::analytic::ExactComplex, crate::analytic::AnalyticError>,
        left: &PyExactComplex,
        right: &PyExactComplex,
    ) -> PyResult<PyExactComplex> {
        op(left.inner, right.inner)
            .map(|inner| PyExactComplex { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[pymethods]
    impl PyExactComplex {
        #[new]
        fn new(real: &PyRational, imag: &PyRational) -> Self {
            Self {
                inner: crate::analytic::ExactComplex {
                    re: real.inner,
                    im: imag.inner,
                },
            }
        }

        /// The real part as an exact rational.
        #[getter]
        fn real(&self) -> PyRational {
            PyRational {
                inner: self.inner.re,
            }
        }

        /// The imaginary part as an exact rational.
        #[getter]
        fn imag(&self) -> PyRational {
            PyRational {
                inner: self.inner.im,
            }
        }

        /// The value as a Python complex. Lossy; exactness lives in this type.
        fn to_complex(&self) -> (f64, f64) {
            let value = self.inner.to_complex64();
            (value.re, value.im)
        }

        fn is_zero(&self) -> bool {
            self.inner.is_zero()
        }

        fn __str__(&self) -> String {
            self.inner.to_string()
        }

        fn __repr__(&self) -> String {
            format!("ExactComplex({}, {})", self.inner.re, self.inner.im)
        }

        fn __add__(&self, other: &PyExactComplex) -> PyResult<PyExactComplex> {
            exact_complex_binary(|a, b| a.checked_add(b), self, other)
        }

        fn __sub__(&self, other: &PyExactComplex) -> PyResult<PyExactComplex> {
            exact_complex_binary(|a, b| a.checked_sub(b), self, other)
        }

        fn __mul__(&self, other: &PyExactComplex) -> PyResult<PyExactComplex> {
            exact_complex_binary(|a, b| a.checked_mul(b), self, other)
        }

        fn __truediv__(&self, other: &PyExactComplex) -> PyResult<PyExactComplex> {
            exact_complex_binary(|a, b| a.checked_div(b), self, other)
        }

        fn __neg__(&self) -> PyResult<PyExactComplex> {
            self.inner
                .checked_neg()
                .map(|inner| PyExactComplex { inner })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn __eq__(&self, other: &PyExactComplex) -> bool {
            self.inner == other.inner
        }
    }

    /// An exact linear combination of Pauli words.
    ///
    /// Coefficients are exact rationals, so a Hamiltonian built here has an
    /// exactly representable energy.
    #[pyclass]
    struct PyPauliSum {
        inner: crate::analytic::PauliSum,
    }

    fn parse_pauli_pattern(pattern: &str, qubits: usize) -> PyResult<crate::analytic::PauliWord> {
        let letters: Vec<char> = pattern.chars().collect();
        if letters.len() != qubits {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "pattern '{pattern}' has {} letters but the sum acts on {qubits} qubits",
                letters.len()
            )));
        }
        let mut word = Vec::with_capacity(qubits);
        for letter in letters {
            word.push(match letter {
                'I' => crate::analytic::Pauli::I,
                'X' => crate::analytic::Pauli::X,
                'Y' => crate::analytic::Pauli::Y,
                'Z' => crate::analytic::Pauli::Z,
                other => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "invalid Pauli letter '{other}' in '{pattern}'; expected I, X, Y or Z"
                    )));
                }
            });
        }
        Ok(crate::analytic::PauliWord::new(word))
    }

    #[pymethods]
    impl PyPauliSum {
        #[new]
        fn new(qubits: usize) -> Self {
            Self {
                inner: crate::analytic::PauliSum::zero(qubits),
            }
        }

        /// Adds `numerator/denominator * |pattern>` to the sum.
        fn add_term(&mut self, pattern: &str, numerator: i128, denominator: i128) -> PyResult<()> {
            let word = parse_pauli_pattern(pattern, self.inner.qubits())?;
            let coefficient = crate::analytic::ExactComplex::real(
                crate::analytic::Rational::new(numerator, denominator)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            );
            self.inner
                .add_term(coefficient, word)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn qubits(&self) -> usize {
            self.inner.qubits()
        }

        fn term_count(&self) -> usize {
            self.inner.terms().len()
        }

        fn __str__(&self) -> String {
            self.inner.to_string()
        }

        fn __repr__(&self) -> String {
            format!("PauliSum({})", self.inner)
        }
    }

    // ------------------------------------------------------------------
    // SubsCoRe: adaptive observation cost control
    // ------------------------------------------------------------------

    /// The outcome of a SubsCoRe run: the best parameters, the exact energy at
    /// them, and the shot accounting.
    #[pyclass]
    struct PySubsCoReOutcome {
        inner: crate::subscore::SubsCoReOutcome,
    }

    #[pymethods]
    impl PySubsCoReOutcome {
        #[getter]
        fn best(&self) -> Vec<f64> {
            self.inner.best.clone()
        }

        #[getter]
        fn best_value(&self) -> f64 {
            self.inner.best_value
        }

        #[getter]
        fn shots_used(&self) -> usize {
            self.inner.shots_used
        }

        #[getter]
        fn iterations(&self) -> usize {
            self.inner.iterations
        }

        #[getter]
        fn shots_per_iteration(&self) -> Vec<usize> {
            self.inner.shots_per_iteration.clone()
        }

        #[getter]
        fn core_variances(&self) -> Vec<f64> {
            self.inner.core_variances.clone()
        }

        fn __repr__(&self) -> String {
            format!(
                "SubsCoReOutcome(best_value={}, shots_used={}, iterations={})",
                self.inner.best_value, self.inner.shots_used, self.inner.iterations
            )
        }
    }

    fn parse_update_rule(name: &str) -> PyResult<crate::subscore::UpdateRule> {
        use crate::subscore::UpdateRule;
        match name {
            "mean" => Ok(UpdateRule::Mean),
            "thompson" => Ok(UpdateRule::Thompson),
            "guarded-thompson" | "guarded" => Ok(UpdateRule::GuardedThompson),
            "free-energy" | "free_energy" => Ok(UpdateRule::FreeEnergy),
            "lower-confidence-bound" | "lcb" => Ok(UpdateRule::LowerConfidenceBound),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown update rule '{other}'; expected mean, thompson, \
                 guarded-thompson, free-energy or lower-confidence-bound"
            ))),
        }
    }

    fn parse_readout_strategy(name: &str) -> PyResult<crate::subscore::ReadoutStrategy> {
        use crate::subscore::ReadoutStrategy;
        match name {
            "bound" => Ok(ReadoutStrategy::Bound),
            "center" => Ok(ReadoutStrategy::Center),
            other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown readout strategy '{other}'; expected bound or center"
            ))),
        }
    }

    /// SubsCoRe (arXiv:2502.01704): minimize `<H>` on `circuit` while spending
    /// `total_shot_budget` shots, cooling the confident-region threshold as the
    /// run progresses.
    ///
    /// `hamiltonian` is a `PauliSum` built with exact rational coefficients.
    /// `eta_sq` defaults to the objective's own single-shot variance; the
    /// thresholds default to the paper's `eta^2/512` and `eta^2/4096`.
    #[pyfunction]
    #[pyo3(signature = (circuit, hamiltonian, params, qubits, total_shot_budget,
                        eta_sq = None, initial_shots = 128,
                        initial_core_variance = None, core_min = None,
                        update_rule = "mean", strategy = "bound", seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn vqe_subscore_py(
        circuit: &PyParameterizedCircuit,
        hamiltonian: &PyPauliSum,
        params: Vec<f64>,
        qubits: usize,
        total_shot_budget: usize,
        eta_sq: Option<f64>,
        initial_shots: usize,
        initial_core_variance: Option<f64>,
        core_min: Option<f64>,
        update_rule: &str,
        strategy: &str,
        seed: u64,
    ) -> PyResult<PySubsCoReOutcome> {
        let objective = crate::subscore::ShotHamiltonian::new(
            circuit.inner.clone(),
            hamiltonian.inner.clone(),
            qubits,
        );
        let eta_sq = match eta_sq {
            Some(value) => value,
            None => objective.single_shot_variance(),
        };
        let initial_core_variance = match initial_core_variance {
            Some(value) => value,
            None => eta_sq / 512.0,
        };
        let core_min = match core_min {
            Some(value) => value,
            None => eta_sq / 4096.0,
        };
        let options = crate::subscore::SubsCoReOptions {
            total_shot_budget,
            eta_sq,
            initial_shots,
            initial_core_variance,
            core_min,
            update_rule: parse_update_rule(update_rule)?,
            strategy: parse_readout_strategy(strategy)?,
            seed,
            ..crate::subscore::SubsCoReOptions::default()
        };
        let counts = objective.parameter_counts();
        let outcome = crate::subscore::run_subscore(
            |point, shots, run_seed| objective.observe(point, shots, run_seed),
            &params,
            &counts,
            &options,
        );
        Ok(PySubsCoReOutcome { inner: outcome })
    }

    fn thermal_error(error: crate::QuantumError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string())
    }

    /// Gibbs state diagnostics: `(probabilities, <E>, F)` at `beta`.
    #[pyfunction]
    fn thermal_gibbs_py(hamiltonian: &PyPauliSum, beta: f64) -> PyResult<(Vec<f64>, f64, f64)> {
        let inner = &hamiltonian.inner;
        let rho = crate::thermal::gibbs_state(inner, beta).map_err(thermal_error)?;
        let probabilities =
            crate::thermal::gibbs_probabilities(inner, beta).map_err(thermal_error)?;
        let energy = crate::thermal::expectation(&rho, inner).map_err(thermal_error)?;
        let free = crate::thermal::free_energy(inner, beta).map_err(thermal_error)?;
        Ok((probabilities, energy, free))
    }

    /// Energy standard deviation of the shot estimator at `shots` shots.
    #[pyfunction]
    fn shot_noise_temperature_py(hamiltonian: &PyPauliSum, shots: usize) -> PyResult<f64> {
        crate::noisy_optimizer::shot_noise_temperature(&hamiltonian.inner, shots)
            .map_err(thermal_error)
    }

    /// Variational quantum thermalization: `(F, F_exact, trace_distance)`.
    #[pyfunction]
    fn vqt_py(
        hamiltonian: &PyPauliSum,
        temperature: f64,
        layers: usize,
        steps: usize,
    ) -> PyResult<(f64, f64, f64)> {
        let options = crate::vqt::VqtOptions {
            system_qubits: hamiltonian.inner.qubits(),
            layers,
            temperature,
            learning_rate: 0.05,
            steps,
            finite_difference_step: 1e-3,
        };
        let outcome = crate::vqt::variational_thermalize(&hamiltonian.inner, &options)
            .map_err(thermal_error)?;
        Ok((
            outcome.free_energy,
            outcome.exact_free_energy,
            outcome.trace_distance,
        ))
    }

    /// Dissipative preparation of a real target amplitude vector: the fidelity.
    #[pyfunction]
    fn dissipative_prepare_py(target: Vec<f64>, dt: f64, steps: usize) -> PyResult<f64> {
        if target.is_empty() || !target.len().is_power_of_two() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "target length must be a power of two",
            ));
        }
        let amplitudes: Vec<crate::Complex64> = target
            .iter()
            .map(|value| crate::Complex64::new(*value, 0.0))
            .collect();
        let qubits = amplitudes.len().ilog2() as usize;
        let mut rho = DensityMatrix::from_state(&StateVector::zero(qubits).map_err(thermal_error)?);
        crate::dissipative::prepare_target_state(&mut rho, &amplitudes, dt, steps)
            .map_err(thermal_error)?;
        Ok(crate::dissipative::state_fidelity(&rho, &amplitudes))
    }

    // ------------------------------------------------------------------
    // Stabilizer formalism
    // ------------------------------------------------------------------

    /// A stabilizer tableau: Clifford-group state tracking (Gottesman-Knill).
    ///
    /// Applies Clifford gates symbolically, so it avoids the exponential cost of
    /// state-vector simulation. See `quantumforge::stabilizer`.
    #[pyclass]
    struct PyStabilizerTableau {
        inner: crate::stabilizer::StabilizerTableau,
    }

    #[pymethods]
    impl PyStabilizerTableau {
        #[new]
        fn new(n: usize) -> Self {
            Self {
                inner: crate::stabilizer::StabilizerTableau::new(n),
            }
        }

        fn h(&mut self, q: usize) {
            self.inner.h(q);
        }

        fn s(&mut self, q: usize) {
            self.inner.s(q);
        }

        fn cnot(&mut self, c: usize, t: usize) {
            self.inner.cnot(c, t);
        }

        fn x(&mut self, q: usize) {
            self.inner.x(q);
        }

        fn z(&mut self, q: usize) {
            self.inner.z(q);
        }

        /// Measures qubit `q`; returns `(outcome, probability)`.
        fn measure(&mut self, q: usize) -> (bool, f64) {
            self.inner.measure(q)
        }

        /// Measures qubit `q`, drawing the random outcome from `seed`.
        ///
        /// Reproducible for a given `(state, q, seed)`; use this rather than
        /// `measure` when the sample distribution matters.
        fn measure_seeded(&mut self, q: usize, seed: u64) -> (bool, f64) {
            self.inner.measure_seeded(q, seed)
        }

        /// Whether two tableaux describe the same stabilizer state.
        fn equivalent(&self, other: &PyStabilizerTableau) -> bool {
            self.inner.equivalent(&other.inner)
        }

        fn num_qubits(&self) -> usize {
            self.inner.num_qubits()
        }

        fn __repr__(&self) -> String {
            format!("StabilizerTableau({} qubits)", self.inner.num_qubits())
        }
    }

    // ------------------------------------------------------------------
    // Cost-aware circuit optimization
    // ------------------------------------------------------------------

    /// Logical resource counts for a circuit. T and Tdg are counted separately
    /// so a cost model can price them differently from other single-qubit gates.
    #[pyclass]
    struct PyCircuitCostProfile {
        inner: crate::optimizer::CircuitCostProfile,
    }

    #[pymethods]
    impl PyCircuitCostProfile {
        #[getter]
        fn single_qubit_gates(&self) -> usize {
            self.inner.single_qubit_gates
        }

        #[getter]
        fn cnot_gates(&self) -> usize {
            self.inner.cnot_gates
        }

        #[getter]
        fn cz_gates(&self) -> usize {
            self.inner.cz_gates
        }

        #[getter]
        fn swap_gates(&self) -> usize {
            self.inner.swap_gates
        }

        #[getter]
        fn t_gates(&self) -> usize {
            self.inner.t_gates
        }

        #[getter]
        fn tdg_gates(&self) -> usize {
            self.inner.tdg_gates
        }

        #[getter]
        fn measurements(&self) -> usize {
            self.inner.measurements
        }

        #[getter]
        fn conditional_gates(&self) -> usize {
            self.inner.conditional_gates
        }

        #[getter]
        fn channels(&self) -> usize {
            self.inner.channels
        }

        #[getter]
        fn depth(&self) -> usize {
            self.inner.depth
        }

        #[getter]
        fn qubits(&self) -> usize {
            self.inner.qubits
        }

        fn total_gates(&self) -> usize {
            self.inner.total_gates()
        }

        fn __repr__(&self) -> String {
            format!(
                "CircuitCostProfile(gates={}, depth={}, qubits={})",
                self.inner.total_gates(),
                self.inner.depth,
                self.inner.qubits
            )
        }
    }

    /// A weighted cost model over a `CircuitCostProfile`. Weights are not
    /// normalised; only comparisons made with the same model are meaningful.
    #[pyclass]
    struct PyCircuitCostModel {
        inner: crate::optimizer::CircuitCostModel,
    }

    #[pymethods]
    impl PyCircuitCostModel {
        #[new]
        #[pyo3(signature = (single_qubit_gate = 1.0, cnot_gate = 1.0, cz_gate = 1.0,
                            swap_gate = 1.0, t_gate = 1.0, tdg_gate = 1.0,
                            measurement = 1.0, conditional_gate = 1.0, channel = 1.0,
                            depth = 0.0, qubit = 0.0))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            single_qubit_gate: f64,
            cnot_gate: f64,
            cz_gate: f64,
            swap_gate: f64,
            t_gate: f64,
            tdg_gate: f64,
            measurement: f64,
            conditional_gate: f64,
            channel: f64,
            depth: f64,
            qubit: f64,
        ) -> Self {
            Self {
                inner: crate::optimizer::CircuitCostModel {
                    single_qubit_gate,
                    cnot_gate,
                    cz_gate,
                    swap_gate,
                    t_gate,
                    tdg_gate,
                    measurement,
                    conditional_gate,
                    channel,
                    depth,
                    qubit,
                },
            }
        }

        /// Relative cost of a profile under this model.
        fn estimate(&self, profile: &PyCircuitCostProfile) -> PyResult<f64> {
            self.inner
                .estimate(&profile.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn __repr__(&self) -> String {
            format!(
                "CircuitCostModel(cnot={}, swap={}, t={})",
                self.inner.cnot_gate, self.inner.swap_gate, self.inner.t_gate
            )
        }
    }

    /// Result of a cost-aware optimization: profiles before and after, modeled
    /// costs, and whether the rewrite was kept.
    #[pyclass]
    struct PyCostAwareOptimizeResult {
        inner: crate::optimizer::CostAwareOptimizeResult,
    }

    #[pymethods]
    impl PyCostAwareOptimizeResult {
        #[getter]
        fn original_profile(&self) -> PyCircuitCostProfile {
            PyCircuitCostProfile {
                inner: self.inner.original_profile.clone(),
            }
        }

        #[getter]
        fn candidate_profile(&self) -> PyCircuitCostProfile {
            PyCircuitCostProfile {
                inner: self.inner.candidate_profile.clone(),
            }
        }

        #[getter]
        fn original_cost(&self) -> f64 {
            self.inner.original_cost
        }

        #[getter]
        fn candidate_cost(&self) -> f64 {
            self.inner.candidate_cost
        }

        #[getter]
        fn applied(&self) -> bool {
            self.inner.applied
        }

        fn __repr__(&self) -> String {
            format!(
                "CostAwareOptimizeResult(original_cost={}, candidate_cost={}, applied={})",
                self.inner.original_cost, self.inner.candidate_cost, self.inner.applied
            )
        }
    }

    /// Resource counts for a circuit.
    #[pyfunction]
    fn cost_profile_py(circuit: &PyCircuit) -> PyCircuitCostProfile {
        PyCircuitCostProfile {
            inner: crate::optimizer::CircuitCostProfile::from_circuit(&circuit.inner),
        }
    }

    /// Logical depth from qubit and classical-register dependencies.
    #[pyfunction]
    fn circuit_depth_py(circuit: &PyCircuit) -> usize {
        crate::optimizer::circuit_depth(&circuit.inner)
    }

    /// Two-qubit gate count.
    #[pyfunction]
    fn count_two_qubit_py(circuit: &PyCircuit) -> usize {
        crate::optimizer::count_two_qubit(&circuit.inner)
    }

    /// Optimize a copy and keep the rewrite only when it strictly lowers the
    /// modeled cost. Returns `(circuit, result)`; the input is not mutated.
    #[pyfunction]
    fn optimize_for_cost_model_py(
        circuit: &PyCircuit,
        config: &PyOptimizeConfig,
        model: &PyCircuitCostModel,
    ) -> PyResult<(PyCircuit, PyCostAwareOptimizeResult)> {
        let (selected, result) =
            crate::optimizer::optimize_for_cost_model(&circuit.inner, &config.inner, &model.inner)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok((
            PyCircuit { inner: selected },
            PyCostAwareOptimizeResult { inner: result },
        ))
    }

    // ------------------------------------------------------------------
    // QAOA parameter transfer
    // ------------------------------------------------------------------

    /// A MaxCut instance: `num_qubits` vertices and weighted `edges`.
    #[pyclass]
    struct PyMaxCutProblem {
        inner: crate::mps_qaoa::MaxCutProblem,
    }

    #[pymethods]
    impl PyMaxCutProblem {
        #[new]
        fn new(num_qubits: usize, edges: Vec<(usize, usize, f64)>) -> Self {
            Self {
                inner: crate::mps_qaoa::MaxCutProblem::new(num_qubits, edges),
            }
        }

        /// Random 3-regular graph (configuration model); deterministic per seed.
        #[staticmethod]
        fn random_3_regular(num_qubits: usize, seed: u64) -> PyResult<Self> {
            if num_qubits < 4 || !num_qubits.is_multiple_of(2) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "3-regular graphs need an even number of vertices >= 4",
                ));
            }
            Ok(Self {
                inner: crate::mps_qaoa::MaxCutProblem::random_3_regular(num_qubits, seed),
            })
        }

        /// Erdős–Rényi graph with edge probability `probability`.
        #[staticmethod]
        fn random_er(num_qubits: usize, probability: f64, seed: u64) -> Self {
            Self {
                inner: crate::mps_qaoa::MaxCutProblem::random_er(num_qubits, probability, seed),
            }
        }

        /// Barabási–Albert preferential-attachment graph.
        #[staticmethod]
        fn random_ba(num_qubits: usize, edges_per_node: usize, seed: u64) -> PyResult<Self> {
            if edges_per_node == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "edges_per_node must be positive",
                ));
            }
            Ok(Self {
                inner: crate::mps_qaoa::MaxCutProblem::random_ba(num_qubits, edges_per_node, seed),
            })
        }

        /// Copy of this graph with fresh uniform weights in `[min_weight, max_weight)`.
        fn with_random_weights(&self, seed: u64, min_weight: f64, max_weight: f64) -> Self {
            Self {
                inner: self.inner.with_random_weights(seed, min_weight, max_weight),
            }
        }

        #[getter]
        fn num_qubits(&self) -> usize {
            self.inner.num_qubits
        }

        #[getter]
        fn edges(&self) -> Vec<(usize, usize, f64)> {
            self.inner.edges.clone()
        }

        fn __repr__(&self) -> String {
            format!(
                "MaxCutProblem(vertices={}, edges={})",
                self.inner.num_qubits,
                self.inner.edges.len()
            )
        }
    }

    /// Settings for `run_parameter_transfer`. The defaults match the Rust
    /// module: `depth = 15`, `tqa_dt = 0.75`, and `donor_learning_rate = 0.05`
    /// (the paper's 0.1 stalls under finite-difference gradients).
    #[pyclass]
    struct PyTransferOptions {
        inner: crate::qaoa_transfer::TransferOptions,
    }

    #[pymethods]
    impl PyTransferOptions {
        #[new]
        #[pyo3(signature = (depth = 15, tqa_dt = 0.75, donor_learning_rate = 0.05,
                            donor_epsilon = 1e-8, donor_iterations = 100,
                            compare_full_optimization = false))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            depth: usize,
            tqa_dt: f64,
            donor_learning_rate: f64,
            donor_epsilon: f64,
            donor_iterations: usize,
            compare_full_optimization: bool,
        ) -> Self {
            Self {
                inner: crate::qaoa_transfer::TransferOptions {
                    depth,
                    tqa_dt,
                    donor_learning_rate,
                    donor_epsilon,
                    donor_iterations,
                    compare_full_optimization,
                    ..crate::qaoa_transfer::TransferOptions::default()
                },
            }
        }

        #[getter]
        fn depth(&self) -> usize {
            self.inner.depth
        }

        #[getter]
        fn donor_learning_rate(&self) -> f64 {
            self.inner.donor_learning_rate
        }

        #[getter]
        fn compare_full_optimization(&self) -> bool {
            self.inner.compare_full_optimization
        }

        fn __repr__(&self) -> String {
            format!(
                "TransferOptions(depth={}, lr={}, compare_full={})",
                self.inner.depth,
                self.inner.donor_learning_rate,
                self.inner.compare_full_optimization
            )
        }
    }

    /// Outcome of the transfer pipeline: donor and acceptor angles, the targeted
    /// layer, and the paper's ratios `r_n` and `r_s`.
    #[pyclass]
    struct PyTransferOutcome {
        inner: crate::qaoa_transfer::TransferOutcome,
    }

    #[pymethods]
    impl PyTransferOutcome {
        #[getter]
        fn donor_gammas(&self) -> Vec<f64> {
            self.inner.donor_parameters.gammas.clone()
        }

        #[getter]
        fn donor_betas(&self) -> Vec<f64> {
            self.inner.donor_parameters.betas.clone()
        }

        #[getter]
        fn optimized_gammas(&self) -> Vec<f64> {
            self.inner.optimized.gammas.clone()
        }

        #[getter]
        fn optimized_betas(&self) -> Vec<f64> {
            self.inner.optimized.betas.clone()
        }

        #[getter]
        fn target_layer(&self) -> usize {
            self.inner.target_layer
        }

        #[getter]
        fn refinement_energies(&self) -> Vec<f64> {
            self.inner.refinement_energies.clone()
        }

        /// Acceptor cut expectation after the targeted step.
        #[getter]
        fn energy(&self) -> f64 {
            self.inner.energy
        }

        /// `energy / reference` — the paper's `r_s`.
        #[getter]
        fn approximation_ratio(&self) -> f64 {
            self.inner.approximation_ratio
        }

        /// Cut expectation of the transferred angles before acceptor optimization.
        #[getter]
        fn transferred_ratio(&self) -> f64 {
            self.inner.transferred_ratio
        }

        /// Ratio of a full `2p`-angle acceptor optimization, if requested.
        #[getter]
        fn full_optimization_ratio(&self) -> Option<f64> {
            self.inner.full_optimization_ratio
        }

        fn __repr__(&self) -> String {
            format!(
                "TransferOutcome(r_s={}, r_n={}, energy={})",
                self.inner.approximation_ratio, self.inner.transferred_ratio, self.inner.energy
            )
        }
    }

    /// Run parameter transfer plus targeted-layer optimization from a donor
    /// graph to a larger acceptor of the same family. Raises `ValueError` when
    /// the donor is larger than the acceptor or the acceptor has no edges.
    #[pyfunction]
    fn run_parameter_transfer_py(
        donor: &PyMaxCutProblem,
        acceptor: &PyMaxCutProblem,
        options: &PyTransferOptions,
    ) -> PyResult<PyTransferOutcome> {
        let outcome = crate::qaoa_transfer::run_parameter_transfer(
            &donor.inner,
            &acceptor.inner,
            &options.inner,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyTransferOutcome { inner: outcome })
    }

    // ------------------------------------------------------------------
    // Adiabatic quantum computing
    // ------------------------------------------------------------------

    /// The annealing path `s(t)`.
    #[pyclass]
    #[derive(Clone)]
    struct PyAnnealPath {
        inner: crate::aqc::AnnealPath,
    }

    #[pymethods]
    impl PyAnnealPath {
        /// `s = tau`, the standard linear interpolation.
        #[staticmethod]
        fn linear() -> Self {
            Self {
                inner: crate::aqc::AnnealPath::Linear,
            }
        }

        /// Roland–Cerf "fast start": `s = tau / (tau + a (1 - tau))`.
        #[staticmethod]
        fn fast_start(a: f64) -> Self {
            Self {
                inner: crate::aqc::AnnealPath::FastStart(a),
            }
        }

        /// Polynomial path `s = tau^p`.
        #[staticmethod]
        fn polynomial(p: f64) -> Self {
            Self {
                inner: crate::aqc::AnnealPath::Polynomial(p),
            }
        }

        /// Evaluate the path at normalized time `tau` in `[0, 1]`.
        fn at(&self, tau: f64) -> f64 {
            self.inner.at(tau)
        }

        fn label(&self) -> String {
            self.inner.label()
        }

        fn __repr__(&self) -> String {
            format!("AnnealPath({})", self.inner.label())
        }
    }

    /// An adiabatic interpolation between a dense driver `H0` and problem `H1`.
    #[pyclass]
    struct PyAdiabaticSchedule {
        inner: crate::aqc::AdiabaticSchedule,
    }

    #[pymethods]
    impl PyAdiabaticSchedule {
        /// Build from row-major dense matrices of dimension `2^qubits`.
        #[staticmethod]
        fn from_dense(qubits: usize, h0: Vec<f64>, h1: Vec<f64>) -> PyResult<Self> {
            Ok(Self {
                inner: crate::aqc::AdiabaticSchedule::from_dense(qubits, h0, h1)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            })
        }

        /// Return a copy using a different annealing path.
        fn with_path(&self, path: &PyAnnealPath) -> Self {
            Self {
                inner: self.inner.clone().with_path(path.inner),
            }
        }

        fn path(&self) -> PyAnnealPath {
            PyAnnealPath {
                inner: self.inner.path(),
            }
        }

        /// The annealing parameter at normalized time `tau`.
        fn s_at(&self, tau: f64) -> f64 {
            self.inner.s_at(tau)
        }

        /// Exact ground-state energy of `H1`.
        fn problem_ground_energy(&self) -> PyResult<f64> {
            self.inner
                .problem_ground_energy()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        fn qubits(&self) -> usize {
            self.inner.qubits()
        }

        fn __repr__(&self) -> String {
            format!(
                "AdiabaticSchedule(qubits={}, path={})",
                self.inner.qubits(),
                self.inner.path().label()
            )
        }
    }

    /// Result of an adiabatic run. `final_energy` close to `exact_ground_energy`
    /// means the state tracked the ground state.
    #[pyclass]
    struct PyAdiabaticResult {
        inner: crate::aqc::AdiabaticResult,
    }

    #[pymethods]
    impl PyAdiabaticResult {
        #[getter]
        fn final_state(&self) -> Vec<(f64, f64)> {
            self.inner
                .final_state
                .iter()
                .map(|c| (c.re, c.im))
                .collect()
        }

        #[getter]
        fn initial_energy(&self) -> f64 {
            self.inner.initial_energy
        }

        #[getter]
        fn final_energy(&self) -> f64 {
            self.inner.final_energy
        }

        #[getter]
        fn exact_ground_energy(&self) -> f64 {
            self.inner.exact_ground_energy
        }

        #[getter]
        fn total_time(&self) -> f64 {
            self.inner.total_time
        }

        fn __repr__(&self) -> String {
            format!(
                "AdiabaticResult(final_energy={}, exact_ground_energy={})",
                self.inner.final_energy, self.inner.exact_ground_energy
            )
        }
    }

    /// Run the adiabatic interpolation from `s = 0` to `s = 1` over
    /// `total_time`, starting from `initial` (typically the `H0` ground state
    /// `|+...+>`). `tolerance` and `max_steps` control the adaptive integrator.
    #[pyfunction]
    #[pyo3(signature = (schedule, initial, total_time, tolerance = 1e-8, max_steps = 100_000))]
    fn run_adiabatic_py(
        schedule: &PyAdiabaticSchedule,
        initial: &PyStateVector,
        total_time: f64,
        tolerance: f64,
        max_steps: usize,
    ) -> PyResult<PyAdiabaticResult> {
        let result = crate::aqc::run_adiabatic(
            &schedule.inner,
            &initial.inner,
            total_time,
            tolerance,
            max_steps,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyAdiabaticResult { inner: result })
    }

    // ------------------------------------------------------------------
    // Quantum Turing machines
    // ------------------------------------------------------------------

    /// One entry of a QTM transition table: the state entered, the symbol
    /// written, and the head movement.
    #[pyclass]
    #[derive(Clone)]
    struct PyQtmTransition {
        inner: crate::qtm::Transition,
    }

    #[pymethods]
    impl PyQtmTransition {
        #[new]
        fn new(next_state: usize, write: u8, direction: &str) -> PyResult<Self> {
            let direction = match direction {
                "left" | "L" => crate::qtm::Direction::Left,
                "right" | "R" => crate::qtm::Direction::Right,
                "stay" | "S" => crate::qtm::Direction::Stay,
                other => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "unknown direction '{other}'; expected left, right or stay"
                    )));
                }
            };
            Ok(Self {
                inner: crate::qtm::Transition {
                    next_state,
                    write,
                    direction,
                },
            })
        }

        #[getter]
        fn next_state(&self) -> usize {
            self.inner.next_state
        }

        #[getter]
        fn write(&self) -> u8 {
            self.inner.write
        }

        fn direction(&self) -> &'static str {
            match self.inner.direction {
                crate::qtm::Direction::Left => "left",
                crate::qtm::Direction::Right => "right",
                crate::qtm::Direction::Stay => "stay",
            }
        }

        fn __repr__(&self) -> String {
            format!(
                "QtmTransition(next_state={}, write={}, direction={})",
                self.inner.next_state,
                self.inner.write,
                self.direction()
            )
        }
    }

    /// A classical QTM configuration: tape contents, head position, state.
    #[pyclass]
    #[derive(Clone)]
    struct PyQtmConfiguration {
        inner: crate::qtm::Configuration,
    }

    #[pymethods]
    impl PyQtmConfiguration {
        #[new]
        fn new(tape: Vec<u8>, head: usize, state: usize) -> Self {
            Self {
                inner: crate::qtm::Configuration { tape, head, state },
            }
        }

        #[getter]
        fn tape(&self) -> Vec<u8> {
            self.inner.tape.clone()
        }

        #[getter]
        fn head(&self) -> usize {
            self.inner.head
        }

        #[getter]
        fn state(&self) -> usize {
            self.inner.state
        }

        fn __repr__(&self) -> String {
            format!(
                "QtmConfiguration(head={}, state={}, tape={:?})",
                self.inner.head, self.inner.state, self.inner.tape
            )
        }
    }

    /// A declarative quantum Turing machine over a finite tape. The transition
    /// table must be a bijection, or the unitary synthesis is rejected.
    #[pyclass]
    struct PyQuantumTuringMachine {
        inner: crate::qtm::QuantumTuringMachine,
    }

    #[pymethods]
    impl PyQuantumTuringMachine {
        /// `transitions[state][read]` is `QtmTransition` or `None`.
        #[new]
        fn new(
            states: usize,
            alphabet: usize,
            tape_cells: usize,
            head_cells: usize,
            transitions: Vec<Vec<Option<PyQtmTransition>>>,
            initial_state: usize,
        ) -> PyResult<Self> {
            let transitions = transitions
                .into_iter()
                .map(|row| row.into_iter().map(|t| t.map(|t| t.inner)).collect())
                .collect();
            Ok(Self {
                inner: crate::qtm::QuantumTuringMachine::new(
                    states,
                    alphabet,
                    tape_cells,
                    head_cells,
                    transitions,
                    initial_state,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            })
        }

        fn qubits(&self) -> usize {
            self.inner.qubits()
        }

        /// Whether the transition table is a reversible bijection.
        fn is_buildable(&self) -> PyResult<()> {
            self.inner
                .is_buildable()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        }

        /// Number of reachable configurations the table fails to match.
        fn reversibility_gap(&self) -> usize {
            self.inner.reversibility_gap()
        }

        /// Entropy produced per step, in nats.
        fn entropy_production_per_step(&self) -> f64 {
            self.inner.entropy_production_per_step()
        }

        /// Landauer heat dissipated per step at `temperature` (in kT units).
        fn landauer_heat_per_step(&self, temperature: f64) -> f64 {
            self.inner.landauer_heat_per_step(temperature)
        }

        /// One classical step; `None` when the machine halts.
        fn step_classical(&self, config: &PyQtmConfiguration) -> Option<PyQtmConfiguration> {
            self.inner
                .step_classical(&config.inner)
                .map(|inner| PyQtmConfiguration { inner })
        }

        /// Run classically for at most `max_steps`.
        fn run_classical(
            &self,
            initial: &PyQtmConfiguration,
            max_steps: usize,
        ) -> PyQtmConfiguration {
            PyQtmConfiguration {
                inner: self.inner.run_classical(&initial.inner, max_steps),
            }
        }

        fn __repr__(&self) -> String {
            format!("QuantumTuringMachine({} qubits)", self.inner.qubits())
        }
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
        m.add_class::<PyParameterizedCircuit>()?;
        m.add_class::<PyQAOAAnsatzBuilder>()?;
        m.add_class::<PyOptimizeConfig>()?;
        m.add_class::<PyQecCode>()?;
        m.add_class::<PyMPS>()?;
        m.add_class::<PyTensor>()?;
        m.add_class::<PyTEBD>()?;
        m.add_class::<PyDMRG>()?;

        // Gradient strategies
        m.add_class::<PyGradientResult>()?;
        m.add_class::<PyCircuitProfile>()?;
        m.add_function(wrap_pyfunction!(gradient_strategies, m)?)?;

        // Exact-rational analytic core
        m.add_class::<PyRational>()?;
        m.add_class::<PyExactComplex>()?;
        m.add_class::<PyPauliSum>()?;
        m.add_class::<PyElectronIntegrals>()?;
        m.add_class::<PyAdaptVqeResult>()?;

        // SubsCoRe
        m.add_class::<PySubsCoReOutcome>()?;
        m.add_function(wrap_pyfunction!(vqe_subscore_py, m)?)?;

        // Stabilizer formalism
        m.add_class::<PyStabilizerTableau>()?;

        // Cost-aware optimization
        m.add_class::<PyCircuitCostProfile>()?;
        m.add_class::<PyCircuitCostModel>()?;
        m.add_class::<PyCostAwareOptimizeResult>()?;
        m.add_function(wrap_pyfunction!(cost_profile_py, m)?)?;
        m.add_function(wrap_pyfunction!(circuit_depth_py, m)?)?;
        m.add_function(wrap_pyfunction!(count_two_qubit_py, m)?)?;
        m.add_function(wrap_pyfunction!(optimize_for_cost_model_py, m)?)?;

        // QAOA parameter transfer
        m.add_class::<PyMaxCutProblem>()?;
        m.add_class::<PyTransferOptions>()?;
        m.add_class::<PyTransferOutcome>()?;
        m.add_function(wrap_pyfunction!(run_parameter_transfer_py, m)?)?;

        // Adiabatic quantum computing
        m.add_class::<PyAnnealPath>()?;
        m.add_class::<PyAdiabaticSchedule>()?;
        m.add_class::<PyAdiabaticResult>()?;
        m.add_function(wrap_pyfunction!(run_adiabatic_py, m)?)?;

        // Quantum Turing machines
        m.add_class::<PyQtmTransition>()?;
        m.add_class::<PyQtmConfiguration>()?;
        m.add_class::<PyQuantumTuringMachine>()?;

        // Noise / thermodynamics
        m.add_function(wrap_pyfunction!(thermal_gibbs_py, m)?)?;
        m.add_function(wrap_pyfunction!(shot_noise_temperature_py, m)?)?;
        m.add_function(wrap_pyfunction!(vqt_py, m)?)?;
        m.add_function(wrap_pyfunction!(dissipative_prepare_py, m)?)?;

        m.add_function(wrap_pyfunction!(qft_circuit_py, m)?)?;
        m.add_function(wrap_pyfunction!(qpe_circuit_py, m)?)?;
        m.add_function(wrap_pyfunction!(sto3g_integrals_py, m)?)?;
        m.add_function(wrap_pyfunction!(molecular_hamiltonian_py, m)?)?;
        m.add_function(wrap_pyfunction!(run_adapt_vqe_py, m)?)?;
        m.add_function(wrap_pyfunction!(hhl_2x2_py, m)?)?;
        m.add_function(wrap_pyfunction!(shor_period_finding_py, m)?)?;
        m.add_function(wrap_pyfunction!(hhl_solve_py, m)?)?;
        m.add_function(wrap_pyfunction!(shor_factor_py, m)?)?;
        m.add_function(wrap_pyfunction!(hhl_solve_general_py, m)?)?;
        m.add_function(wrap_pyfunction!(hhl_solve_quantum_general_py, m)?)?;
        m.add_function(wrap_pyfunction!(shor_factor_large_py, m)?)?;
        m.add_function(wrap_pyfunction!(shor_factor_quantum_py, m)?)?;
        m.add_function(wrap_pyfunction!(export_qasm_py, m)?)?;
        m.add_function(wrap_pyfunction!(parse_qasm_py, m)?)?;
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
