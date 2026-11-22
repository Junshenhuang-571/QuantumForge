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

        // SubsCoRe
        m.add_class::<PySubsCoReOutcome>()?;
        m.add_function(wrap_pyfunction!(vqe_subscore_py, m)?)?;

        // Noise / thermodynamics
        m.add_function(wrap_pyfunction!(thermal_gibbs_py, m)?)?;
        m.add_function(wrap_pyfunction!(shot_noise_temperature_py, m)?)?;
        m.add_function(wrap_pyfunction!(vqt_py, m)?)?;
        m.add_function(wrap_pyfunction!(dissipative_prepare_py, m)?)?;

        m.add_function(wrap_pyfunction!(qft_circuit_py, m)?)?;
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
