//! Python bindings for QuantumForge using PyO3.
//!
//! Enable with `--features python`. Provides a `quantumforge` Python module
//! with classes for Circuit, StateVector, DensityMatrix, and the REPL commands.

#[cfg(feature = "python")]
pub mod python_bindings {
    use crate::{Circuit, DensityMatrix, Gate, StateVector, parse_circuit, qasm::parse_qasm};
    use pyo3::prelude::*;

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

        fn draw(&self) -> String {
            // Use the circuit's draw method via the REPL's draw function
            // For now, return a simple representation
            format!("Circuit with {} operations", self.inner.operations().len())
        }

        fn num_operations(&self) -> usize {
            self.inner.operations().len()
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

        fn depolarize(&mut self, qubit: usize, probability: f64) -> PyResult<()> {
            self.inner
                .depolarize(qubit, probability)
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
        inner: crate::ShotResult,
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

    /// Python module definition
    #[pymodule]
    fn quantumforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyCircuit>()?;
        m.add_class::<PyStateVector>()?;
        m.add_class::<PyDensityMatrix>()?;
        m.add_class::<PyShotResult>()?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::quantumforge;
