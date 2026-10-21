//! Hardware backend interfaces for quantum cloud providers.
//!
//! Provides trait-based abstractions for submitting circuits to
//! IBM Quantum, IonQ, and Rigetti QCS backends.

use crate::{Circuit, CircuitOp, Gate, QuantumError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::result::Result as StdResult;
use std::time::Duration;

type Result<T> = StdResult<T, QuantumError>;

/// Trait for quantum hardware backends.
pub trait HardwareBackend: Send + Sync {
    /// Provider name (e.g., "ibm", "ionq", "rigetti").
    fn name(&self) -> &str;

    /// List available devices/backends.
    fn list_devices(&self) -> Result<Vec<DeviceInfo>>;

    /// Get device properties (qubit count, topology, error rates, etc.).
    fn get_device(&self, device_id: &str) -> Result<DeviceInfo>;

    /// Submit a circuit for execution.
    fn submit_job(&self, job: &JobRequest) -> Result<JobHandle>;

    /// Check job status.
    fn job_status(&self, handle: &JobHandle) -> Result<JobStatus>;

    /// Get job results when complete.
    fn job_results(&self, handle: &JobHandle) -> Result<JobResults>;

    /// Cancel a running job.
    fn cancel_job(&self, handle: &JobHandle) -> Result<()>;

    /// Wait for job completion with timeout.
    fn wait_for_job(&self, handle: &JobHandle, timeout: Duration) -> Result<JobResults> {
        let start = std::time::Instant::now();
        loop {
            let status = self.job_status(handle)?;
            if matches!(
                status,
                JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
            ) {
                return self.job_results(handle);
            }
            if start.elapsed() > timeout {
                return Err(QuantumError::Timeout);
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }
}

/// Device information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub num_qubits: usize,
    pub topology: Vec<(usize, usize)>, // Coupling map edges
    pub basis_gates: Vec<String>,
    pub max_shots: usize,
    pub max_experiments: usize,
    pub simulator: bool,
    pub operational: bool,
    pub avg_readout_error: Option<f64>,
    pub avg_cnot_error: Option<f64>,
    pub avg_single_qubit_error: Option<f64>,
}

/// Job submission request.
#[derive(Debug, Clone, Serialize)]
pub struct JobRequest {
    pub device_id: String,
    /// Circuit serialized as OpenQASM 2.0
    pub circuit_qasm: String,
    pub shots: usize,
    pub seed: Option<u64>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Opaque job handle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct JobHandle {
    pub id: String,
    pub provider: String,
}

/// Job execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Job results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResults {
    pub handle: JobHandle,
    pub status: JobStatus,
    pub counts: HashMap<Vec<u8>, usize>, // bitstring -> count
    pub shots: usize,
    pub execution_time: Duration,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// IBM Quantum backend (requires Qiskit Runtime service).
#[cfg(feature = "ibm")]
pub mod ibm {
    use super::{
        DeviceInfo, HardwareBackend, JobHandle, JobRequest, JobResults, JobStatus, QuantumError,
        Result,
    };
    use reqwest::Client;
    use std::sync::Arc;

    /// IBM Quantum backend using Qiskit Runtime REST API.
    pub struct IbmBackend {
        client: Client,
        api_token: String,
        instance: String, // hub/group/project
        base_url: String,
    }

    impl IbmBackend {
        /// Create a new IBM Quantum backend.
        /// `api_token` from IBM Quantum Platform.
        /// `instance` format: "hub/group/project" (e.g., "ibm-q/open/main").
        pub fn new(api_token: String, instance: String) -> Self {
            Self {
                client: Client::new(),
                api_token,
                instance,
                base_url: "https://api.quantum-computing.ibm.com".to_string(),
            }
        }

        fn auth_header(&self) -> String {
            format!("Bearer {}", self.api_token)
        }
    }

    impl HardwareBackend for IbmBackend {
        fn name(&self) -> &str {
            "ibm"
        }

        fn list_devices(&self) -> Result<Vec<DeviceInfo>> {
            // TODO: Implement REST API call to /v1/devices
            Err(QuantumError::UnsupportedOperation(
                "IBM backend not fully implemented".into(),
            ))
        }

        fn get_device(&self, device_id: &str) -> Result<DeviceInfo> {
            Err(QuantumError::UnsupportedOperation(
                "IBM backend not fully implemented".into(),
            ))
        }

        fn submit_job(&self, _job: &JobRequest) -> Result<JobHandle> {
            Err(QuantumError::UnsupportedOperation(
                "IBM backend not fully implemented".into(),
            ))
        }

        fn job_status(&self, _handle: &JobHandle) -> Result<JobStatus> {
            Err(QuantumError::UnsupportedOperation(
                "IBM backend not fully implemented".into(),
            ))
        }

        fn job_results(&self, _handle: &JobHandle) -> Result<JobResults> {
            Err(QuantumError::UnsupportedOperation(
                "IBM backend not fully implemented".into(),
            ))
        }

        fn cancel_job(&self, _handle: &JobHandle) -> Result<()> {
            Err(QuantumError::UnsupportedOperation(
                "IBM backend not fully implemented".into(),
            ))
        }
    }
}

/// IonQ backend (requires IonQ API access).
#[cfg(feature = "ionq")]
pub mod ionq {
    use super::{
        DeviceInfo, HardwareBackend, JobHandle, JobRequest, JobResults, JobStatus, QuantumError,
        Result,
    };
    use reqwest::Client;

    /// IonQ backend using IonQ REST API.
    pub struct IonqBackend {
        client: Client,
        api_key: String,
        base_url: String,
    }

    impl IonqBackend {
        /// Create a new IonQ backend.
        /// `api_key` from IonQ Cloud Console.
        pub fn new(api_key: String) -> Self {
            Self {
                client: Client::new(),
                api_key,
                base_url: "https://api.ionq.co/v0.3".to_string(),
            }
        }
    }

    impl HardwareBackend for IonqBackend {
        fn name(&self) -> &str {
            "ionq"
        }

        fn list_devices(&self) -> Result<Vec<DeviceInfo>> {
            Err(QuantumError::UnsupportedOperation(
                "IonQ backend not fully implemented".into(),
            ))
        }

        fn get_device(&self, device_id: &str) -> Result<DeviceInfo> {
            Err(QuantumError::UnsupportedOperation(
                "IonQ backend not fully implemented".into(),
            ))
        }

        fn submit_job(&self, _job: &JobRequest) -> Result<JobHandle> {
            Err(QuantumError::UnsupportedOperation(
                "IonQ backend not fully implemented".into(),
            ))
        }

        fn job_status(&self, _handle: &JobHandle) -> Result<JobStatus> {
            Err(QuantumError::UnsupportedOperation(
                "IonQ backend not fully implemented".into(),
            ))
        }

        fn job_results(&self, _handle: &JobHandle) -> Result<JobResults> {
            Err(QuantumError::UnsupportedOperation(
                "IonQ backend not fully implemented".into(),
            ))
        }

        fn cancel_job(&self, _handle: &JobHandle) -> Result<()> {
            Err(QuantumError::UnsupportedOperation(
                "IonQ backend not fully implemented".into(),
            ))
        }
    }
}

/// Rigetti QCS backend (requires QCS credentials).
#[cfg(feature = "rigetti")]
pub mod rigetti {
    use super::{
        DeviceInfo, HardwareBackend, JobHandle, JobRequest, JobResults, JobStatus, QuantumError,
        Result,
    };
    use reqwest::Client;

    /// Rigetti QCS backend.
    pub struct RigettiBackend {
        client: Client,
        api_key: String,
        base_url: String,
    }

    impl RigettiBackend {
        /// Create a new Rigetti QCS backend.
        /// `api_key` from Rigetti QCS.
        pub fn new(api_key: String) -> Self {
            Self {
                client: Client::new(),
                api_key,
                base_url: "https://qcs.rigetti.com".to_string(),
            }
        }
    }

    impl HardwareBackend for RigettiBackend {
        fn name(&self) -> &str {
            "rigetti"
        }

        fn list_devices(&self) -> Result<Vec<DeviceInfo>> {
            Err(QuantumError::UnsupportedOperation(
                "Rigetti backend not fully implemented".into(),
            ))
        }

        fn get_device(&self, device_id: &str) -> Result<DeviceInfo> {
            Err(QuantumError::UnsupportedOperation(
                "Rigetti backend not fully implemented".into(),
            ))
        }

        fn submit_job(&self, _job: &JobRequest) -> Result<JobHandle> {
            Err(QuantumError::UnsupportedOperation(
                "Rigetti backend not fully implemented".into(),
            ))
        }

        fn job_status(&self, _handle: &JobHandle) -> Result<JobStatus> {
            Err(QuantumError::UnsupportedOperation(
                "Rigetti backend not fully implemented".into(),
            ))
        }

        fn job_results(&self, _handle: &JobHandle) -> Result<JobResults> {
            Err(QuantumError::UnsupportedOperation(
                "Rigetti backend not fully implemented".into(),
            ))
        }

        fn cancel_job(&self, _handle: &JobHandle) -> Result<()> {
            Err(QuantumError::UnsupportedOperation(
                "Rigetti backend not fully implemented".into(),
            ))
        }
    }
}

/// Local simulator backend (always available, no credentials needed).
pub mod local {
    use super::{
        DeviceInfo, HardwareBackend, JobHandle, JobRequest, JobResults, JobStatus, QuantumError,
        Result,
    };
    #[cfg(feature = "cuda")]
    use crate::cuda::CudaBackend;

    /// Local CPU/GPU simulator backend.
    #[cfg(feature = "cuda")]
    pub struct LocalBackend {
        cuda_backend: Option<CudaBackend>,
    }

    #[cfg(not(feature = "cuda"))]
    pub struct LocalBackend;

    impl LocalBackend {
        pub fn new() -> Self {
            #[cfg(feature = "cuda")]
            let cuda_backend = CudaBackend::new(0).ok();
            #[cfg(feature = "cuda")]
            return Self { cuda_backend };
            #[cfg(not(feature = "cuda"))]
            return Self;
        }

        pub fn with_cuda(device: i32) -> Result<Self> {
            #[cfg(feature = "cuda")]
            {
                let _ = device; // suppress unused warning when feature is disabled
                Ok(Self {
                    cuda_backend: Some(
                        CudaBackend::new(device)
                            .map_err(|e| QuantumError::UnsupportedOperation(e.to_string()))?,
                    ),
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device;
                Err(QuantumError::UnsupportedOperation(
                    "CUDA feature not enabled".into(),
                ))
            }
        }
    }

    impl Default for LocalBackend {
        fn default() -> Self {
            Self::new()
        }
    }

    impl HardwareBackend for LocalBackend {
        fn name(&self) -> &str {
            "local"
        }

        fn list_devices(&self) -> Result<Vec<DeviceInfo>> {
            #[cfg(feature = "cuda")]
            let mut devices = vec![DeviceInfo {
                id: "cpu".to_string(),
                name: "Local CPU Simulator".to_string(),
                provider: "local".to_string(),
                num_qubits: 30,
                topology: vec![],
                basis_gates: vec![
                    "H".into(),
                    "X".into(),
                    "Y".into(),
                    "Z".into(),
                    "RX".into(),
                    "RY".into(),
                    "RZ".into(),
                    "CNOT".into(),
                    "CZ".into(),
                    "SWAP".into(),
                ],
                max_shots: 1_000_000,
                max_experiments: 1,
                simulator: true,
                operational: true,
                avg_readout_error: None,
                avg_cnot_error: None,
                avg_single_qubit_error: None,
            }];

            #[cfg(not(feature = "cuda"))]
            let devices = vec![DeviceInfo {
                id: "cpu".to_string(),
                name: "Local CPU Simulator".to_string(),
                provider: "local".to_string(),
                num_qubits: 30,
                topology: vec![],
                basis_gates: vec![
                    "H".into(),
                    "X".into(),
                    "Y".into(),
                    "Z".into(),
                    "RX".into(),
                    "RY".into(),
                    "RZ".into(),
                    "CNOT".into(),
                    "CZ".into(),
                    "SWAP".into(),
                ],
                max_shots: 1_000_000,
                max_experiments: 1,
                simulator: true,
                operational: true,
                avg_readout_error: None,
                avg_cnot_error: None,
                avg_single_qubit_error: None,
            }];

            #[cfg(feature = "cuda")]
            if let Some(cuda_backend) = &self.cuda_backend {
                let _ = cuda_backend;
                devices.push(DeviceInfo {
                    id: "cuda:0".to_string(),
                    name: "Local GPU Simulator (CUDA)".to_string(),
                    provider: "local".to_string(),
                    num_qubits: 28,
                    topology: vec![],
                    basis_gates: vec![
                        "H".into(),
                        "X".into(),
                        "Y".into(),
                        "Z".into(),
                        "RX".into(),
                        "RY".into(),
                        "RZ".into(),
                        "CNOT".into(),
                        "CZ".into(),
                        "SWAP".into(),
                    ],
                    max_shots: 100_000,
                    max_experiments: 10_000,
                    simulator: true,
                    operational: true,
                    avg_readout_error: None,
                    avg_cnot_error: None,
                    avg_single_qubit_error: None,
                });
            }

            Ok(devices)
        }

        fn get_device(&self, device_id: &str) -> Result<DeviceInfo> {
            self.list_devices()?
                .into_iter()
                .find(|d| d.id == device_id)
                .ok_or(QuantumError::QubitOutOfRange {
                    qubit: 0,
                    qubits: 0,
                })
        }

        fn submit_job(&self, _job: &JobRequest) -> Result<JobHandle> {
            // Execute locally and return immediate results
            // For async interface, we'd need a thread pool
            Err(QuantumError::UnsupportedOperation(
                "Local backend uses synchronous run_circuit, not submit/poll".into(),
            ))
        }

        fn job_status(&self, _handle: &JobHandle) -> Result<JobStatus> {
            Err(QuantumError::UnsupportedOperation(
                "Local backend doesn't use job handles".into(),
            ))
        }

        fn job_results(&self, _handle: &JobHandle) -> Result<JobResults> {
            Err(QuantumError::UnsupportedOperation(
                "Local backend doesn't use job handles".into(),
            ))
        }

        fn cancel_job(&self, _handle: &JobHandle) -> Result<()> {
            Err(QuantumError::UnsupportedOperation(
                "Local backend doesn't use job handles".into(),
            ))
        }
    }
}

/// Convert QuantumForge Circuit to provider-specific formats.
pub mod transpile {
    use super::{Circuit, CircuitOp, Gate};
    use crate::qasm_export;

    /// Convert circuit to OpenQASM 2.0 string.
    pub fn to_openqasm(circuit: &Circuit) -> String {
        qasm_export::export_qasm(circuit)
    }

    /// Convert circuit to Qiskit-compatible JSON.
    pub fn to_qiskit_json(circuit: &Circuit) -> serde_json::Value {
        let mut gates = Vec::new();
        for op in circuit.operations() {
            match op {
                CircuitOp::SingleQubit { gate, target } => {
                    gates.push(serde_json::json!({
                        "name": gate.to_string().to_lowercase(),
                        "qubits": [target],
                        "params": gate.params(),
                    }));
                }
                CircuitOp::Cnot { control, target } => {
                    gates.push(serde_json::json!({
                        "name": "cx",
                        "qubits": [control, target],
                    }));
                }
                CircuitOp::Cz { first, second } => {
                    gates.push(serde_json::json!({
                        "name": "cz",
                        "qubits": [first, second],
                    }));
                }
                CircuitOp::Swap { first, second } => {
                    gates.push(serde_json::json!({
                        "name": "swap",
                        "qubits": [first, second],
                    }));
                }
                CircuitOp::Measure {
                    target,
                    classical_bit,
                } => {
                    gates.push(serde_json::json!({
                        "name": "measure",
                        "qubits": [target],
                        "clbits": [classical_bit],
                    }));
                }
                CircuitOp::ConditionalSingleQubit {
                    classical_bit,
                    value,
                    gate,
                    target,
                } => {
                    gates.push(serde_json::json!({
                        "name": gate.to_string().to_lowercase(),
                        "qubits": [target],
                        "params": gate.params(),
                        "condition": {
                            "register": "c",
                            "bit": classical_bit,
                            "value": value,
                        },
                    }));
                }
                CircuitOp::Channel { channel, target } => {
                    gates.push(serde_json::json!({
                        "name": format!("{:?}", channel).to_lowercase(),
                        "qubits": [target],
                        "params": channel.params(),
                    }));
                }
            }
        }
        serde_json::json!({
            "n_qubits": circuit.num_qubits(),
            "gates": gates,
        })
    }

    /// Convert circuit to IonQ JSON format.
    pub fn to_ionq_json(circuit: &Circuit) -> serde_json::Value {
        // IonQ uses native gates: GPI, GPI2, MS
        // This would need a transpilation pass
        to_qiskit_json(circuit)
    }

    /// Convert circuit to Rigetti Quil format.
    pub fn to_quil(circuit: &Circuit) -> String {
        let mut lines = vec!["DEFCIRCUIT QUANTUMFORGE_CIRCUIT:".to_string()];
        for op in circuit.operations() {
            match op {
                CircuitOp::SingleQubit { gate, target } => {
                    lines.push(format!("    {} {}", gate_to_quil(*gate), target));
                }
                CircuitOp::Cnot { control, target } => {
                    lines.push(format!("    CNOT {} {}", control, target));
                }
                CircuitOp::Cz { first, second } => {
                    lines.push(format!("    CZ {} {}", first, second));
                }
                CircuitOp::Swap { first, second } => {
                    lines.push(format!("    SWAP {} {}", first, second));
                }
                CircuitOp::Measure {
                    target,
                    classical_bit,
                } => {
                    lines.push(format!("    MEASURE {} ro[{}]", target, classical_bit));
                }
                CircuitOp::ConditionalSingleQubit {
                    classical_bit,
                    value,
                    gate,
                    target,
                } => {
                    lines.push(format!(
                        "    JUMP-WHEN @ro[{}] {} {} {}",
                        classical_bit,
                        value,
                        gate_to_quil(*gate),
                        target
                    ));
                }
                CircuitOp::Channel { channel, target } => {
                    lines.push(format!(
                        "    # Channel {:?} on qubit {} (not native)",
                        channel, target
                    ));
                }
            }
        }
        lines.join("\n")
    }

    fn gate_to_quil(gate: Gate) -> &'static str {
        match gate {
            Gate::H => "H",
            Gate::X => "X",
            Gate::Y => "Y",
            Gate::Z => "Z",
            Gate::S => "S",
            Gate::Sdg => "S†",
            Gate::T => "T",
            Gate::Tdg => "T†",
            Gate::Rx(_) => "RX",
            Gate::Ry(_) => "RY",
            Gate::Rz(_) => "RZ",
            _ => "GATE",
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Circuit, Gate, hardware::HardwareBackend, hardware::local, hardware::transpile};

    #[test]
    fn test_local_backend_devices() {
        let backend = local::LocalBackend::new();
        let devices = backend.list_devices().unwrap();
        assert!(!devices.is_empty());
        assert_eq!(devices[0].id, "cpu");
        assert_eq!(devices[0].provider, "local");
    }

    #[test]
    fn test_transpile_openqasm() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.cnot(0, 1);
        let qasm = transpile::to_openqasm(&circuit);
        assert!(qasm.contains("OPENQASM 2.0"));
        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("cx q[0], q[1]"));
    }

    #[test]
    fn test_transpile_quil() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.cnot(0, 1);
        let quil = transpile::to_quil(&circuit);
        assert!(quil.contains("H 0"));
        assert!(quil.contains("CNOT 0 1"));
    }
}
