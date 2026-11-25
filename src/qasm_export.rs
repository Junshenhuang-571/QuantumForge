//! QASM 2.0 exporter — round-trip Circuit to OpenQASM 2.0 string.
//!
//! Complements the QASM parser by exporting circuits to standard OpenQASM 2.0 format.

use crate::{Circuit, Gate};
use std::collections::HashMap;

/// Export a circuit to OpenQASM 2.0 format.
///
/// # Example
///
/// ```
/// use quantumforge::{Circuit, Gate, qasm::parse_qasm, qasm_export::export_qasm};
///
/// let qasm = r#"
/// OPENQASM 2.0;
/// include "qelib1.inc";
/// qreg q[2];
/// creg c[2];
/// h q[0];
/// cx q[0], q[1];
/// measure q[0] -> c[0];
/// "#;
///
/// let circuit = parse_qasm(qasm).unwrap();
/// let exported = export_qasm(&circuit);
/// assert!(exported.contains("OPENQASM 2.0"));
/// assert!(exported.contains("h q[0]"));
/// assert!(exported.contains("cx q[0], q[1]"));
/// ```
pub fn export_qasm(circuit: &Circuit) -> String {
    let mut exporter = QasmExporter::new(circuit);
    exporter.export()
}

/// Export a circuit while retaining the caller's declared register width.
///
/// This is useful after optimization removes every operation on a trailing
/// qubit: the serialized circuit must still have the same register size as the
/// input circuit. Returns an error when an operation addresses a qubit outside
/// the declared register.
pub fn export_qasm_with_qubits(
    circuit: &Circuit,
    num_qubits: usize,
) -> Result<String, &'static str> {
    let mut exporter = QasmExporter::new(circuit);
    if exporter.num_qubits > num_qubits {
        return Err("circuit uses a qubit outside the declared register");
    }
    for qubit in exporter.num_qubits..num_qubits {
        exporter.qubit_map.insert(qubit, ("q".to_string(), qubit));
    }
    exporter.num_qubits = num_qubits;
    Ok(exporter.export())
}

struct QasmExporter<'a> {
    circuit: &'a Circuit,
    /// Map from qubit index to (qreg_name, index_in_reg)
    qubit_map: HashMap<usize, (String, usize)>,
    /// Map from classical bit index to (creg_name, index_in_reg)
    cbit_map: HashMap<usize, (String, usize)>,
    /// Number of qubits detected
    num_qubits: usize,
    /// Number of classical bits detected
    num_cbits: usize,
    /// Whether we've seen measurements (need creg)
    has_measurements: bool,
}

impl<'a> QasmExporter<'a> {
    fn new(circuit: &'a Circuit) -> Self {
        let mut exporter = Self {
            circuit,
            qubit_map: HashMap::new(),
            cbit_map: HashMap::new(),
            num_qubits: 0,
            num_cbits: 0,
            has_measurements: false,
        };
        exporter.analyze();
        exporter
    }

    fn analyze(&mut self) {
        let mut max_qubit = 0;
        let mut max_cbit = 0;

        for op in self.circuit.operations() {
            match op {
                crate::CircuitOp::SingleQubit { target, .. } => {
                    max_qubit = max_qubit.max(*target);
                }
                crate::CircuitOp::Cnot { control, target } => {
                    max_qubit = max_qubit.max(*control).max(*target);
                }
                crate::CircuitOp::Cz { first, second } => {
                    max_qubit = max_qubit.max(*first).max(*second);
                }
                crate::CircuitOp::Swap { first, second } => {
                    max_qubit = max_qubit.max(*first).max(*second);
                }
                crate::CircuitOp::Measure {
                    target,
                    classical_bit,
                } => {
                    max_qubit = max_qubit.max(*target);
                    max_cbit = max_cbit.max(*classical_bit);
                    self.has_measurements = true;
                }
                crate::CircuitOp::ConditionalSingleQubit {
                    classical_bit,
                    target,
                    ..
                } => {
                    max_qubit = max_qubit.max(*target);
                    max_cbit = max_cbit.max(*classical_bit);
                }
                crate::CircuitOp::Channel { target, .. } => {
                    max_qubit = max_qubit.max(*target);
                }
            }
        }

        self.num_qubits = if self.circuit.operations().is_empty() {
            0
        } else {
            max_qubit + 1
        };
        self.num_cbits = max_cbit + 1;

        // Simple single register for now
        if self.num_qubits > 0 {
            for i in 0..self.num_qubits {
                self.qubit_map.insert(i, ("q".to_string(), i));
            }
        }
        if self.has_measurements && self.num_cbits > 0 {
            for i in 0..self.num_cbits {
                self.cbit_map.insert(i, ("c".to_string(), i));
            }
        }
    }

    fn export(&mut self) -> String {
        let mut lines = Vec::new();

        // Header
        lines.push("OPENQASM 2.0;".to_string());
        lines.push("include \"qelib1.inc\";".to_string());
        lines.push("".to_string());

        // Qubit register
        if self.num_qubits > 0 {
            lines.push(format!("qreg q[{}];", self.num_qubits));
        }

        // Classical register (only if measurements exist)
        if self.has_measurements && self.num_cbits > 0 {
            lines.push(format!("creg c[{}];", self.num_cbits));
        }

        lines.push("".to_string());

        // Operations
        for op in self.circuit.operations() {
            if let Some(line) = self.export_op(op) {
                lines.push(line);
            }
        }

        lines.join("\n")
    }

    fn export_op(&self, op: &crate::CircuitOp) -> Option<String> {
        use crate::CircuitOp::*;

        match op {
            SingleQubit { gate, target } => {
                let q = self.format_qubit(*target)?;
                Some(format!("{} {};", self.gate_to_qasm(*gate)?, q))
            }
            Cnot { control, target } => {
                let c = self.format_qubit(*control)?;
                let t = self.format_qubit(*target)?;
                Some(format!("cx {c}, {t};"))
            }
            Cz { first, second } => {
                let f = self.format_qubit(*first)?;
                let s = self.format_qubit(*second)?;
                Some(format!("cz {f}, {s};"))
            }
            Swap { first, second } => {
                let f = self.format_qubit(*first)?;
                let s = self.format_qubit(*second)?;
                // Express SWAP using the universally available qelib CNOT
                // primitive. Some OpenQASM 2 parsers do not load `swap` from
                // qelib1.inc even though it is a common circuit operation.
                Some(format!("cx {f}, {s};\ncx {s}, {f};\ncx {f}, {s};"))
            }
            Measure {
                target,
                classical_bit,
            } => {
                let q = self.format_qubit(*target)?;
                let c = self.format_cbit(*classical_bit)?;
                Some(format!("measure {q} -> {c};"))
            }
            ConditionalSingleQubit {
                classical_bit,
                value,
                gate,
                target,
            } => {
                // QASM 2.0 uses `if (c == val) gate q;`
                let c = self.format_cbit(*classical_bit)?;
                let q = self.format_qubit(*target)?;
                Some(format!(
                    "if ({} == {}) {} {};",
                    c,
                    value,
                    self.gate_to_qasm(*gate)?,
                    q
                ))
            }
            Channel { .. } => {
                // Noise channels not representable in standard QASM 2.0
                None
            }
        }
    }

    fn format_qubit(&self, idx: usize) -> Option<String> {
        self.qubit_map
            .get(&idx)
            .map(|(reg, idx)| format!("{reg}[{idx}]"))
    }

    fn format_cbit(&self, idx: usize) -> Option<String> {
        self.cbit_map
            .get(&idx)
            .map(|(reg, idx)| format!("{reg}[{idx}]"))
    }

    fn gate_to_qasm(&self, gate: Gate) -> Option<String> {
        use Gate::*;
        match gate {
            I => Some("id".to_string()),
            X => Some("x".to_string()),
            Y => Some("y".to_string()),
            Z => Some("z".to_string()),
            H => Some("h".to_string()),
            S => Some("s".to_string()),
            Sdg => Some("sdg".to_string()),
            T => Some("t".to_string()),
            Tdg => Some("tdg".to_string()),
            Rx(angle) => Some(format!("rx({})", format_angle(angle))),
            Ry(angle) => Some(format!("ry({})", format_angle(angle))),
            Rz(angle) => Some(format!("rz({})", format_angle(angle))),
        }
    }
}

fn format_angle(angle: f64) -> String {
    // Try to express as fraction of pi for readability
    let pi = std::f64::consts::PI;
    if angle.abs() < 1e-12 {
        return "0".to_string();
    }

    // Check if it's a simple fraction of pi
    let ratio = angle / pi;
    if ratio.abs() > 1e-10 {
        let denom = (1.0 / ratio).round() as i64;
        if denom != 0 && (ratio - 1.0 / denom as f64).abs() < 1e-10 {
            if denom == 1 {
                return "pi".to_string();
            } else if denom == -1 {
                return "-pi".to_string();
            } else if denom > 0 {
                return format!("pi/{denom}");
            } else {
                return format!("-pi/{}", -denom);
            }
        }
    }

    // Default to decimal
    format!("{angle:.10}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

/// Round-trip test: parse QASM, export, parse again, compare operations.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Circuit, Gate, qasm::parse_qasm};

    #[test]
    fn test_export_simple_bell() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        let exported = export_qasm(&circuit);

        assert!(exported.contains("OPENQASM 2.0"));
        assert!(exported.contains("qreg q[2]"));
        assert!(exported.contains("creg c[2]"));
        assert!(exported.contains("h q[0]"));
        assert!(exported.contains("cx q[0], q[1]"));
        assert!(exported.contains("measure q[0] -> c[0]"));
    }

    #[test]
    fn test_export_single_qubit_gates() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::X, 0);
        circuit.push(Gate::Rx(std::f64::consts::FRAC_PI_2), 0);
        circuit.push(Gate::Rz(std::f64::consts::PI), 0);

        let exported = export_qasm(&circuit);
        assert!(exported.contains("h q[0]"));
        assert!(exported.contains("x q[0]"));
        assert!(exported.contains("rx("));
        assert!(exported.contains("rz("));
    }

    #[test]
    fn test_export_roundtrip_bell() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let circuit1 = parse_qasm(qasm).unwrap();
        let exported = export_qasm(&circuit1);
        let circuit2 = parse_qasm(&exported).unwrap();

        // Operations should be equivalent (order may differ due to commutation)
        assert_eq!(circuit1.operations().len(), circuit2.operations().len());
    }

    #[test]
    fn test_export_two_qubit_gates() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::H, 1);
        circuit.cnot(0, 1);
        circuit.cz(1, 0);
        circuit.swap(0, 1);

        let exported = export_qasm(&circuit);
        assert!(exported.contains("h q[0]"));
        assert!(exported.contains("h q[1]"));
        assert!(exported.contains("cx q[0], q[1]"));
        assert!(exported.contains("cz q[1], q[0]"));
        assert_eq!(exported.matches("cx ").count(), 4);
        assert!(!exported.contains("swap "));
        let roundtrip = parse_qasm(&exported).unwrap();
        assert!(
            crate::optimizer::verify_unitary_equivalent(&circuit, &roundtrip, 2, 1e-9)
                .unwrap()
                .equivalent
        );
    }

    #[test]
    fn test_export_retains_declared_idle_qubits() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        let exported = export_qasm_with_qubits(&circuit, 4).unwrap();
        assert!(exported.contains("qreg q[4];"));

        let error = export_qasm_with_qubits(&circuit, 0).unwrap_err();
        assert_eq!(error, "circuit uses a qubit outside the declared register");
    }

    #[test]
    fn test_format_angle() {
        assert_eq!(format_angle(0.0), "0");
        assert_eq!(format_angle(std::f64::consts::PI), "pi");
        assert_eq!(format_angle(-std::f64::consts::PI), "-pi");
        assert_eq!(format_angle(std::f64::consts::PI / 2.0), "pi/2");
        assert_eq!(format_angle(-std::f64::consts::PI / 2.0), "-pi/2");
        assert_eq!(format_angle(std::f64::consts::PI / 4.0), "pi/4");
    }
}
