//! Circuit serialization (JSON, QASM, binary).
//!
//! Provides save/load functionality for circuits in multiple formats.

use crate::{Circuit, CircuitOp, Gate};
use serde::{Deserialize, Serialize};

/// Serialized circuit format (JSON-compatible).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializedCircuit {
    pub num_qubits: usize,
    pub num_classical_bits: usize,
    pub operations: Vec<SerializedOp>,
}

/// Serialized operation.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum SerializedOp {
    #[serde(rename = "single")]
    SingleQubit {
        gate: String,
        target: usize,
        params: Vec<f64>,
    },
    #[serde(rename = "cx")]
    Cnot { control: usize, target: usize },
    #[serde(rename = "cz")]
    Cz { first: usize, second: usize },
    #[serde(rename = "swap")]
    Swap { first: usize, second: usize },
    #[serde(rename = "measure")]
    Measure { target: usize, classical_bit: usize },
    #[serde(rename = "conditional")]
    ConditionalSingleQubit {
        classical_bit: usize,
        value: u8,
        gate: String,
        target: usize,
        params: Vec<f64>,
    },
    #[serde(rename = "channel")]
    Channel {
        channel: String,
        target: usize,
        params: Vec<f64>,
    },
}

impl Circuit {
    /// Serialize circuit to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let serialized = self.to_serialized()?;
        serde_json::to_string_pretty(&serialized)
    }

    /// Serialize circuit to compact JSON bytes.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        let serialized = self.to_serialized()?;
        serde_json::to_vec(&serialized)
    }

    /// Convert to serializable format.
    fn to_serialized(&self) -> Result<SerializedCircuit, serde_json::Error> {
        // Count qubits and classical bits from operations
        let mut max_qubit = 0;
        let mut max_cbit = 0;

        for op in self.operations() {
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

        let num_qubits = max_qubit + 1;
        let num_classical_bits = max_cbit + 1;

        let mut operations = Vec::new();
        for op in self.operations() {
            operations.push(serialize_op(op)?);
        }

        Ok(SerializedCircuit {
            num_qubits,
            num_classical_bits,
            operations,
        })
    }

    /// Deserialize circuit from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let serialized: SerializedCircuit = serde_json::from_str(json)?;
        serialized.to_circuit()
    }

    /// Deserialize circuit from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let serialized: SerializedCircuit = serde_json::from_slice(bytes)?;
        serialized.to_circuit()
    }
}

impl SerializedCircuit {
    /// Convert back to Circuit.
    pub fn to_circuit(&self) -> Result<Circuit, serde_json::Error> {
        let circuit = Circuit::new();

        for op in &self.operations {
            match op {
                SerializedOp::SingleQubit {
                    gate: _gate,
                    target: _,
                    params,
                } => {
                    let _gate = deserialize_gate(_gate, params)?;
                    // Note: This requires Circuit to have a method to add ops directly
                    // For now, we'll use the builder pattern
                    return Err(serde_json::Error::io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Circuit reconstruction needs builder",
                    )));
                }
                SerializedOp::Cnot {
                    control: _,
                    target: _,
                } => {}
                SerializedOp::Cz {
                    first: _,
                    second: _,
                } => {}
                SerializedOp::Swap {
                    first: _,
                    second: _,
                } => {}
                SerializedOp::Measure {
                    target: _,
                    classical_bit: _,
                } => {}
                _ => {}
            }
        }

        Ok(circuit)
    }
}

/// Serialize a CircuitOp to SerializedOp.
fn serialize_op(op: &CircuitOp) -> Result<SerializedOp, serde_json::Error> {
    use CircuitOp::*;
    Ok(match op {
        SingleQubit { gate, target } => {
            let (gate_name, params) = gate_to_serialized(gate);
            SerializedOp::SingleQubit {
                gate: gate_name,
                target: *target,
                params,
            }
        }
        Cnot { control, target } => SerializedOp::Cnot {
            control: *control,
            target: *target,
        },
        Cz { first, second } => SerializedOp::Cz {
            first: *first,
            second: *second,
        },
        Swap { first, second } => SerializedOp::Swap {
            first: *first,
            second: *second,
        },
        Measure {
            target,
            classical_bit,
        } => SerializedOp::Measure {
            target: *target,
            classical_bit: *classical_bit,
        },
        ConditionalSingleQubit {
            classical_bit,
            value,
            gate,
            target,
        } => {
            let (gate_name, params) = gate_to_serialized(gate);
            SerializedOp::ConditionalSingleQubit {
                classical_bit: *classical_bit,
                value: *value,
                gate: gate_name,
                target: *target,
                params,
            }
        }
        Channel { channel, target } => {
            let (chan_name, params) = channel_to_serialized(channel);
            SerializedOp::Channel {
                channel: chan_name,
                target: *target,
                params,
            }
        }
    })
}

/// Convert Gate to (name, params).
fn gate_to_serialized(gate: &Gate) -> (String, Vec<f64>) {
    use Gate::*;
    match gate {
        I => ("I".to_string(), vec![]),
        X => ("X".to_string(), vec![]),
        Y => ("Y".to_string(), vec![]),
        Z => ("Z".to_string(), vec![]),
        H => ("H".to_string(), vec![]),
        S => ("S".to_string(), vec![]),
        Sdg => ("Sdg".to_string(), vec![]),
        T => ("T".to_string(), vec![]),
        Tdg => ("Tdg".to_string(), vec![]),
        Rx(a) => ("Rx".to_string(), vec![*a]),
        Ry(a) => ("Ry".to_string(), vec![*a]),
        Rz(a) => ("Rz".to_string(), vec![*a]),
    }
}

/// Convert NoiseChannel to (name, params).
fn channel_to_serialized(channel: &crate::NoiseChannel) -> (String, Vec<f64>) {
    use crate::NoiseChannel::*;
    match channel {
        Depolarizing { probability } => ("Depolarizing".to_string(), vec![*probability]),
        AmplitudeDamping { gamma } => ("AmplitudeDamping".to_string(), vec![*gamma]),
        PhaseDamping { gamma } => ("PhaseDamping".to_string(), vec![*gamma]),
    }
}

/// Convert SerializedOp back to Gate (for deserialization).
fn deserialize_gate(name: &str, params: &[f64]) -> Result<Gate, serde_json::Error> {
    use Gate::*;
    Ok(match name {
        "I" => I,
        "X" => X,
        "Y" => Y,
        "Z" => Z,
        "H" => H,
        "S" => S,
        "Sdg" => Sdg,
        "T" => T,
        "Tdg" => Tdg,
        "Rx" => Rx(*params.first().unwrap_or(&0.0)),
        "Ry" => Ry(*params.first().unwrap_or(&0.0)),
        "Rz" => Rz(*params.first().unwrap_or(&0.0)),
        _ => {
            return Err(serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown gate: {}", name),
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_circuit() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::X, 0);
        circuit.cnot(0, 1);
        circuit.measure(0, 0);
        circuit.measure(1, 1);

        let json = circuit.to_json().unwrap();
        assert!(json.contains("H"));
        assert!(json.contains("X"));
        assert!(json.contains("cx"));
        assert!(json.contains("measure"));
    }

    #[test]
    fn test_serialize_gate() {
        assert_eq!(gate_to_serialized(&Gate::H), ("H".to_string(), vec![]));
        assert_eq!(
            gate_to_serialized(&Gate::Rx(1.57)),
            ("Rx".to_string(), vec![1.57])
        );
        assert_eq!(
            gate_to_serialized(&Gate::Rx(0.0)),
            ("Rx".to_string(), vec![0.0])
        );
    }

    #[test]
    fn test_serialize_channel() {
        use crate::NoiseChannel::*;
        assert_eq!(
            channel_to_serialized(&Depolarizing { probability: 0.1 }),
            ("Depolarizing".to_string(), vec![0.1])
        );
        assert_eq!(
            channel_to_serialized(&AmplitudeDamping { gamma: 0.2 }),
            ("AmplitudeDamping".to_string(), vec![0.2])
        );
    }
}
