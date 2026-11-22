//! Physically parameterized circuit noise models.
//!
//! This module turns a small set of open-system parameters into explicit
//! circuit noise channels. It currently models independent thermal relaxation
//! and pure dephasing after gates; it does not model idle scheduling, correlated
//! noise, leakage, or readout error.

use crate::{Circuit, CircuitOp, NoiseChannel};
use std::fmt;

/// A two-level thermal noise model parameterized by coherence times and bath state.
///
/// `t1`, `t2`, and gate durations must use the same time unit. `beta_delta` is
/// the dimensionless inverse-temperature gap `beta * (E1 - E0)`, with the
/// excited state at higher energy. The model enforces detailed balance and
/// chooses a pure-dephasing channel so the per-gate coherence factor matches
/// `T2` after thermal relaxation is included.
#[derive(Clone, Debug, PartialEq)]
pub struct ThermalGateNoiseModel {
    t1: f64,
    t2: f64,
    beta_delta: f64,
    single_qubit_gate_time: f64,
    two_qubit_gate_time: f64,
}

/// Summary of the noise channels inserted into a circuit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NoiseApplicationSummary {
    /// Number of thermal relaxation channels inserted.
    pub thermal_channels: usize,
    /// Number of pure-dephasing channels inserted.
    pub dephasing_channels: usize,
}

/// A copy of a circuit with thermal gate noise inserted, plus insertion counts.
#[derive(Clone, Debug, PartialEq)]
pub struct NoisyCircuit {
    /// Circuit with explicit channel operations inserted after gates.
    pub circuit: Circuit,
    /// Number and types of channels inserted by the model.
    pub summary: NoiseApplicationSummary,
}

/// Invalid parameter supplied to [`ThermalGateNoiseModel::new`].
#[derive(Clone, Debug, PartialEq)]
pub enum NoiseModelError {
    /// A time or energy parameter was not finite and in its supported range.
    InvalidParameter {
        /// Name of the invalid parameter.
        name: &'static str,
        /// Supplied value.
        value: f64,
    },
    /// `T2` exceeds the Markovian two-level bound `T2 <= 2*T1`.
    T2ExceedsTwoT1 {
        /// Supplied transverse coherence time.
        t2: f64,
        /// Supplied energy-relaxation time.
        t1: f64,
    },
}

impl fmt::Display for NoiseModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { name, value } => {
                write!(f, "invalid thermal noise parameter {name}={value}")
            }
            Self::T2ExceedsTwoT1 { t2, t1 } => {
                write!(f, "T2={t2} exceeds the Markovian limit 2*T1={}", 2.0 * t1)
            }
        }
    }
}

impl std::error::Error for NoiseModelError {}

impl ThermalGateNoiseModel {
    /// Create a thermal gate-noise model.
    ///
    /// `t1` and `t2` are positive; both gate durations may be zero but must be
    /// finite and nonnegative. `beta_delta` must be finite and nonnegative.
    /// The model assumes Markovian relaxation and requires `t2 <= 2*t1`.
    pub fn new(
        t1: f64,
        t2: f64,
        beta_delta: f64,
        single_qubit_gate_time: f64,
        two_qubit_gate_time: f64,
    ) -> Result<Self, NoiseModelError> {
        for (name, value, allow_zero) in [
            ("t1", t1, false),
            ("t2", t2, false),
            ("beta_delta", beta_delta, true),
            ("single_qubit_gate_time", single_qubit_gate_time, true),
            ("two_qubit_gate_time", two_qubit_gate_time, true),
        ] {
            if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
                return Err(NoiseModelError::InvalidParameter { name, value });
            }
        }
        if t2 > 2.0 * t1 {
            return Err(NoiseModelError::T2ExceedsTwoT1 { t2, t1 });
        }
        Ok(Self {
            t1,
            t2,
            beta_delta,
            single_qubit_gate_time,
            two_qubit_gate_time,
        })
    }

    /// Return the energy-relaxation time `T1` in the model's time unit.
    pub fn t1(&self) -> f64 {
        self.t1
    }

    /// Return the transverse coherence time `T2` in the model's time unit.
    pub fn t2(&self) -> f64 {
        self.t2
    }

    /// Return the dimensionless inverse-temperature gap `beta * Delta`.
    pub fn beta_delta(&self) -> f64 {
        self.beta_delta
    }

    /// Return a new circuit with local thermal and dephasing channels after each gate.
    ///
    /// Single-qubit gates use `single_qubit_gate_time`; CNOT, CZ, and SWAP use
    /// `two_qubit_gate_time` and receive independent noise on each operand.
    /// Measurements and already explicit channels are copied unchanged. Idle
    /// periods and readout errors require separate models.
    pub fn apply(&self, circuit: &Circuit) -> NoisyCircuit {
        let mut noisy = Circuit::new();
        let mut summary = NoiseApplicationSummary::default();
        for operation in circuit.operations() {
            noisy.operations.push(*operation);
            let (qubits, duration): (Vec<usize>, f64) = match *operation {
                CircuitOp::SingleQubit { target, .. }
                | CircuitOp::ConditionalSingleQubit { target, .. } => {
                    (vec![target], self.single_qubit_gate_time)
                }
                CircuitOp::Cnot { control, target } => {
                    (vec![control, target], self.two_qubit_gate_time)
                }
                CircuitOp::Cz { first, second } | CircuitOp::Swap { first, second } => {
                    (vec![first, second], self.two_qubit_gate_time)
                }
                CircuitOp::Channel { .. } | CircuitOp::Measure { .. } => continue,
            };

            let (gamma_down, gamma_phi) = self.channel_parameters(duration);
            if gamma_down > 0.0 {
                for qubit in &qubits {
                    noisy.channel(
                        *qubit,
                        NoiseChannel::ThermalDamping {
                            gamma_down,
                            beta_delta: self.beta_delta,
                        },
                    );
                    summary.thermal_channels += 1;
                }
            }
            if gamma_phi > 0.0 {
                for qubit in qubits {
                    noisy.channel(qubit, NoiseChannel::PhaseDamping { gamma: gamma_phi });
                    summary.dephasing_channels += 1;
                }
            }
        }
        NoisyCircuit {
            circuit: noisy,
            summary,
        }
    }

    fn channel_parameters(&self, duration: f64) -> (f64, f64) {
        if duration == 0.0 {
            return (0.0, 0.0);
        }
        let relaxation_probability = -(-duration / self.t1).exp_m1();
        let boltzmann_factor = (-self.beta_delta).exp();
        let gamma_down = relaxation_probability / (1.0 + boltzmann_factor);
        let gamma_up = gamma_down * boltzmann_factor;
        let thermal_coherence_squared = (1.0 - gamma_up) * (1.0 - gamma_down);
        let target_coherence_squared = (-2.0 * duration / self.t2).exp();
        let gamma_phi =
            (1.0 - target_coherence_squared / thermal_coherence_squared).clamp(0.0, 1.0);
        (gamma_down, gamma_phi)
    }
}
