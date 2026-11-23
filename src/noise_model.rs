//! Physically parameterized circuit noise models.
//!
//! This module turns a small set of open-system parameters into explicit
//! circuit noise channels. It models independent thermal relaxation and pure
//! dephasing after gates, and can add noise for idle periods when supplied with
//! an explicit operation schedule. It does not model correlated noise, leakage,
//! or readout error.

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
    /// Thermal relaxation channels inserted during qubit idle periods.
    pub idle_thermal_channels: usize,
    /// Pure-dephasing channels inserted during qubit idle periods.
    pub idle_dephasing_channels: usize,
}

/// Start time and duration for one operation in a circuit schedule.
///
/// Entries passed to [`ThermalGateNoiseModel::apply_scheduled`] correspond to
/// circuit operations by index. All values must use the same time unit as the
/// model's `T1` and `T2` parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperationTiming {
    /// Operation start time, relative to the beginning of the schedule.
    pub start: f64,
    /// Wall-clock duration of the operation.
    pub duration: f64,
}

/// A copy of a circuit with explicit thermal/dephasing operations, plus counts.
#[derive(Clone, Debug, PartialEq)]
pub struct NoisyCircuit {
    /// Circuit with gate and, when scheduled, idle channel operations inserted.
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
    /// The schedule does not contain exactly one timing per circuit operation.
    ScheduleLengthMismatch {
        /// Number of circuit operations.
        operations: usize,
        /// Number of supplied timings.
        timings: usize,
    },
    /// A schedule entry contains an invalid start time or duration.
    InvalidScheduleTiming {
        /// Operation index with invalid timing.
        operation: usize,
        /// Supplied start time.
        start: f64,
        /// Supplied duration.
        duration: f64,
    },
    /// Scheduled operations do not preserve program order on a shared qubit.
    QubitScheduleOrderViolation {
        /// Qubit shared by both operations.
        qubit: usize,
        /// Earlier operation in the circuit.
        first_operation: usize,
        /// Later operation in the circuit.
        second_operation: usize,
    },
    /// The supplied schedule reorders operations that share a classical bit.
    ClassicalDependencyViolation {
        /// Earlier operation in the circuit.
        first_operation: usize,
        /// Later operation in the circuit.
        second_operation: usize,
        /// Classical bit shared by the operations.
        classical_bit: usize,
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
            Self::ScheduleLengthMismatch {
                operations,
                timings,
            } => write!(
                f,
                "schedule has {timings} timings for {operations} circuit operations"
            ),
            Self::InvalidScheduleTiming {
                operation,
                start,
                duration,
            } => write!(
                f,
                "invalid schedule timing for operation {operation}: start={start}, duration={duration}"
            ),
            Self::QubitScheduleOrderViolation {
                qubit,
                first_operation,
                second_operation,
            } => write!(
                f,
                "schedule overlaps or reorders operations {first_operation} and {second_operation} on qubit {qubit}"
            ),
            Self::ClassicalDependencyViolation {
                first_operation,
                second_operation,
                classical_bit,
            } => write!(
                f,
                "schedule reorders operations {first_operation} and {second_operation} that share classical bit {classical_bit}"
            ),
        }
    }
}

impl std::error::Error for NoiseModelError {}

impl ThermalGateNoiseModel {
    /// Create a model for use with [`Self::apply_scheduled`].
    ///
    /// Unlike [`Self::new`], this constructor needs no default gate durations:
    /// `apply_scheduled` takes each operation's duration from its schedule.
    /// Calling [`Self::apply`] on a model created this way inserts no gate noise
    /// because the default durations are zero.
    pub fn for_schedule(t1: f64, t2: f64, beta_delta: f64) -> Result<Self, NoiseModelError> {
        Self::new(t1, t2, beta_delta, 0.0, 0.0)
    }

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
    /// Measurements and already explicit channels are copied unchanged. Use
    /// [`Self::apply_scheduled`] to include idle periods. Readout errors require
    /// the separate readout model.
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

    /// Apply gate noise and thermal idle noise using caller-supplied timings.
    ///
    /// The schedule must contain one start/duration pair per circuit operation.
    /// Operations that overlap on a shared qubit are rejected. The operation
    /// order in the circuit is retained; idle channels are inserted at the end
    /// of each idle interval, and gate channels use the supplied gate duration.
    /// Idle time before the first operation and through the schedule makespan is
    /// included. A caller should provide a schedule from a target-aware scheduler
    /// when compiling for hardware; this method does not choose gate start times
    /// or enforce pulse alignment constraints.
    pub fn apply_scheduled(
        &self,
        circuit: &Circuit,
        timings: &[OperationTiming],
    ) -> Result<NoisyCircuit, NoiseModelError> {
        let operations = circuit.operations();
        if timings.len() != operations.len() {
            return Err(NoiseModelError::ScheduleLengthMismatch {
                operations: operations.len(),
                timings: timings.len(),
            });
        }
        for (operation, timing) in timings.iter().enumerate() {
            if !timing.start.is_finite()
                || !timing.duration.is_finite()
                || timing.start < 0.0
                || timing.duration < 0.0
                || !(timing.start + timing.duration).is_finite()
            {
                return Err(NoiseModelError::InvalidScheduleTiming {
                    operation,
                    start: timing.start,
                    duration: timing.duration,
                });
            }
        }
        let mut last_on_classical: std::collections::HashMap<usize, (f64, usize)> =
            std::collections::HashMap::new();
        for (index, operation) in operations.iter().enumerate() {
            let timing = timings[index];
            for bit in operation_classical_bits(*operation) {
                if let Some(&(previous_end, previous_index)) = last_on_classical.get(&bit) {
                    if timing.start < previous_end {
                        return Err(NoiseModelError::ClassicalDependencyViolation {
                            first_operation: previous_index,
                            second_operation: index,
                            classical_bit: bit,
                        });
                    }
                }
                last_on_classical.insert(bit, (timing.start + timing.duration, index));
            }
        }

        let mut occupied: std::collections::HashMap<usize, Vec<(f64, f64, usize)>> =
            std::collections::HashMap::new();
        let mut last_on_qubit: std::collections::HashMap<usize, (f64, usize)> =
            std::collections::HashMap::new();
        for (index, operation) in operations.iter().enumerate() {
            let timing = timings[index];
            for qubit in operation_qubits(*operation) {
                if let Some(&(previous_end, previous_index)) = last_on_qubit.get(&qubit) {
                    if timing.start < previous_end {
                        return Err(NoiseModelError::QubitScheduleOrderViolation {
                            qubit,
                            first_operation: previous_index,
                            second_operation: index,
                        });
                    }
                }
                let end = timing.start + timing.duration;
                last_on_qubit.insert(qubit, (end, index));
                if timing.duration > 0.0 {
                    occupied
                        .entry(qubit)
                        .or_default()
                        .push((timing.start, end, index));
                }
            }
        }

        let makespan = timings
            .iter()
            .map(|timing| timing.start + timing.duration)
            .fold(0.0_f64, f64::max);
        let qubit_count = operations
            .iter()
            .flat_map(|operation| operation_qubits(*operation))
            .max()
            .map_or(0, |qubit| qubit + 1);
        let mut events: Vec<(f64, u8, usize, CircuitOp)> = Vec::new();
        let mut summary = NoiseApplicationSummary::default();

        for (index, operation) in operations.iter().copied().enumerate() {
            let timing = timings[index];
            events.push((timing.start, 1, index, operation));
            let qubits = operation_qubits(operation);
            if qubits.is_empty() || timing.duration == 0.0 {
                continue;
            }
            if matches!(
                operation,
                CircuitOp::Measure { .. } | CircuitOp::Channel { .. }
            ) {
                continue;
            }
            let (gamma_down, gamma_phi) = self.channel_parameters(timing.duration);
            for qubit in qubits {
                if gamma_down > 0.0 {
                    events.push((
                        timing.start + timing.duration,
                        0,
                        index,
                        CircuitOp::Channel {
                            channel: NoiseChannel::ThermalDamping {
                                gamma_down,
                                beta_delta: self.beta_delta,
                            },
                            target: qubit,
                        },
                    ));
                    summary.thermal_channels += 1;
                }
                if gamma_phi > 0.0 {
                    events.push((
                        timing.start + timing.duration,
                        0,
                        index,
                        CircuitOp::Channel {
                            channel: NoiseChannel::PhaseDamping { gamma: gamma_phi },
                            target: qubit,
                        },
                    ));
                    summary.dephasing_channels += 1;
                }
            }
        }

        for qubit in 0..qubit_count {
            let mut cursor = 0.0;
            if let Some(intervals) = occupied.get(&qubit) {
                for &(start, end, index) in intervals {
                    if start > cursor {
                        self.push_idle_channels(
                            &mut events,
                            qubit,
                            start - cursor,
                            start,
                            index,
                            &mut summary,
                        );
                    }
                    cursor = end;
                }
            }
            if makespan > cursor {
                self.push_idle_channels(
                    &mut events,
                    qubit,
                    makespan - cursor,
                    makespan,
                    usize::MAX,
                    &mut summary,
                );
            }
        }

        events.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut noisy = Circuit::new();
        noisy
            .operations_mut()
            .extend(events.into_iter().map(|event| event.3));
        Ok(NoisyCircuit {
            circuit: noisy,
            summary,
        })
    }

    fn push_idle_channels(
        &self,
        events: &mut Vec<(f64, u8, usize, CircuitOp)>,
        qubit: usize,
        duration: f64,
        time: f64,
        index: usize,
        summary: &mut NoiseApplicationSummary,
    ) {
        let (gamma_down, gamma_phi) = self.channel_parameters(duration);
        if gamma_down > 0.0 {
            events.push((
                time,
                0,
                index,
                CircuitOp::Channel {
                    channel: NoiseChannel::ThermalDamping {
                        gamma_down,
                        beta_delta: self.beta_delta,
                    },
                    target: qubit,
                },
            ));
            summary.thermal_channels += 1;
            summary.idle_thermal_channels += 1;
        }
        if gamma_phi > 0.0 {
            events.push((
                time,
                0,
                index,
                CircuitOp::Channel {
                    channel: NoiseChannel::PhaseDamping { gamma: gamma_phi },
                    target: qubit,
                },
            ));
            summary.dephasing_channels += 1;
            summary.idle_dephasing_channels += 1;
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

fn operation_qubits(operation: CircuitOp) -> Vec<usize> {
    match operation {
        CircuitOp::SingleQubit { target, .. }
        | CircuitOp::ConditionalSingleQubit { target, .. }
        | CircuitOp::Channel { target, .. }
        | CircuitOp::Measure { target, .. } => vec![target],
        CircuitOp::Cnot { control, target } => vec![control, target],
        CircuitOp::Cz { first, second } | CircuitOp::Swap { first, second } => {
            vec![first, second]
        }
    }
}

fn operation_classical_bits(operation: CircuitOp) -> Vec<usize> {
    match operation {
        CircuitOp::Measure { classical_bit, .. } => vec![classical_bit],
        CircuitOp::ConditionalSingleQubit { classical_bit, .. } => vec![classical_bit],
        _ => Vec::new(),
    }
}
