//! Gradient strategies for parameterized circuits.
//!
//! A production variational stack offers several ways to differentiate
//! `f(x) = <psi_x|O|psi_x>`, and the right one depends on the circuit and on
//! whether the expectation itself is sampled:
//!
//! | strategy | exact? | needs | cost |
//! |---|---|---|---|
//! | `ParameterShift` | yes | two eigenvalues in the generator | 2 evals per parameterized occurrence; works with finite shots |
//! | `Adjoint` | yes | state vector | one forward + one backward sweep |
//! | `Backprop` | yes | state vector | one forward + one backward sweep; independent implementation |
//! | `FiniteDifference` | no (`O(h^2)`) | anything | 2 evals per parameterized occurrence |
//!
//! [`GradientStrategy::Auto`] picks the cheapest exact rule for a
//! [`CircuitProfile`]: `ParameterShift` when shots are requested (the only
//! exact rule that supports sampling), otherwise `Adjoint`. Circuits
//! containing a noise channel return `None`, because their gradient lives on
//! the density-matrix path.
//!
//! ```
//! use quantumforge::gradients::{
//!     CircuitProfile, GradientOptions, GradientStrategy,
//! };
//! use quantumforge::{ParameterizedCircuit, RotationAxis, SingleQubitObservable};
//!
//! let mut circuit = ParameterizedCircuit::new(1);
//! circuit.rotation(RotationAxis::Y, 0, 0);
//! circuit.cnot(0, 1);
//!
//! let profile = circuit.circuit_profile();
//! assert_eq!(GradientStrategy::select(&profile), Some(GradientStrategy::Adjoint));
//!
//! let options = GradientOptions::default();
//! let estimate = circuit
//!     .gradient(&options, &[0.4], 2, SingleQubitObservable::Z, 0)
//!     .unwrap();
//! assert_eq!(estimate.strategy, GradientStrategy::Adjoint);
//! assert_eq!(estimate.gradient.len(), 1);
//! ```

use crate::{
    ParameterizedCircuit, ParameterizedOp, QuantumError, RotationAxis, SingleQubitObservable,
    StateVector, adjoint_gate, rotation_gate,
};

/// Which differentiation rule a variational gradient uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientStrategy {
    /// Parameter-shift rule. Exact for generators with two distinct
    /// eigenvalues, which covers every rotation in [`ParameterizedOp`], and
    /// the only exact rule here that supports finite shots.
    ParameterShift,
    /// Adjoint method: differentiate the expectation by carrying a costate
    /// backwards through the circuit. Exact, state vector only, and the
    /// cheapest exact rule for a single observable.
    Adjoint,
    /// Reverse-mode backpropagation through the gate list. Exact and state
    /// vector only. Algebraically the same reverse-mode computation as
    /// [`Self::Adjoint`] for a Pauli observable; it is kept as a separate
    /// implementation so the two cross-check each other and so a generic
    /// reverse-mode path exists for gate types the adjoint shortcut does not
    /// cover.
    Backprop,
    /// Central finite differences with a configurable step. Approximate, but
    /// valid for any differentiable gate — the fallback when no shift rule is
    /// known.
    FiniteDifference,
    /// Resolve with [`GradientStrategy::select`].
    Auto,
}

impl GradientStrategy {
    /// The cheapest rule that applies to `profile`, or `None` when the circuit
    /// needs the density-matrix path.
    ///
    /// * a noise channel disqualifies every state-vector rule;
    /// * finite shots force [`Self::ParameterShift`], the only exact sampled
    ///   rule;
    /// * otherwise [`Self::Adjoint`], which costs one backward sweep instead of
    ///   two evaluations per parameterized gate.
    pub fn select(profile: &CircuitProfile) -> Option<Self> {
        if profile.has_channel {
            return None;
        }
        if profile.shots > 0 {
            return Some(Self::ParameterShift);
        }
        Some(Self::Adjoint)
    }

    /// Human-readable name, as used by the `qf` REPL.
    pub fn name(self) -> &'static str {
        match self {
            Self::ParameterShift => "parameter-shift",
            Self::Adjoint => "adjoint",
            Self::Backprop => "backprop",
            Self::FiniteDifference => "finite-difference",
            Self::Auto => "auto",
        }
    }

    /// Whether the rule is exact rather than a finite-difference
    /// approximation.
    pub fn is_exact(self) -> bool {
        !matches!(self, Self::FiniteDifference)
    }

    /// Parse a strategy from the name used in [`Self::name`] and the REPL.
    ///
    /// # Panics
    ///
    /// Panics on an unknown name.
    pub fn parse(name: &str) -> Self {
        match name {
            "parameter-shift" | "shift" => Self::ParameterShift,
            "adjoint" => Self::Adjoint,
            "backprop" | "backpropagation" => Self::Backprop,
            "finite-difference" | "finite-diff" | "fd" => Self::FiniteDifference,
            "auto" => Self::Auto,
            other => panic!("unknown gradient strategy '{other}'"),
        }
    }
}

/// Static facts about a circuit that decide which rule applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CircuitProfile {
    /// Number of operations in the circuit.
    pub operations: usize,
    /// Number of trainable parameters.
    pub parameters: usize,
    /// Number of parameterized rotation operations. Shared parameters count
    /// once per gate, because every occurrence is differentiated separately.
    pub parameterized_operations: usize,
    /// Whether the circuit contains a noise channel.
    pub has_channel: bool,
    /// Shots requested for the expectation; zero means an exact state vector.
    pub shots: usize,
}

/// Tuning for [`ParameterizedCircuit::gradient`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientOptions {
    /// Which rule to use; [`GradientStrategy::Auto`] resolves it from the
    /// circuit.
    pub strategy: GradientStrategy,
    /// Central-difference step for [`GradientStrategy::FiniteDifference`].
    pub finite_difference_step: f64,
}

impl Default for GradientOptions {
    fn default() -> Self {
        Self {
            strategy: GradientStrategy::Auto,
            finite_difference_step: 1e-3,
        }
    }
}

/// A gradient together with how it was produced.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientEstimate {
    /// `d<observable>/d(parameter)`, one entry per parameter.
    pub gradient: Vec<f64>,
    /// The rule that actually ran (`Auto` is resolved to a concrete rule).
    pub strategy: GradientStrategy,
    /// State-vector sweeps consumed: two per parameterized occurrence for the
    /// shift and finite-difference rules, one forward plus one backward for
    /// the reverse-mode rules.
    pub evaluations: usize,
}

impl ParameterizedCircuit {
    /// Facts about this circuit used by [`GradientStrategy::select`].
    pub fn circuit_profile(&self) -> CircuitProfile {
        let mut profile = CircuitProfile {
            operations: self.operations.len(),
            parameters: self.parameter_count,
            ..CircuitProfile::default()
        };
        for operation in &self.operations {
            match operation {
                ParameterizedOp::Rotation { .. } => {
                    profile.parameterized_operations += 1;
                }
                ParameterizedOp::Channel { .. } => profile.has_channel = true,
                _ => {}
            }
        }
        profile
    }

    /// Gradient of `<observable>` on `observable_qubit` with the chosen rule.
    pub fn gradient(
        &self,
        options: &GradientOptions,
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<GradientEstimate, QuantumError> {
        self.gradient_with_inputs(
            options,
            &[],
            parameters,
            qubits,
            observable,
            observable_qubit,
        )
    }

    /// Gradient with encoded inputs and the chosen rule.
    ///
    /// `Auto` resolves to [`GradientStrategy::ParameterShift`] when
    /// `options.strategy` is `Auto` and a shot budget is set through
    /// [`CircuitProfile::shots`] — callers that want sampled gradients should
    /// pass [`GradientStrategy::ParameterShift`] explicitly, since the exact
    /// API here has no shot parameter.
    ///
    /// # Errors
    ///
    /// Returns [`QuantumError::OperationRequiresDensityMatrix`] for a circuit
    /// with a noise channel, and the usual validation errors otherwise.
    pub fn gradient_with_inputs(
        &self,
        options: &GradientOptions,
        inputs: &[f64],
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<GradientEstimate, QuantumError> {
        let mut profile = self.circuit_profile();
        profile.shots = 0;
        let strategy = match options.strategy {
            GradientStrategy::Auto => GradientStrategy::select(&profile)
                .ok_or(QuantumError::OperationRequiresDensityMatrix)?,
            explicit => explicit,
        };
        match strategy {
            GradientStrategy::ParameterShift => {
                let occurrences = profile.parameterized_operations;
                let gradient = self.parameter_shift_gradient_with_inputs(
                    inputs,
                    parameters,
                    qubits,
                    observable,
                    observable_qubit,
                )?;
                Ok(GradientEstimate {
                    gradient,
                    strategy,
                    evaluations: 2 * occurrences,
                })
            }
            GradientStrategy::Adjoint => {
                let gradient = self.adjoint_gradient_with_inputs(
                    inputs,
                    parameters,
                    qubits,
                    observable,
                    observable_qubit,
                )?;
                Ok(GradientEstimate {
                    gradient,
                    strategy,
                    evaluations: 2,
                })
            }
            GradientStrategy::Backprop => {
                let gradient = backprop_gradient_with_inputs(
                    self,
                    inputs,
                    parameters,
                    qubits,
                    observable,
                    observable_qubit,
                )?;
                Ok(GradientEstimate {
                    gradient,
                    strategy,
                    evaluations: 2,
                })
            }
            GradientStrategy::FiniteDifference => {
                let occurrences = profile.parameterized_operations;
                let gradient = finite_difference_gradient_with_inputs(
                    self,
                    inputs,
                    parameters,
                    qubits,
                    observable,
                    observable_qubit,
                    options.finite_difference_step,
                )?;
                Ok(GradientEstimate {
                    gradient,
                    strategy,
                    evaluations: 2 * occurrences,
                })
            }
            GradientStrategy::Auto => unreachable!("Auto is resolved above"),
        }
    }

    /// Gradient with the default options ([`GradientStrategy::Auto`]).
    pub fn gradient_auto(
        &self,
        parameters: &[f64],
        qubits: usize,
        observable: SingleQubitObservable,
        observable_qubit: usize,
    ) -> Result<GradientEstimate, QuantumError> {
        self.gradient(
            &GradientOptions::default(),
            parameters,
            qubits,
            observable,
            observable_qubit,
        )
    }
}

/// Central-difference gradient, differentiated per gate occurrence so that a
/// parameter shared by several gates gets the sum of its occurrence-wise
/// derivatives, matching [`ParameterizedCircuit::parameter_shift_gradient`].
///
/// Exact for neither the shift rule nor the reverse-mode rules; its job is to
/// be valid for any differentiable gate and to anchor the other rules in tests.
///
/// # Errors
///
/// Returns the usual validation errors, plus
/// [`QuantumError::InvalidFiniteDifferenceStep`] when `step` is not positive
/// and finite.
pub fn finite_difference_gradient_with_inputs(
    circuit: &ParameterizedCircuit,
    inputs: &[f64],
    parameters: &[f64],
    qubits: usize,
    observable: SingleQubitObservable,
    observable_qubit: usize,
    step: f64,
) -> Result<Vec<f64>, QuantumError> {
    if !step.is_finite() || step <= 0.0 {
        return Err(QuantumError::InvalidFiniteDifferenceStep(step));
    }
    circuit.validate_inputs(inputs)?;
    circuit.validate_parameters(parameters)?;
    if observable_qubit >= qubits {
        return Err(QuantumError::QubitOutOfRange {
            qubit: observable_qubit,
            qubits,
        });
    }
    let mut gradient = vec![0.0; circuit.parameter_count];
    for (index, operation) in circuit.operations.iter().enumerate() {
        if let ParameterizedOp::Rotation { parameter, .. } = operation {
            let plus = circuit.run_with_shift(inputs, parameters, qubits, Some((index, step)))?;
            let minus = circuit.run_with_shift(inputs, parameters, qubits, Some((index, -step)))?;
            let plus = plus.expectation(observable, observable_qubit)?;
            let minus = minus.expectation(observable, observable_qubit)?;
            gradient[*parameter] += (plus - minus) / (2.0 * step);
        }
    }
    Ok(gradient)
}

/// Reverse-mode gradient, written from the backprop definition rather than the
/// adjoint shortcut: carry the costate `O|psi>` backwards through the gate
/// list and accumulate `2 Re <lambda| (dU/dtheta) |psi_before>` at every
/// parameterized rotation.
///
/// For a Pauli observable this agrees with
/// [`ParameterizedCircuit::adjoint_gradient`] to round-off; the value of the
/// second implementation is that it is an independent derivation and a generic
/// path for gates whose adjoint shortcut is not hand-derived.
///
/// # Errors
///
/// Returns the usual validation errors and
/// [`QuantumError::OperationRequiresDensityMatrix`] for a circuit with a noise
/// channel.
pub fn backprop_gradient_with_inputs(
    circuit: &ParameterizedCircuit,
    inputs: &[f64],
    parameters: &[f64],
    qubits: usize,
    observable: SingleQubitObservable,
    observable_qubit: usize,
) -> Result<Vec<f64>, QuantumError> {
    circuit.validate_inputs(inputs)?;
    circuit.validate_parameters(parameters)?;
    if observable_qubit >= qubits {
        return Err(QuantumError::QubitOutOfRange {
            qubit: observable_qubit,
            qubits,
        });
    }

    // Forward pass: keep every intermediate state, which is what makes the
    // per-gate derivative available without re-running the circuit.
    let mut states = Vec::with_capacity(circuit.operations.len() + 1);
    states.push(StateVector::zero(qubits)?);
    for (index, operation) in circuit.operations.iter().enumerate() {
        let mut next = states[index].clone();
        match *operation {
            ParameterizedOp::Fixed { gate, target } => next.apply(gate, target)?,
            ParameterizedOp::Rotation {
                axis,
                target,
                parameter,
            } => {
                next.apply(rotation_gate(axis, parameters[parameter]), target)?;
            }
            ParameterizedOp::EncodingRotation {
                axis,
                target,
                input,
            } => {
                next.apply(rotation_gate(axis, inputs[input]), target)?;
            }
            ParameterizedOp::Cnot { control, target } => next.apply_cnot(control, target)?,
            ParameterizedOp::Cz { first, second } => next.apply_cz(first, second)?,
            ParameterizedOp::Swap { first, second } => next.apply_swap(first, second)?,
            ParameterizedOp::Channel { .. } => {
                return Err(QuantumError::OperationRequiresDensityMatrix);
            }
        }
        states.push(next);
    }

    // Costate: d<psi|O|psi>/d<psi| is O|psi> for Hermitian O.
    let pauli = match observable {
        SingleQubitObservable::X => crate::Gate::X,
        SingleQubitObservable::Y => crate::Gate::Y,
        SingleQubitObservable::Z => crate::Gate::Z,
    };
    let mut costate = states[states.len() - 1].clone();
    costate.apply(pauli, observable_qubit)?;

    let mut gradient = vec![0.0; circuit.parameter_count];
    let minus_i_half = crate::Complex64::new(0.0, -0.5);
    for (index, operation) in circuit.operations.iter().enumerate().rev() {
        let after = &states[index + 1];
        match *operation {
            ParameterizedOp::Rotation {
                axis,
                target,
                parameter,
            } => {
                // dU/dtheta = (-i/2) P U, so the contribution is
                // 2 Re <lambda| (-i/2) P |psi_after>.
                let mut derivative = after.clone();
                derivative.apply(pauli_generator(axis), target)?;
                let overlap_re = costate
                    .amplitudes
                    .iter()
                    .zip(&derivative.amplitudes)
                    .map(|(left, right)| (left.conj() * (*right * minus_i_half)).re)
                    .sum::<f64>();
                gradient[parameter] += 2.0 * overlap_re;
            }
            ParameterizedOp::Channel { .. } => {
                return Err(QuantumError::OperationRequiresDensityMatrix);
            }
            _ => {}
        }

        // Propagate the costate through this gate's adjoint.
        match *operation {
            ParameterizedOp::Fixed { gate, target } => {
                costate.apply(adjoint_gate(gate), target)?;
            }
            ParameterizedOp::Rotation {
                axis,
                target,
                parameter,
            } => {
                costate.apply(rotation_gate(axis, -parameters[parameter]), target)?;
            }
            ParameterizedOp::EncodingRotation {
                axis,
                target,
                input,
            } => {
                costate.apply(rotation_gate(axis, -inputs[input]), target)?;
            }
            ParameterizedOp::Cnot { control, target } => costate.apply_cnot(control, target)?,
            ParameterizedOp::Cz { first, second } => costate.apply_cz(first, second)?,
            ParameterizedOp::Swap { first, second } => costate.apply_swap(first, second)?,
            ParameterizedOp::Channel { .. } => {
                return Err(QuantumError::OperationRequiresDensityMatrix);
            }
        }
    }
    Ok(gradient)
}

/// The Pauli generator of a rotation axis.
fn pauli_generator(axis: RotationAxis) -> crate::Gate {
    match axis {
        RotationAxis::X => crate::Gate::X,
        RotationAxis::Y => crate::Gate::Y,
        RotationAxis::Z => crate::Gate::Z,
    }
}
