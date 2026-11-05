#![allow(missing_docs)]
//! Circuit optimization passes.
//!
//! Provides optimization passes for quantum circuits:
//! - Single-qubit gate fusion (rotation merging)
//! - Gate cancellation (inverse pairs)
//! - Commutation-based reordering
//! - Redundant gate removal

use crate::{Circuit, CircuitOp, Gate};
use std::collections::HashMap;

/// Optimization configuration.
#[derive(Clone, Debug, Default)]
pub struct OptimizeConfig {
    /// Fuse consecutive single-qubit rotations on the same qubit.
    pub fuse_rotations: bool,
    /// Cancel adjacent inverse gates (e.g., H·H, X·X, Rx(a)·Rx(-a)).
    pub cancel_inverses: bool,
    /// Remove identity gates.
    pub remove_identity: bool,
    /// Commute gates to enable more cancellations.
    pub commute_gates: bool,
}

impl OptimizeConfig {
    /// All optimizations enabled.
    pub fn all() -> Self {
        Self {
            fuse_rotations: true,
            cancel_inverses: true,
            remove_identity: true,
            commute_gates: true,
        }
    }

    /// Conservative: only remove identities and cancel exact inverses.
    pub fn conservative() -> Self {
        Self {
            fuse_rotations: false,
            cancel_inverses: true,
            remove_identity: true,
            commute_gates: false,
        }
    }
}

/// Optimization result with statistics.
#[derive(Clone, Debug, Default)]
pub struct OptimizeResult {
    pub original_ops: usize,
    pub optimized_ops: usize,
    pub original_two_qubit: usize,
    pub optimized_two_qubit: usize,
    pub original_depth: usize,
    pub optimized_depth: usize,
    pub fusions: usize,
    pub cancellations: usize,
    pub identities_removed: usize,
}

impl OptimizeResult {
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_ops == 0 {
            0.0
        } else {
            1.0 - (self.optimized_ops as f64 / self.original_ops as f64)
        }
    }
}

/// Count two-qubit gates in a circuit.
pub fn count_two_qubit(circuit: &Circuit) -> usize {
    circuit
        .operations()
        .iter()
        .filter(|op| {
            matches!(
                op,
                CircuitOp::Cnot { .. } | CircuitOp::Cz { .. } | CircuitOp::Swap { .. }
            )
        })
        .count()
}

/// Compute circuit depth (longest path through qubit dependencies).
pub fn circuit_depth(circuit: &Circuit) -> usize {
    let mut qubit_depth = HashMap::<usize, usize>::new();
    let mut classical_depth = HashMap::<usize, usize>::new();
    let mut max_depth = 0;
    for op in circuit.operations() {
        match op {
            CircuitOp::SingleQubit { target, .. } | CircuitOp::Channel { target, .. } => {
                let depth = qubit_depth.get(target).copied().unwrap_or(0) + 1;
                qubit_depth.insert(*target, depth);
                max_depth = max_depth.max(depth);
            }
            CircuitOp::Cnot { control, target }
            | CircuitOp::Cz {
                first: control,
                second: target,
            } => {
                let d = qubit_depth
                    .get(control)
                    .copied()
                    .unwrap_or(0)
                    .max(qubit_depth.get(target).copied().unwrap_or(0))
                    + 1;
                qubit_depth.insert(*control, d);
                qubit_depth.insert(*target, d);
                max_depth = max_depth.max(d);
            }
            CircuitOp::Swap { first, second } => {
                let d = qubit_depth
                    .get(first)
                    .copied()
                    .unwrap_or(0)
                    .max(qubit_depth.get(second).copied().unwrap_or(0))
                    + 1;
                qubit_depth.insert(*first, d);
                qubit_depth.insert(*second, d);
                max_depth = max_depth.max(d);
            }
            CircuitOp::Measure {
                target,
                classical_bit,
            } => {
                let d = qubit_depth.get(target).copied().unwrap_or(0) + 1;
                qubit_depth.insert(*target, d);
                classical_depth.insert(*classical_bit, d);
                max_depth = max_depth.max(d);
            }
            CircuitOp::ConditionalSingleQubit {
                classical_bit,
                target,
                ..
            } => {
                let d = qubit_depth
                    .get(target)
                    .copied()
                    .unwrap_or(0)
                    .max(classical_depth.get(classical_bit).copied().unwrap_or(0))
                    + 1;
                qubit_depth.insert(*target, d);
                max_depth = max_depth.max(d);
            }
        }
    }
    max_depth
}

/// Resource counts for comparing a circuit against a backend-specific cost model.
///
/// The counts are structural estimates, not device runtime predictions. In
/// particular, depth uses logical qubit/classical dependencies and does not
/// account for routing, native gate decomposition, or calibration data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CircuitCostProfile {
    pub single_qubit_gates: usize,
    pub cnot_gates: usize,
    pub cz_gates: usize,
    pub swap_gates: usize,
    pub t_gates: usize,
    pub tdg_gates: usize,
    pub measurements: usize,
    pub conditional_gates: usize,
    pub channels: usize,
    pub depth: usize,
    pub qubits: usize,
}

/// Hardware-aware resource estimate with optional routing and native-basis
/// adjustments.
#[derive(Clone, Debug, PartialEq)]
pub struct BackendCostProfile {
    pub profile: CircuitCostProfile,
    pub routing_swaps: usize,
    pub native_gate_adjustment: f64,
}

impl Default for BackendCostProfile {
    fn default() -> Self {
        Self {
            profile: CircuitCostProfile::default(),
            routing_swaps: 0,
            native_gate_adjustment: 0.0,
        }
    }
}

impl BackendCostProfile {
    /// Estimate logical cost after applying a simple hardware-aware correction.
    /// When a coupling map is supplied, non-adjacent CNOTs are charged with a
    /// SWAP path estimate; a native gate adjustment is added for any backend that
    /// makes some gates more expensive than the logical reference model.
    pub fn from_circuit(
        circuit: &Circuit,
        coupling: Option<&CouplingMap>,
        native_gate_adjustment: f64,
    ) -> Result<Self, CircuitCostModelError> {
        let profile = CircuitCostProfile::from_circuit(circuit);
        let mut routing_swaps = 0;
        if let Some(coupling) = coupling {
            let mut routed = circuit.clone();
            routing_swaps = route_to_hardware(&mut routed, coupling);
        }
        if !native_gate_adjustment.is_finite() || native_gate_adjustment < 0.0 {
            return Err(CircuitCostModelError::InvalidWeight {
                resource: "native_gate_adjustment",
                weight: native_gate_adjustment,
            });
        }
        Ok(Self {
            profile,
            routing_swaps,
            native_gate_adjustment,
        })
    }

    pub fn estimate_with_model(
        &self,
        model: &CircuitCostModel,
    ) -> Result<f64, CircuitCostModelError> {
        let mut profile = self.profile.clone();
        profile.swap_gates += self.routing_swaps;
        let base = model.estimate(&profile)?;
        Ok(base + self.native_gate_adjustment)
    }
}

impl CircuitCostProfile {
    /// Count logical resources in a circuit. T and Tdg are counted separately
    /// from other single-qubit gates so target models can price them differently.
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let mut profile = Self {
            depth: circuit_depth(circuit),
            qubits: circuit.num_qubits(),
            ..Self::default()
        };
        for operation in circuit.operations() {
            match operation {
                CircuitOp::SingleQubit { gate, .. } => match gate {
                    Gate::T => profile.t_gates += 1,
                    Gate::Tdg => profile.tdg_gates += 1,
                    _ => profile.single_qubit_gates += 1,
                },
                CircuitOp::Cnot { .. } => profile.cnot_gates += 1,
                CircuitOp::Cz { .. } => profile.cz_gates += 1,
                CircuitOp::Swap { .. } => profile.swap_gates += 1,
                CircuitOp::Measure { .. } => profile.measurements += 1,
                CircuitOp::ConditionalSingleQubit { .. } => profile.conditional_gates += 1,
                CircuitOp::Channel { .. } => profile.channels += 1,
            }
        }
        profile
    }

    pub fn total_gates(&self) -> usize {
        self.single_qubit_gates
            + self.cnot_gates
            + self.cz_gates
            + self.swap_gates
            + self.t_gates
            + self.tdg_gates
            + self.measurements
            + self.conditional_gates
            + self.channels
    }
}

/// Weights for a transparent, target-specific logical resource estimate.
///
/// This is a comparison aid, not a physical hardware cost model. Set weights
/// from the target backend's documented costs; the default prices each logical
/// gate equally and ignores depth and qubit footprint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CircuitCostModel {
    pub single_qubit_gate: f64,
    pub cnot_gate: f64,
    pub cz_gate: f64,
    pub swap_gate: f64,
    pub t_gate: f64,
    pub tdg_gate: f64,
    pub measurement: f64,
    pub conditional_gate: f64,
    pub channel: f64,
    pub depth: f64,
    pub qubit: f64,
}

impl Default for CircuitCostModel {
    fn default() -> Self {
        Self {
            single_qubit_gate: 1.0,
            cnot_gate: 1.0,
            cz_gate: 1.0,
            swap_gate: 1.0,
            t_gate: 1.0,
            tdg_gate: 1.0,
            measurement: 1.0,
            conditional_gate: 1.0,
            channel: 1.0,
            depth: 0.0,
            qubit: 0.0,
        }
    }
}

impl CircuitCostModel {
    /// Estimate a relative cost from a structural profile.
    ///
    /// T/Tdg, depth, and qubit count are explicit terms so users can emphasize
    /// fault-tolerant or hardware-specific resources. Weights are not
    /// normalized; only comparisons made with the same model are meaningful.
    /// Negative or non-finite weights and overflowing estimates are rejected.
    pub fn estimate(&self, profile: &CircuitCostProfile) -> Result<f64, CircuitCostModelError> {
        let weights = [
            ("single_qubit_gate", self.single_qubit_gate),
            ("cnot_gate", self.cnot_gate),
            ("cz_gate", self.cz_gate),
            ("swap_gate", self.swap_gate),
            ("t_gate", self.t_gate),
            ("tdg_gate", self.tdg_gate),
            ("measurement", self.measurement),
            ("conditional_gate", self.conditional_gate),
            ("channel", self.channel),
            ("depth", self.depth),
            ("qubit", self.qubit),
        ];
        for (resource, weight) in weights {
            if !weight.is_finite() || weight < 0.0 {
                return Err(CircuitCostModelError::InvalidWeight { resource, weight });
            }
        }

        let estimate = profile.single_qubit_gates as f64 * self.single_qubit_gate
            + profile.cnot_gates as f64 * self.cnot_gate
            + profile.cz_gates as f64 * self.cz_gate
            + profile.swap_gates as f64 * self.swap_gate
            + profile.t_gates as f64 * self.t_gate
            + profile.tdg_gates as f64 * self.tdg_gate
            + profile.measurements as f64 * self.measurement
            + profile.conditional_gates as f64 * self.conditional_gate
            + profile.channels as f64 * self.channel
            + profile.depth as f64 * self.depth
            + profile.qubits as f64 * self.qubit;
        if !estimate.is_finite() {
            return Err(CircuitCostModelError::NonFiniteEstimate);
        }
        Ok(estimate)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CircuitCostModelError {
    InvalidWeight { resource: &'static str, weight: f64 },
    NonFiniteEstimate,
}

impl std::fmt::Display for CircuitCostModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWeight { resource, weight } => {
                write!(
                    f,
                    "cost weight for {resource} must be finite and nonnegative, got {weight}"
                )
            }
            Self::NonFiniteEstimate => write!(f, "circuit cost estimate is not finite"),
        }
    }
}

impl std::error::Error for CircuitCostModelError {}

/// Outcome of comparing an optimized candidate against the original circuit.
#[derive(Clone, Debug)]
pub struct CostAwareOptimizeResult {
    pub original_profile: CircuitCostProfile,
    pub candidate_profile: CircuitCostProfile,
    pub original_cost: f64,
    pub candidate_cost: f64,
    pub applied: bool,
    pub optimization: OptimizeResult,
}

/// Optimize a copy and keep the rewrite only when it strictly lowers modeled cost.
///
/// The function never mutates the input. If the candidate ties or costs more,
/// the returned circuit is unchanged and `applied` is false.
pub fn optimize_for_cost_model(
    circuit: &Circuit,
    config: &OptimizeConfig,
    model: &CircuitCostModel,
) -> Result<(Circuit, CostAwareOptimizeResult), CircuitCostModelError> {
    let original_profile = CircuitCostProfile::from_circuit(circuit);
    let original_cost = model.estimate(&original_profile)?;

    let mut candidate = circuit.clone();
    let optimization = optimize_circuit(&mut candidate, config);
    let candidate_profile = CircuitCostProfile::from_circuit(&candidate);
    let candidate_cost = model.estimate(&candidate_profile)?;
    let applied = candidate_cost < original_cost;
    let selected = if applied { candidate } else { circuit.clone() };

    Ok((
        selected,
        CostAwareOptimizeResult {
            original_profile,
            candidate_profile,
            original_cost,
            candidate_cost,
            applied,
            optimization,
        },
    ))
}

/// Optimize a circuit in-place, returning statistics.
pub fn optimize_circuit(circuit: &mut Circuit, config: &OptimizeConfig) -> OptimizeResult {
    let orig_2q = count_two_qubit(circuit);
    let orig_depth = circuit_depth(circuit);
    let mut result = OptimizeResult {
        original_ops: circuit.operations().len(),
        original_two_qubit: orig_2q,
        original_depth: orig_depth,
        ..Default::default()
    };

    if config.remove_identity {
        result.identities_removed = remove_identity_gates(circuit);
    }

    if config.cancel_inverses {
        result.cancellations = cancel_inverse_pairs(circuit);
    }

    if config.fuse_rotations {
        result.fusions = fuse_rotations(circuit);
        if config.remove_identity {
            result.identities_removed += remove_identity_gates(circuit);
        }
    }

    if config.commute_gates {
        commute_for_cancellation(circuit);
        // Re-run cancellation after commutation
        result.cancellations += cancel_inverse_pairs(circuit);
    }

    result.optimized_ops = circuit.operations().len();
    result.optimized_two_qubit = count_two_qubit(circuit);
    result.optimized_depth = circuit_depth(circuit);
    result
}

/// Remove identity gates (I, Rx(0), Ry(0), Rz(0), Rz(2π), etc.).
fn remove_identity_gates(circuit: &mut Circuit) -> usize {
    let ops = circuit.operations_mut();
    let original_len = ops.len();
    ops.retain(|op| !is_identity_op(op));
    original_len - ops.len()
}

fn is_identity_op(op: &CircuitOp) -> bool {
    match op {
        CircuitOp::SingleQubit { gate, .. } => match gate {
            Gate::I => true,
            Gate::Rx(angle) | Gate::Ry(angle) | Gate::Rz(angle) => {
                normalize_angle(*angle).abs() < 1e-10
            }
            _ => false,
        },
        _ => false,
    }
}

/// Cancel adjacent inverse gate pairs on the same qubit.
/// Examples: H·H, X·X, Y·Y, Z·Z, S·S†, T·T†, Rx(a)·Rx(-a), etc.
fn cancel_inverse_pairs(circuit: &mut Circuit) -> usize {
    let ops = circuit.operations();
    let mut reduced = Vec::with_capacity(ops.len());
    let mut cancellations = 0;
    for op in ops.iter().copied() {
        if reduced
            .last()
            .is_some_and(|previous| can_cancel_pair(previous, &op))
        {
            reduced.pop();
            cancellations += 1;
        } else {
            reduced.push(op);
        }
    }
    *circuit.operations_mut() = reduced;
    cancellations
}

/// Check if two adjacent operations are cancellable inverse pairs.
fn can_cancel_pair(op1: &CircuitOp, op2: &CircuitOp) -> bool {
    if matches!(
        (op1, op2),
        (CircuitOp::Cnot { control: c1, target: t1 }, CircuitOp::Cnot { control: c2, target: t2 })
            if c1 == c2 && t1 == t2
    ) || matches!(
        (op1, op2),
        (CircuitOp::Cz { first: f1, second: s1 }, CircuitOp::Cz { first: f2, second: s2 })
            if (f1 == f2 && s1 == s2) || (f1 == s2 && s1 == f2)
    ) || matches!(
        (op1, op2),
        (CircuitOp::Swap { first: f1, second: s1 }, CircuitOp::Swap { first: f2, second: s2 })
            if (f1 == f2 && s1 == s2) || (f1 == s2 && s1 == f2)
    ) {
        return true;
    }

    if let (
        CircuitOp::SingleQubit {
            gate: g1,
            target: t1,
        },
        CircuitOp::SingleQubit {
            gate: g2,
            target: t2,
        },
    ) = (op1, op2)
    {
        t1 == t2 && are_inverses(*g1, *g2)
    } else {
        false
    }
}

/// Check if two gates are inverses of each other.
fn are_inverses(g1: Gate, g2: Gate) -> bool {
    use Gate::*;
    match (g1, g2) {
        // Self-inverse gates
        (H, H) | (X, X) | (Y, Y) | (Z, Z) | (S, Sdg) | (Sdg, S) | (T, Tdg) | (Tdg, T) => true,
        // Rotation inverses: Rx(a) · Rx(-a) = I
        (Rx(a), Rx(b))
            if (a + b).abs() < 1e-10
                || (a + b - 2.0 * std::f64::consts::PI).abs() < 1e-10
                || (a + b + 2.0 * std::f64::consts::PI).abs() < 1e-10 =>
        {
            true
        }
        (Ry(a), Ry(b))
            if (a + b).abs() < 1e-10
                || (a + b - 2.0 * std::f64::consts::PI).abs() < 1e-10
                || (a + b + 2.0 * std::f64::consts::PI).abs() < 1e-10 =>
        {
            true
        }
        (Rz(a), Rz(b))
            if (a + b).abs() < 1e-10
                || (a + b - 2.0 * std::f64::consts::PI).abs() < 1e-10
                || (a + b + 2.0 * std::f64::consts::PI).abs() < 1e-10 =>
        {
            true
        }
        _ => false,
    }
}

/// Fuse consecutive single-qubit rotations on the same qubit.
/// Rx(a)·Rx(b) = Rx(a+b), same for Ry, Rz.
/// Also fuses Rz into adjacent rotations where possible.
fn fuse_rotations(circuit: &mut Circuit) -> usize {
    let ops = circuit.operations();
    let mut fused_ops = Vec::with_capacity(ops.len());
    let mut fusions = 0;
    for op in ops.iter().copied() {
        if let (
            Some(CircuitOp::SingleQubit {
                gate: previous_gate,
                target: previous_target,
            }),
            CircuitOp::SingleQubit { gate, target },
        ) = (fused_ops.last_mut(), op)
            && *previous_target == target
            && let Some(fused) = try_fuse_gates(*previous_gate, gate)
        {
            *previous_gate = fused;
            fusions += 1;
        } else {
            fused_ops.push(op);
        }
    }
    *circuit.operations_mut() = fused_ops;
    fusions
}

fn try_fuse_gates(g1: Gate, g2: Gate) -> Option<Gate> {
    use Gate::*;
    match (g1, g2) {
        // Pauli rotations that combine to identity (sum ≈ 0)
        (Rx(a), Rx(b)) if (a + b).abs() < 1e-10 => Some(I),
        (Ry(a), Ry(b)) if (a + b).abs() < 1e-10 => Some(I),
        (Rz(a), Rz(b)) if (a + b).abs() < 1e-10 => Some(I),

        // Same-axis rotations add angles
        (Rx(a), Rx(b)) => Some(Rx(normalize_angle(a + b))),
        (Ry(a), Ry(b)) => Some(Ry(normalize_angle(a + b))),
        (Rz(a), Rz(b)) => Some(Rz(normalize_angle(a + b))),

        // H = Rz(π/2) · Rx(π/2) · Rz(π/2) - not easily fused
        // S = Rz(π/2), S† = Rz(-π/2)
        (S, S) => Some(Rz(std::f64::consts::PI)),
        (Sdg, Sdg) => Some(Rz(-std::f64::consts::PI)),
        (T, T) => Some(S),
        (Tdg, Tdg) => Some(Sdg),

        _ => None,
    }
}

fn normalize_angle(angle: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut a = angle % two_pi;
    if a < -std::f64::consts::PI {
        a += two_pi;
    } else if a > std::f64::consts::PI {
        a -= two_pi;
    }
    a
}

/// Commute gates to bring cancellable pairs together.
/// Single-qubit gates on different qubits commute.
/// Single-qubit gates commute through CNOT if they act on the other qubit.
fn commute_for_cancellation(circuit: &mut Circuit) {
    let ops = circuit.operations_mut();
    let mut changed = true;
    let max_passes = 5;

    for _ in 0..max_passes {
        if !changed {
            break;
        }
        changed = false;
        let mut i = 0;
        while i + 1 < ops.len() {
            if can_commute(&ops[i], &ops[i + 1]) {
                ops.swap(i, i + 1);
                changed = true;
                // After swap, check if we can cancel with the previous gate
                if i > 0 {
                    i -= 1;
                    continue;
                }
            }
            i += 1;
        }
    }
}

fn can_commute(op1: &CircuitOp, op2: &CircuitOp) -> bool {
    use CircuitOp::*;

    // Single-qubit on different qubits always commute
    if let (SingleQubit { target: t1, .. }, SingleQubit { target: t2, .. }) = (op1, op2) {
        return t1 != t2;
    }

    // Single-qubit commutes with CNOT if on different qubits
    if let (
        SingleQubit { target: t1, .. },
        Cnot {
            control: c,
            target: t2,
        },
    ) = (op1, op2)
    {
        return t1 != c && t1 != t2;
    }
    if let (
        Cnot {
            control: c,
            target: t2,
        },
        SingleQubit { target: t1, .. },
    ) = (op1, op2)
    {
        return t1 != c && t1 != t2;
    }

    // Single-qubit commutes with CZ if on different qubits
    if let (
        SingleQubit { target: t1, .. },
        Cz {
            first: f,
            second: s,
        },
    ) = (op1, op2)
    {
        return t1 != f && t1 != s;
    }
    if let (
        Cz {
            first: f,
            second: s,
        },
        SingleQubit { target: t1, .. },
    ) = (op1, op2)
    {
        return t1 != f && t1 != s;
    }

    // Single-qubit commutes with Swap if on different qubits
    if let (
        SingleQubit { target: t1, .. },
        Swap {
            first: f,
            second: s,
        },
    ) = (op1, op2)
    {
        return t1 != f && t1 != s;
    }
    if let (
        Swap {
            first: f,
            second: s,
        },
        SingleQubit { target: t1, .. },
    ) = (op1, op2)
    {
        return t1 != f && t1 != s;
    }

    // Two-qubit gates on disjoint qubits commute
    if let (
        Cnot {
            control: c1,
            target: t1,
        },
        Cnot {
            control: c2,
            target: t2,
        },
    ) = (op1, op2)
    {
        let q1 = [*c1, *t1];
        let q2 = [*c2, *t2];
        return q1.iter().all(|&q| !q2.contains(&q));
    }
    if let (
        Cz {
            first: f1,
            second: s1,
        },
        Cz {
            first: f2,
            second: s2,
        },
    ) = (op1, op2)
    {
        let q1 = [*f1, *s1];
        let q2 = [*f2, *s2];
        return q1.iter().all(|&q| !q2.contains(&q));
    }

    false
}

/// Convenience function for the REPL.
pub fn optimize_circuit_default(circuit: &mut Circuit) -> OptimizeResult {
    optimize_circuit(circuit, &OptimizeConfig::all())
}

impl Circuit {
    /// Return an optimized copy and the optimization statistics.
    ///
    /// Optimize once and reuse the returned circuit when executing it many
    /// times; optimization has its own cost and should not be repeated for
    /// every shot or simulation.
    pub fn optimized(&self) -> (Self, OptimizeResult) {
        let mut circuit = self.clone();
        let result = optimize_circuit_default(&mut circuit);
        (circuit, result)
    }

    /// Optimize a copy of this circuit and execute it on `state`.
    ///
    /// For repeated execution, call [`Self::optimized`] once and reuse the
    /// returned circuit instead of paying the optimization cost each time.
    pub fn run_optimized(
        &self,
        state: crate::StateVector,
    ) -> Result<(crate::StateVector, OptimizeResult), crate::QuantumError> {
        let (circuit, result) = self.optimized();
        Ok((circuit.run(state)?, result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Circuit, Gate, StateVector};

    #[test]
    fn test_identity_removal() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::I, 0);
        circuit.push(Gate::X, 0);

        let result = optimize_circuit(&mut circuit, &OptimizeConfig::conservative());
        assert_eq!(result.identities_removed, 1);
        assert_eq!(circuit.operations().len(), 2);
    }

    #[test]
    fn test_inverse_cancellation() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::H, 0);
        circuit.push(Gate::X, 0);
        circuit.push(Gate::X, 0);

        let result = optimize_circuit(&mut circuit, &OptimizeConfig::conservative());
        assert_eq!(result.cancellations, 2);
        assert_eq!(circuit.operations().len(), 0);
    }

    #[test]
    fn test_rotation_fusion() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::Rx(0.5), 0);
        circuit.push(Gate::Rx(1.5), 0);
        circuit.push(Gate::Rz(0.3), 1);
        circuit.push(Gate::Rz(0.7), 1);

        let result = optimize_circuit(&mut circuit, &OptimizeConfig::all());
        assert_eq!(result.fusions, 2);
        assert_eq!(circuit.operations().len(), 2);

        // Check angles added correctly
        if let crate::CircuitOp::SingleQubit {
            gate: Gate::Rx(a), ..
        } = &circuit.operations()[0]
        {
            assert!((a - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn removes_zero_rotations_and_rotations_fused_to_identity() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::Rx(0.0), 0);
        circuit.push(Gate::Rz(0.25), 1);
        circuit.push(Gate::Rz(-0.25), 1);

        let config = OptimizeConfig {
            fuse_rotations: true,
            cancel_inverses: false,
            remove_identity: true,
            commute_gates: false,
        };
        let result = optimize_circuit(&mut circuit, &config);
        assert_eq!(result.identities_removed, 2);
        assert_eq!(result.cancellations, 0);
        assert!(circuit.operations().is_empty());
    }

    #[test]
    fn test_s_self_inverse() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::S, 0);
        circuit.push(Gate::Sdg, 0);

        let result = optimize_circuit(&mut circuit, &OptimizeConfig::all());
        assert_eq!(result.cancellations, 1);
        assert_eq!(circuit.operations().len(), 0);
    }

    #[test]
    fn test_end_to_end_optimization() {
        use crate::StateVector;
        // Circuit: H·H·X·X = I (should optimize to empty)
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::H, 0);
        circuit.push(Gate::X, 0);
        circuit.push(Gate::X, 0);

        let initial = StateVector::zero(1).unwrap();
        let before = circuit.run(initial.clone()).unwrap();

        optimize_circuit(&mut circuit, &OptimizeConfig::all());

        let after = circuit.run(initial).unwrap();
        // Both should produce |0⟩
        assert!((before.amplitudes()[0].re - 1.0).abs() < 1e-10);
        assert!((after.amplitudes()[0].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cancellation_pass_handles_long_inverse_sequences() {
        let mut circuit = Circuit::new();
        for _ in 0..2_000 {
            circuit.push(Gate::H, 0);
            circuit.push(Gate::H, 0);
        }

        let result = optimize_circuit(&mut circuit, &OptimizeConfig::conservative());
        assert_eq!(result.cancellations, 2_000);
        assert!(circuit.operations().is_empty());
    }

    #[test]
    fn depth_tracks_large_qubit_indices_and_measurement_feed_forward() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 300);
        circuit.measure(0, 7);
        circuit.if_classical_bit(7, 1, Gate::X, 2);

        assert_eq!(circuit_depth(&circuit), 2);
    }

    #[test]
    fn self_optimize_reports_original_costs() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::H, 0);
        circuit.cnot(0, 1);

        let result = self_optimize(&mut circuit);
        assert_eq!(result.original_ops, 3);
        assert_eq!(result.optimized_ops, 1);
        assert_eq!(result.original_two_qubit, 1);
        assert_eq!(result.optimized_two_qubit, 1);
        assert_eq!(result.original_depth, 3);
        assert_eq!(result.optimized_depth, 1);
    }

    #[test]
    fn cancels_adjacent_self_inverse_two_qubit_gates() {
        let mut circuit = Circuit::new();
        circuit.cnot(0, 1);
        circuit.cnot(0, 1);
        circuit.cz(0, 1);
        circuit.cz(0, 1);
        circuit.swap(0, 1);
        circuit.swap(0, 1);

        let result = optimize_circuit(&mut circuit, &OptimizeConfig::conservative());
        assert_eq!(result.cancellations, 3);
        assert!(circuit.operations().is_empty());
    }

    #[test]
    fn t_gate_minimization_reports_removed_t_gates() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::T, 0);
        circuit.push(Gate::T, 0);
        circuit.push(Gate::Tdg, 1);
        circuit.push(Gate::Tdg, 1);

        assert_eq!(minimize_t_gates(&mut circuit), 4);
        assert_eq!(
            circuit.operations(),
            &[
                CircuitOp::SingleQubit {
                    gate: Gate::S,
                    target: 0,
                },
                CircuitOp::SingleQubit {
                    gate: Gate::Sdg,
                    target: 1,
                },
            ]
        );
    }

    #[test]
    fn optimized_execution_preserves_state_and_reports_reduction() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::H, 0);
        circuit.push(Gate::X, 1);
        let initial = StateVector::zero(2).unwrap();
        let expected = circuit.run(initial.clone()).unwrap();

        let (actual, result) = circuit.run_optimized(initial).unwrap();
        assert_eq!(result.original_ops, 3);
        assert_eq!(result.optimized_ops, 1);
        for (expected, actual) in expected.amplitudes().iter().zip(actual.amplitudes()) {
            assert!((expected.re - actual.re).abs() < 1e-10);
            assert!((expected.im - actual.im).abs() < 1e-10);
        }
    }

    #[test]
    fn cost_profile_counts_logical_resources_and_depth() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.push(Gate::T, 1);
        circuit.push(Gate::Tdg, 1);
        circuit.cnot(0, 1);
        circuit.measure(1, 0);
        circuit.if_classical_bit(0, 1, Gate::X, 0);

        let profile = CircuitCostProfile::from_circuit(&circuit);
        assert_eq!(profile.single_qubit_gates, 1);
        assert_eq!(profile.t_gates, 1);
        assert_eq!(profile.tdg_gates, 1);
        assert_eq!(profile.cnot_gates, 1);
        assert_eq!(profile.measurements, 1);
        assert_eq!(profile.conditional_gates, 1);
        assert_eq!(profile.channels, 0);
        assert_eq!(profile.qubits, 2);
        assert_eq!(profile.depth, 5);
        assert_eq!(profile.total_gates(), 6);
    }

    #[test]
    fn cost_model_can_emphasize_two_qubit_gates_and_depth() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::H, 0);
        circuit.cnot(0, 1);
        circuit.cz(1, 2);
        let profile = CircuitCostProfile::from_circuit(&circuit);
        let model = CircuitCostModel {
            single_qubit_gate: 0.0,
            cnot_gate: 10.0,
            cz_gate: 8.0,
            depth: 2.0,
            qubit: 3.0,
            ..CircuitCostModel::default()
        };

        assert_eq!(
            model.estimate(&profile).unwrap(),
            10.0 + 8.0 + 2.0 * 3.0 + 3.0 * 3.0
        );
    }

    #[test]
    fn backend_cost_profile_counts_routing_swaps_once() {
        let mut circuit = Circuit::new();
        circuit.cnot(0, 2);
        let coupling = CouplingMap::linear(3);
        let model = CircuitCostModel {
            cnot_gate: 0.0,
            swap_gate: 3.0,
            ..CircuitCostModel::default()
        };

        let profile = BackendCostProfile::from_circuit(&circuit, Some(&coupling), 0.0).unwrap();
        assert_eq!(profile.routing_swaps, 2);
        assert_eq!(profile.profile.swap_gates, 0);
        assert_eq!(profile.estimate_with_model(&model).unwrap(), 6.0);
    }

    #[test]
    fn optimization_cost_profiles_show_t_count_reduction() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::T, 0);
        circuit.push(Gate::T, 0);
        circuit.push(Gate::Tdg, 1);
        circuit.push(Gate::Tdg, 1);
        let fault_tolerant = CircuitCostModel {
            single_qubit_gate: 0.0,
            t_gate: 1.0,
            tdg_gate: 1.0,
            ..CircuitCostModel::default()
        };

        let before = fault_tolerant
            .estimate(&CircuitCostProfile::from_circuit(&circuit))
            .unwrap();
        optimize_circuit(&mut circuit, &OptimizeConfig::all());
        let after = fault_tolerant
            .estimate(&CircuitCostProfile::from_circuit(&circuit))
            .unwrap();

        assert_eq!(before, 4.0);
        assert_eq!(after, 0.0);
    }

    #[test]
    fn cost_aware_optimization_applies_only_strict_improvements() {
        let mut circuit = Circuit::new();
        circuit.push(Gate::T, 0);
        circuit.push(Gate::T, 0);
        let model = CircuitCostModel::default();
        let (optimized, decision) =
            optimize_for_cost_model(&circuit, &OptimizeConfig::all(), &model).unwrap();

        assert!(decision.applied);
        assert_eq!(decision.original_cost, 2.0);
        assert_eq!(decision.candidate_cost, 1.0);
        assert_eq!(optimized.operations().len(), 1);
        assert_eq!(circuit.operations().len(), 2);

        let t_weighted = CircuitCostModel {
            single_qubit_gate: 100.0,
            t_gate: 1.0,
            tdg_gate: 1.0,
            ..CircuitCostModel::default()
        };
        let (unchanged, decision) =
            optimize_for_cost_model(&circuit, &OptimizeConfig::all(), &t_weighted).unwrap();
        assert!(!decision.applied);
        assert_eq!(decision.candidate_cost, 100.0);
        assert_eq!(unchanged.operations(), circuit.operations());
    }

    #[test]
    fn cost_model_rejects_negative_nonfinite_and_overflowing_costs() {
        let profile = CircuitCostProfile {
            single_qubit_gates: 2,
            ..CircuitCostProfile::default()
        };
        let negative = CircuitCostModel {
            cnot_gate: -1.0,
            ..CircuitCostModel::default()
        };
        assert!(matches!(
            negative.estimate(&profile),
            Err(CircuitCostModelError::InvalidWeight { .. })
        ));

        let non_finite = CircuitCostModel {
            depth: f64::NAN,
            ..CircuitCostModel::default()
        };
        assert!(matches!(
            non_finite.estimate(&profile),
            Err(CircuitCostModelError::InvalidWeight { .. })
        ));

        let overflowing = CircuitCostModel {
            single_qubit_gate: f64::MAX,
            ..CircuitCostModel::default()
        };
        assert_eq!(
            overflowing.estimate(&profile),
            Err(CircuitCostModelError::NonFiniteEstimate)
        );
    }
}

/// T-gate statistics for a circuit.
#[derive(Clone, Debug, Default)]
pub struct TGateStats {
    pub t_count: usize,
    pub tdg_count: usize,
    pub total_clifford: usize,
}

/// Count T gates and Clifford gates in a circuit.
pub fn t_gate_count(circuit: &Circuit) -> TGateStats {
    let mut stats = TGateStats::default();
    for op in circuit.operations() {
        match op {
            CircuitOp::SingleQubit { gate, .. } => match gate {
                Gate::T => stats.t_count += 1,
                Gate::Tdg => stats.tdg_count += 1,
                Gate::I | Gate::X | Gate::Y | Gate::Z | Gate::H | Gate::S | Gate::Sdg => {
                    stats.total_clifford += 1;
                }
                Gate::Rx(_) | Gate::Ry(_) | Gate::Rz(_) => {}
            },
            CircuitOp::Cnot { .. } | CircuitOp::Cz { .. } | CircuitOp::Swap { .. } => {
                stats.total_clifford += 1;
            }
            _ => {}
        }
    }
    stats
}

/// Reduce adjacent T/Tdg pairs and fuse neighboring rotations.
///
/// Returns the number of T/Tdg gates removed. This is a local peephole pass,
/// not a general Clifford+T synthesis algorithm.
pub fn minimize_t_gates(circuit: &mut Circuit) -> usize {
    let before = t_gate_count(circuit);
    fuse_rotations(circuit);
    remove_identity_gates(circuit);
    cancel_inverse_pairs(circuit);
    let after = t_gate_count(circuit);
    (before.t_count + before.tdg_count).saturating_sub(after.t_count + after.tdg_count)
}

/// Hardware coupling map: list of (physical_qubit_a, physical_qubit_b) edges.
#[derive(Clone, Debug)]
pub struct CouplingMap {
    pub num_qubits: usize,
    pub edges: Vec<(usize, usize)>,
}

impl CouplingMap {
    /// Linear chain topology.
    pub fn linear(n: usize) -> Self {
        let edges = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        Self {
            num_qubits: n,
            edges,
        }
    }
    /// Grid topology (rows x cols).
    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                if c + 1 < cols {
                    edges.push((idx, idx + 1));
                }
                if r + 1 < rows {
                    edges.push((idx, idx + cols));
                }
            }
        }
        Self {
            num_qubits: n,
            edges,
        }
    }
    pub fn are_connected(&self, a: usize, b: usize) -> bool {
        self.edges
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
    }
}

/// Route a circuit to a hardware topology by inserting SWAP gates.
/// Returns the number of SWAPs inserted.
/// BFS shortest path between two qubits on the coupling map.
fn bfs_path(coupling: &CouplingMap, start: usize, end: usize) -> Vec<usize> {
    if start == end {
        return vec![start];
    }
    let mut visited = vec![false; coupling.num_qubits];
    let mut parent = vec![usize::MAX; coupling.num_qubits];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    visited[start] = true;
    while let Some(cur) = queue.pop_front() {
        if cur == end {
            break;
        }
        let neighbors: Vec<usize> = coupling
            .edges
            .iter()
            .filter_map(|&(a, b)| {
                if a == cur && !visited[b] {
                    Some(b)
                } else if b == cur && !visited[a] {
                    Some(a)
                } else {
                    None
                }
            })
            .collect();
        for nb in neighbors {
            visited[nb] = true;
            parent[nb] = cur;
            queue.push_back(nb);
        }
    }
    if !visited[end] {
        return vec![];
    }
    // Reconstruct path
    let mut path = vec![end];
    let mut cur = end;
    while cur != start {
        cur = parent[cur];
        path.push(cur);
    }
    path.reverse();
    path
}

pub fn route_to_hardware(circuit: &mut Circuit, coupling: &CouplingMap) -> usize {
    let mut swaps = 0;
    let ops = circuit.operations().to_vec();
    let mut new_ops: Vec<CircuitOp> = Vec::new();
    for op in &ops {
        match op {
            CircuitOp::Cnot { control, target }
            | CircuitOp::Cz {
                first: control,
                second: target,
            } => {
                if !coupling.are_connected(*control, *target) {
                    let path = bfs_path(coupling, *control, *target);
                    // Insert SWAPs along the path to bring qubits together
                    for i in 0..path.len().saturating_sub(1) {
                        new_ops.push(CircuitOp::Swap {
                            first: path[i],
                            second: path[i + 1],
                        });
                        swaps += 1;
                    }
                }
                new_ops.push(*op);
            }
            CircuitOp::Swap { first, second } => {
                if !coupling.are_connected(*first, *second) {
                    let path = bfs_path(coupling, *first, *second);
                    for i in 0..path.len().saturating_sub(1) {
                        new_ops.push(CircuitOp::Swap {
                            first: path[i],
                            second: path[i + 1],
                        });
                        swaps += 1;
                    }
                } else {
                    new_ops.push(*op);
                }
            }
            _ => new_ops.push(*op),
        }
    }
    *circuit.operations_mut() = new_ops;
    swaps
}

/// Self-optimizing circuit: repeatedly applies optimization passes until
/// no further reduction is achieved.
pub fn self_optimize(circuit: &mut Circuit) -> OptimizeResult {
    let original = circuit.operations().len();
    let original_two_qubit = count_two_qubit(circuit);
    let original_depth = circuit_depth(circuit);
    let mut total_fusions = 0;
    let mut total_cancellations = 0;
    let mut total_identities = 0;
    let mut iterations = 0;

    loop {
        let before = circuit.operations().to_vec();
        let result = optimize_circuit(circuit, &OptimizeConfig::all());
        total_fusions += result.fusions;
        total_cancellations += result.cancellations;
        total_identities += result.identities_removed;
        iterations += 1;

        if circuit.operations() == before || iterations >= 10 {
            break;
        }
    }

    OptimizeResult {
        original_ops: original,
        optimized_ops: circuit.operations().len(),
        original_two_qubit,
        optimized_two_qubit: count_two_qubit(circuit),
        original_depth,
        optimized_depth: circuit_depth(circuit),
        fusions: total_fusions,
        cancellations: total_cancellations,
        identities_removed: total_identities,
    }
}

/// Check if two circuits are equivalent by comparing state vectors.
/// Global phase is allowed.
pub fn circuits_equivalent(c1: &Circuit, c2: &Circuit, num_qubits: usize) -> bool {
    use crate::StateVector;
    let mut sv1 = StateVector::zero(num_qubits).unwrap();
    let mut sv2 = StateVector::zero(num_qubits).unwrap();

    // Apply all gates from c1
    for op in c1.operations() {
        apply_op(&mut sv1, op);
    }
    for op in c2.operations() {
        apply_op(&mut sv2, op);
    }

    // Compare with global phase tolerance
    let amps1 = sv1.amplitudes();
    let amps2 = sv2.amplitudes();
    if amps1.len() != amps2.len() {
        return false;
    }

    // Find global phase from first non-zero amplitude
    let mut phase = None;
    for (a, b) in amps1.iter().zip(amps2.iter()) {
        if a.norm_sqr() > 1e-12 {
            let p = b.re * a.re + b.im * a.im; // cos(phase)
            let q = b.im * a.re - b.re * a.im; // sin(phase)
            let norm = (p * p + q * q).sqrt();
            if norm < 1e-12 {
                return false;
            }
            phase = Some((p / norm, q / norm));
            break;
        }
    }
    let (pc, ps) = phase.unwrap_or((1.0, 0.0));

    for (a, b) in amps1.iter().zip(amps2.iter()) {
        // b should equal a * (pc + i*ps)
        let expected_re = a.re * pc - a.im * ps;
        let expected_im = a.re * ps + a.im * pc;
        if (b.re - expected_re).abs() > 1e-9 || (b.im - expected_im).abs() > 1e-9 {
            return false;
        }
    }
    true
}

fn apply_op(sv: &mut crate::StateVector, op: &CircuitOp) {
    match op {
        CircuitOp::SingleQubit { gate, target } => {
            sv.apply(*gate, *target).unwrap();
        }
        CircuitOp::Cnot { control, target } => {
            sv.apply_cnot(*control, *target).unwrap();
        }
        CircuitOp::Cz { first, second } => {
            sv.apply_cz(*first, *second).unwrap();
        }
        _ => {}
    }
}
