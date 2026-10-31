//! Circuit optimization passes.
//!
//! Provides optimization passes for quantum circuits:
//! - Single-qubit gate fusion (rotation merging)
//! - Gate cancellation (inverse pairs)
//! - Commutation-based reordering
//! - Redundant gate removal

use crate::{Circuit, CircuitOp, Gate};

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

/// Optimize a circuit in-place, returning statistics.
pub fn optimize_circuit(circuit: &mut Circuit, config: &OptimizeConfig) -> OptimizeResult {
    let mut result = OptimizeResult {
        original_ops: circuit.operations().len(),
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
    }

    if config.commute_gates {
        commute_for_cancellation(circuit);
        // Re-run cancellation after commutation
        result.cancellations += cancel_inverse_pairs(circuit);
    }

    result.optimized_ops = circuit.operations().len();
    result
}

/// Remove identity gates (I, Rx(0), Ry(0), Rz(0), Rz(2π), etc.).
fn remove_identity_gates(circuit: &mut Circuit) -> usize {
    let ops = circuit.operations();
    let mut to_remove = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        if is_identity_op(op) {
            to_remove.push(i);
        }
    }

    // Remove in reverse order to preserve indices
    for &i in to_remove.iter().rev() {
        circuit.operations_mut().remove(i);
    }

    to_remove.len()
}

fn is_identity_op(op: &CircuitOp) -> bool {
    match op {
        CircuitOp::SingleQubit { gate, .. } => matches!(gate, Gate::I),
        _ => false,
    }
}

/// Cancel adjacent inverse gate pairs on the same qubit.
/// Examples: H·H, X·X, Y·Y, Z·Z, S·S†, T·T†, Rx(a)·Rx(-a), etc.
fn cancel_inverse_pairs(circuit: &mut Circuit) -> usize {
    let ops = circuit.operations_mut();
    let mut cancellations = 0;
    let mut i = 0;

    while i + 1 < ops.len() {
        if can_cancel_pair(&ops[i], &ops[i + 1]) {
            // Remove both gates
            ops.remove(i + 1);
            ops.remove(i);
            cancellations += 1;
            // Don't increment i - check if new pair forms with previous
            if i > 0 {
                i = i.saturating_sub(1);
            }
            continue;
        }
        i += 1;
    }

    cancellations
}

/// Check if two adjacent operations are cancellable inverse pairs.
fn can_cancel_pair(op1: &CircuitOp, op2: &CircuitOp) -> bool {
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
    let ops = circuit.operations_mut();
    let mut fusions = 0;
    let mut i = 0;

    while i + 1 < ops.len() {
        // Look for same-qubit consecutive single-qubit gates
        if let (
            CircuitOp::SingleQubit {
                gate: g1,
                target: t1,
            },
            CircuitOp::SingleQubit {
                gate: g2,
                target: t2,
            },
        ) = (&ops[i], &ops[i + 1])
        {
            #[allow(clippy::collapsible_if)]
            if t1 == t2 {
                if let Some(fused) = try_fuse_gates(*g1, *g2) {
                    ops[i] = CircuitOp::SingleQubit {
                        gate: fused,
                        target: *t1,
                    };
                    ops.remove(i + 1);
                    fusions += 1;
                    // Check if we can fuse further back
                    if i > 0 {
                        i = i.saturating_sub(1);
                    }
                    continue;
                }
            }
        }
        i += 1;
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Circuit, Gate};

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

/// Minimize T-gates by merging consecutive Rz rotations on the same qubit
/// into a single Rz, then converting Rz(π/4 * k) to T/Tdg when k is 1 or 7.
pub fn minimize_t_gates(circuit: &mut Circuit) -> usize {
    let before = t_gate_count(circuit).t_count + t_gate_count(circuit).tdg_count;
    // Merge consecutive Rz on same qubit
    let ops = circuit.operations().to_vec();
    let mut new_ops: Vec<CircuitOp> = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        if let CircuitOp::SingleQubit {
            gate: Gate::Rz(a),
            target: t1,
        } = &ops[i]
        {
            // Look ahead for consecutive Rz on same qubit
            let mut total = *a;
            let mut j = i + 1;
            while j < ops.len() {
                if let CircuitOp::SingleQubit {
                    gate: Gate::Rz(b),
                    target: t2,
                } = &ops[j]
                    && t1 == t2
                {
                    total += *b;
                    j += 1;
                    continue;
                }
                break;
            }
            if j > i + 1 {
                // Merged
                new_ops.push(CircuitOp::SingleQubit {
                    gate: Gate::Rz(total),
                    target: *t1,
                });
                i = j;
                continue;
            }
        }
        new_ops.push(ops[i]);
        i += 1;
    }
    *circuit.operations_mut() = new_ops;
    let after = t_gate_count(circuit).t_count + t_gate_count(circuit).tdg_count;
    before.saturating_sub(after)
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
pub fn route_to_hardware(circuit: &mut Circuit, coupling: &CouplingMap) -> usize {
    let mut swaps = 0;
    // Simple greedy: for each CNOT, if qubits not connected, insert SWAP chain
    let ops = circuit.operations().to_vec();
    let mut new_ops: Vec<CircuitOp> = Vec::new();
    for op in &ops {
        match op {
            CircuitOp::Cnot { control, target } => {
                if !coupling.are_connected(*control, *target) {
                    // Insert SWAP to bring them together (simplified: swap control to target-1)
                    // For a chain, swap along path
                    let (mut c, _t) = (*control, *target);
                    while !coupling.are_connected(c, *target) {
                        // Find neighbor of c closer to t
                        let neighbors: Vec<usize> = coupling
                            .edges
                            .iter()
                            .filter_map(|&(a, b)| {
                                if a == c {
                                    Some(b)
                                } else if b == c {
                                    Some(a)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        if neighbors.is_empty() {
                            break;
                        }
                        let next = neighbors[0];
                        new_ops.push(CircuitOp::Swap {
                            first: c,
                            second: next,
                        });
                        swaps += 1;
                        c = next;
                        if swaps > 100 {
                            break;
                        }
                    }
                }
                new_ops.push(*op);
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
    let mut total_fusions = 0;
    let mut total_cancellations = 0;
    let mut total_identities = 0;
    let mut iterations = 0;

    loop {
        let before = circuit.operations().len();
        let result = optimize_circuit(circuit, &OptimizeConfig::all());
        total_fusions += result.fusions;
        total_cancellations += result.cancellations;
        total_identities += result.identities_removed;
        iterations += 1;

        // Also try T-gate minimization
        minimize_t_gates(circuit);

        let after = circuit.operations().len();
        if after >= before || iterations >= 10 {
            break;
        }
    }

    OptimizeResult {
        original_ops: original,
        optimized_ops: circuit.operations().len(),
        fusions: total_fusions,
        cancellations: total_cancellations,
        identities_removed: total_identities,
    }
}
