//! Tests for `noise_model`: physically parameterized thermal + dephasing noise.
//!
//! The module had 3/245 lines covered. It is the subsystem every noisy result
//! depends on, so the tests here pin both the plumbing (which channels land
//! where, and what the summary counts) and the physics (detailed balance and the
//! documented `T2` coherence contract).

use quantumforge::noise_model::{
    NoiseApplicationSummary, NoiseModelError, OperationTiming, ThermalGateNoiseModel,
};
use quantumforge::{Circuit, CircuitOp, DensityMatrix, NoiseChannel};

const T1: f64 = 30.0;
const T2: f64 = 20.0;
const BETA_DELTA: f64 = 1.0;
const SQ_TIME: f64 = 1.0;
const TQ_TIME: f64 = 2.0;

fn model() -> ThermalGateNoiseModel {
    ThermalGateNoiseModel::new(T1, T2, BETA_DELTA, SQ_TIME, TQ_TIME).expect("valid model")
}

/// `(target, channel)` for every channel operation, in circuit order.
fn channel_ops(circuit: &Circuit) -> Vec<(usize, NoiseChannel)> {
    circuit
        .operations()
        .iter()
        .filter_map(|operation| match *operation {
            CircuitOp::Channel { target, channel } => Some((target, channel)),
            _ => None,
        })
        .collect()
}

/// Non-channel operations, in circuit order.
fn plain_ops(circuit: &Circuit) -> Vec<CircuitOp> {
    circuit
        .operations()
        .iter()
        .copied()
        .filter(|operation| !matches!(operation, CircuitOp::Channel { .. }))
        .collect()
}

fn thermal_rate(channel: &NoiseChannel) -> f64 {
    match *channel {
        NoiseChannel::ThermalDamping { gamma_down, .. } => gamma_down,
        other => panic!("expected a thermal channel, got {other:?}"),
    }
}

fn dephasing_rate(channel: &NoiseChannel) -> f64 {
    match *channel {
        NoiseChannel::PhaseDamping { gamma } => gamma,
        other => panic!("expected a phase-damping channel, got {other:?}"),
    }
}

/// NaN-safe equality: the constructor must echo back exactly what it rejected.
fn same_value(left: f64, right: f64) -> bool {
    if left.is_nan() && right.is_nan() {
        return true;
    }
    left == right
}

fn timing(start: f64, duration: f64) -> OperationTiming {
    OperationTiming { start, duration }
}

// ---------------------------------------------------------------- construction

#[test]
fn constructor_rejects_non_positive_or_non_finite_parameters() {
    let cases = [
        ("t1", 0.0, 20.0, 1.0, 1.0, 2.0),
        ("t1", -1.0, 20.0, 1.0, 1.0, 2.0),
        ("t1", f64::NAN, 20.0, 1.0, 1.0, 2.0),
        ("t1", f64::INFINITY, 20.0, 1.0, 1.0, 2.0),
        ("t2", 30.0, 0.0, 1.0, 1.0, 2.0),
        ("t2", 30.0, -5.0, 1.0, 1.0, 2.0),
        ("beta_delta", 30.0, 20.0, -0.1, 1.0, 2.0),
        ("single_qubit_gate_time", 30.0, 20.0, 1.0, -1.0, 2.0),
        ("two_qubit_gate_time", 30.0, 20.0, 1.0, 1.0, f64::NAN),
    ];
    for (name, t1, t2, beta, sq, tq) in cases {
        let error = ThermalGateNoiseModel::new(t1, t2, beta, sq, tq)
            .expect_err("must reject the invalid parameter");
        match error {
            NoiseModelError::InvalidParameter { name: got, value } => {
                assert_eq!(got, name, "wrong parameter named for {name}={value}");
                let expected = match name {
                    "t1" => t1,
                    "t2" => t2,
                    "beta_delta" => beta,
                    "single_qubit_gate_time" => sq,
                    _ => tq,
                };
                assert!(
                    same_value(value, expected),
                    "error echoed {value} for {name}, supplied {expected}"
                );
            }
            other => panic!("expected InvalidParameter for {name}, got {other:?}"),
        }
    }
}

#[test]
fn t2_above_the_markovian_bound_is_rejected() {
    let error =
        ThermalGateNoiseModel::new(10.0, 20.5, 1.0, 1.0, 2.0).expect_err("T2 must not exceed 2*T1");
    match error {
        NoiseModelError::T2ExceedsTwoT1 { t2, t1 } => {
            assert_eq!(t2, 20.5);
            assert_eq!(t1, 10.0);
        }
        other => panic!("expected T2ExceedsTwoT1, got {other:?}"),
    }
    // Exactly at the bound is legal.
    assert!(ThermalGateNoiseModel::new(10.0, 20.0, 1.0, 1.0, 2.0).is_ok());
}

#[test]
fn getters_expose_the_constructor_values() {
    let model = model();
    assert_eq!(model.t1(), T1);
    assert_eq!(model.t2(), T2);
    assert_eq!(model.beta_delta(), BETA_DELTA);
    assert_eq!(model.single_qubit_gate_time(), SQ_TIME);
    assert_eq!(model.two_qubit_gate_time(), TQ_TIME);
}

#[test]
fn for_schedule_has_zero_default_gate_times() {
    let model = ThermalGateNoiseModel::for_schedule(T1, T2, BETA_DELTA).expect("valid");
    assert_eq!(model.single_qubit_gate_time(), 0.0);
    assert_eq!(model.two_qubit_gate_time(), 0.0);

    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    let noisy = model.apply(&circuit);
    // Zero durations mean no relaxation and no dephasing, so nothing is inserted.
    assert!(channel_ops(&noisy.circuit).is_empty());
    assert_eq!(plain_ops(&noisy.circuit).len(), 1);
}

// --------------------------------------------------------------- apply()

#[test]
fn apply_places_all_thermal_then_all_dephasing_after_each_gate() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    circuit.cnot(0, 1);
    let noisy = model().apply(&circuit);

    let ops = noisy.circuit.operations();
    // H(0), thermal(0), dephase(0), CNOT(0,1), thermal(0), thermal(1), dephase(0), dephase(1)
    assert_eq!(ops.len(), 8, "got {}", ops.len());
    assert!(matches!(ops[0], CircuitOp::SingleQubit { target: 0, .. }));
    assert!(matches!(ops[1], CircuitOp::Channel { target: 0, .. }));
    assert!(matches!(ops[2], CircuitOp::Channel { target: 0, .. }));
    assert!(matches!(
        ops[3],
        CircuitOp::Cnot {
            control: 0,
            target: 1
        }
    ));
    assert!(matches!(ops[4], CircuitOp::Channel { target: 0, .. }));
    assert!(matches!(ops[5], CircuitOp::Channel { target: 1, .. }));
    assert!(matches!(ops[6], CircuitOp::Channel { target: 0, .. }));
    assert!(matches!(ops[7], CircuitOp::Channel { target: 1, .. }));

    let channels = channel_ops(&noisy.circuit);
    assert_eq!(
        channels
            .iter()
            .filter(|c| matches!(c.1, NoiseChannel::ThermalDamping { .. }))
            .count(),
        3
    );
    assert_eq!(
        channels
            .iter()
            .filter(|c| matches!(c.1, NoiseChannel::PhaseDamping { .. }))
            .count(),
        3
    );
    assert_eq!(
        noisy.summary,
        NoiseApplicationSummary {
            thermal_channels: 3,
            dephasing_channels: 3,
            idle_thermal_channels: 0,
            idle_dephasing_channels: 0,
        }
    );
}

#[test]
fn apply_copies_measurements_and_explicit_channels_unchanged() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::X, 0);
    circuit.channel(1, NoiseChannel::Depolarizing { probability: 0.1 });
    circuit.measure(0, 0);
    let noisy = model().apply(&circuit);

    let ops = noisy.circuit.operations();
    assert_eq!(ops.len(), 5, "X + 2 channels + explicit channel + measure");
    // X(0) emits thermal(0) then dephase(0); only then come the copied operations.
    assert!(matches!(ops[2], CircuitOp::Channel { target: 0, .. }));
    assert!(matches!(
        ops[3],
        CircuitOp::Channel {
            target: 1,
            channel: NoiseChannel::Depolarizing { probability },
        } if (probability - 0.1).abs() < 1e-12
    ));
    assert!(matches!(
        ops[4],
        CircuitOp::Measure {
            target: 0,
            classical_bit: 0
        }
    ));
    // Only the X gate produced noise: one thermal + one dephasing on qubit 0.
    assert!(
        channel_ops(&noisy.circuit)
            .iter()
            .filter(|(target, _)| *target == 0)
            .all(|(_, channel)| matches!(
                channel,
                NoiseChannel::ThermalDamping { .. } | NoiseChannel::PhaseDamping { .. }
            ))
    );
    assert_eq!(noisy.summary.thermal_channels, 1);
    assert_eq!(noisy.summary.dephasing_channels, 1);
}

#[test]
fn apply_with_zero_gate_durations_inserts_nothing() {
    let model = ThermalGateNoiseModel::new(T1, T2, BETA_DELTA, 0.0, 0.0).expect("valid");
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    circuit.cnot(0, 1);
    let noisy = model.apply(&circuit);
    assert!(channel_ops(&noisy.circuit).is_empty());
    assert_eq!(plain_ops(&noisy.circuit).len(), 2);
}

#[test]
fn two_qubit_gates_use_the_two_qubit_duration_for_both_operands() {
    let mut circuit = Circuit::new();
    circuit.cz(1, 2);
    let noisy = model().apply(&circuit);
    let channels = channel_ops(&noisy.circuit);
    assert_eq!(channels.len(), 4, "thermal+dephasing on each operand");
    let expected = 1.0 - (-TQ_TIME / T1).exp();
    let boltzmann = (-BETA_DELTA).exp();
    let expected_down = expected / (1.0 + boltzmann);
    for (target, channel) in &channels {
        assert!(*target == 1 || *target == 2);
        if let NoiseChannel::ThermalDamping {
            gamma_down,
            beta_delta,
        } = channel
        {
            assert!((gamma_down - expected_down).abs() < 1e-12);
            assert_eq!(*beta_delta, BETA_DELTA);
        }
    }
}

// ------------------------------------------------------------------ physics

#[test]
fn thermal_rate_matches_the_t1_relaxation_probability() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    let noisy = model().apply(&circuit);
    let channels = channel_ops(&noisy.circuit);
    let gamma_down = thermal_rate(&channels[0].1);

    let relaxation = -(-SQ_TIME / T1).exp_m1();
    let boltzmann = (-BETA_DELTA).exp();
    assert!(
        (gamma_down - relaxation / (1.0 + boltzmann)).abs() < 1e-12,
        "gamma_down {gamma_down}"
    );
    assert!(gamma_down > 0.0 && gamma_down < 1.0);
}

/// The documented contract: after thermal relaxation plus the chosen pure
/// dephasing, the coherence magnitude decays by exactly `exp(-duration / T2)`,
/// and populations follow detailed balance with `p1/p0 = exp(-beta_delta)`.
#[test]
fn coherence_and_populations_match_the_t2_contract() {
    let duration = SQ_TIME;
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    let noisy = model().apply(&circuit);
    let channels = channel_ops(&noisy.circuit);
    let gamma_down = thermal_rate(&channels[0].1);
    let gamma_phi = dephasing_rate(&channels[1].1);
    let gamma_up = gamma_down * (-BETA_DELTA).exp();

    // Start from |+> so the off-diagonal is 1/2.
    let mut rho = DensityMatrix::from_entries(
        vec![
            quantumforge::Complex64::new(0.5, 0.0),
            quantumforge::Complex64::new(0.5, 0.0),
            quantumforge::Complex64::new(0.5, 0.0),
            quantumforge::Complex64::new(0.5, 0.0),
        ],
        1,
    );
    rho.apply_channel(channels[0].1, 0).expect("applies");
    rho.apply_channel(channels[1].1, 0).expect("applies");

    let entries = rho.entries();
    let p1_expected = 0.5 * (1.0 - gamma_down) + 0.5 * gamma_up;
    let p0_expected = 0.5 * (1.0 - gamma_up) + 0.5 * gamma_down;
    assert!(
        (entries[3].re - p1_expected).abs() < 1e-12,
        "p1 {}",
        entries[3].re
    );
    assert!(
        (entries[0].re - p0_expected).abs() < 1e-12,
        "p0 {}",
        entries[0].re
    );

    let coherence = entries[1].norm_sqr().sqrt();
    let expected_coherence = 0.5 * (-duration / T2).exp();
    assert!(
        (coherence - expected_coherence).abs() < 1e-12,
        "coherence {coherence} vs exp(-t/T2)/2 {expected_coherence}"
    );
    // Dephasing must never amplify coherence, and the pair must not be a no-op.
    assert!(gamma_phi > 0.0 && gamma_phi < 1.0);
    assert!(coherence < 0.5);
    assert_eq!(rho.trace().re, 1.0);
}

#[test]
fn repeated_thermal_channels_converge_to_the_gibbs_state() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    let noisy = model().apply(&circuit);
    let mut channels = channel_ops(&noisy.circuit);
    let thermal = channels.remove(0).1;

    // Start from |1>.
    let mut rho = DensityMatrix::from_entries(
        vec![
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ONE,
        ],
        1,
    );
    for _ in 0..5_000 {
        rho.apply_channel(thermal, 0).expect("applies");
    }
    let excited = rho.entries()[3].re;
    let boltzmann = (-BETA_DELTA).exp();
    let expected = boltzmann / (1.0 + boltzmann);
    assert!(
        (excited - expected).abs() < 1e-9,
        "excited population {excited} vs Gibbs {expected}"
    );
}

#[test]
fn zero_beta_delta_is_the_symmetric_high_temperature_limit() {
    let model = ThermalGateNoiseModel::new(T1, T2, 0.0, SQ_TIME, TQ_TIME).expect("valid");
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    let mut channels = channel_ops(&model.apply(&circuit).circuit);
    let thermal = channels.remove(0).1;

    // beta*Delta -> 0 means infinite temperature: both levels equally likely.
    let mut rho = DensityMatrix::from_entries(
        vec![
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::ONE,
        ],
        1,
    );
    for _ in 0..5_000 {
        rho.apply_channel(thermal, 0).expect("applies");
    }
    assert!(
        (rho.entries()[3].re - 0.5).abs() < 1e-9,
        "excited population should tend to 1/2, got {}",
        rho.entries()[3].re
    );
}

// ---------------------------------------------------------- apply_scheduled

#[test]
fn scheduled_rejects_a_wrong_number_of_timings() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    let error = model()
        .apply_scheduled(&circuit, &[])
        .expect_err("one operation needs one timing");
    match error {
        NoiseModelError::ScheduleLengthMismatch {
            operations,
            timings,
        } => {
            assert_eq!(operations, 1);
            assert_eq!(timings, 0);
        }
        other => panic!("expected ScheduleLengthMismatch, got {other:?}"),
    }
}

#[test]
fn scheduled_rejects_invalid_timings() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    circuit.push(quantumforge::Gate::X, 0);
    for (index, bad) in [
        timing(-1.0, 1.0),
        timing(0.0, -1.0),
        timing(f64::NAN, 1.0),
        timing(0.0, f64::INFINITY),
    ]
    .into_iter()
    .enumerate()
    {
        let error = model()
            .apply_scheduled(&circuit, &[timing(0.0, 1.0), bad])
            .expect_err("invalid timing must be rejected");
        match error {
            NoiseModelError::InvalidScheduleTiming {
                operation,
                start,
                duration,
            } => {
                assert_eq!(
                    operation, 1,
                    "the second timing is the bad one (case {index})"
                );
                assert!(same_value(start, bad.start));
                assert!(same_value(duration, bad.duration));
            }
            other => panic!("expected InvalidScheduleTiming, got {other:?}"),
        }
    }
}

#[test]
fn scheduled_rejects_overlapping_operations_on_a_shared_qubit() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    circuit.push(quantumforge::Gate::X, 0);
    let error = model()
        .apply_scheduled(&circuit, &[timing(0.0, 1.0), timing(0.5, 1.0)])
        .expect_err("operations overlap on qubit 0");
    match error {
        NoiseModelError::QubitScheduleOrderViolation {
            qubit,
            first_operation,
            second_operation,
        } => {
            assert_eq!(qubit, 0);
            assert_eq!(first_operation, 0);
            assert_eq!(second_operation, 1);
        }
        other => panic!("expected QubitScheduleOrderViolation, got {other:?}"),
    }
    // Back-to-back is legal: starting exactly when the previous one ends.
    assert!(
        model()
            .apply_scheduled(&circuit, &[timing(0.0, 1.0), timing(1.0, 1.0)])
            .is_ok()
    );
}

#[test]
fn scheduled_rejects_a_classical_dependency_reordering() {
    let mut circuit = Circuit::new();
    circuit.measure(0, 0);
    circuit.if_classical_bit(0, 1, quantumforge::Gate::X, 1);
    let error = model()
        .apply_scheduled(&circuit, &[timing(0.0, 1.0), timing(0.5, 1.0)])
        .expect_err("the conditional reads a bit before the measurement finishes");
    match error {
        NoiseModelError::ClassicalDependencyViolation {
            first_operation,
            second_operation,
            classical_bit,
        } => {
            assert_eq!(first_operation, 0);
            assert_eq!(second_operation, 1);
            assert_eq!(classical_bit, 0);
        }
        other => panic!("expected ClassicalDependencyViolation, got {other:?}"),
    }
}

#[test]
fn scheduled_places_idle_noise_before_the_gate_that_follows_it() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    // One operation starting at t=2 and lasting 1, so qubit 0 idles over [0, 2).
    let noisy = model()
        .apply_scheduled(&circuit, &[timing(2.0, 1.0)])
        .expect("valid schedule");

    let ops = noisy.circuit.operations();
    // idle thermal, idle dephase, the gate, then the gate's own thermal + dephase
    assert_eq!(ops.len(), 5, "got {}", ops.len());
    // Idle channels are emitted at the END of the idle interval, i.e. immediately
    // before the gate, so they land first in the sorted event list.
    assert!(matches!(ops[0], CircuitOp::Channel { target: 0, .. }));
    assert!(matches!(ops[1], CircuitOp::Channel { target: 0, .. }));
    assert!(matches!(ops[2], CircuitOp::SingleQubit { target: 0, .. }));
    assert!(matches!(ops[3], CircuitOp::Channel { target: 0, .. }));
    // Idle uses the 2-unit gap; the gate uses its 1-unit duration.
    // Circuit order: idle thermal, idle dephase, gate, gate thermal, gate dephase.
    let channels = channel_ops(&noisy.circuit);
    assert_eq!(channels.len(), 4);
    assert!(matches!(channels[0].1, NoiseChannel::ThermalDamping { .. }));
    assert!(matches!(channels[1].1, NoiseChannel::PhaseDamping { .. }));
    assert!(matches!(channels[2].1, NoiseChannel::ThermalDamping { .. }));
    assert!(matches!(channels[3].1, NoiseChannel::PhaseDamping { .. }));
    let idle = thermal_rate(&channels[0].1);
    let gate = thermal_rate(&channels[2].1);
    assert!((idle - (-(-2.0 / T1).exp_m1()) / (1.0 + (-BETA_DELTA).exp())).abs() < 1e-12);
    assert!((gate - (-(-1.0 / T1).exp_m1()) / (1.0 + (-BETA_DELTA).exp())).abs() < 1e-12);
    assert!(idle > gate, "a longer idle must relax further");
    assert_eq!(
        noisy.summary,
        NoiseApplicationSummary {
            thermal_channels: 2,
            dephasing_channels: 2,
            idle_thermal_channels: 1,
            idle_dephasing_channels: 1,
        }
    );
}

#[test]
fn contiguous_schedule_inserts_no_idle_noise() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    circuit.push(quantumforge::Gate::X, 0);
    let noisy = model()
        .apply_scheduled(&circuit, &[timing(0.0, 1.0), timing(1.0, 1.0)])
        .expect("valid schedule");
    assert_eq!(noisy.summary.idle_thermal_channels, 0);
    assert_eq!(noisy.summary.idle_dephasing_channels, 0);
    // Two single-qubit gates: one thermal + one dephasing each.
    assert_eq!(noisy.summary.thermal_channels, 2);
    assert_eq!(noisy.summary.dephasing_channels, 2);
}

#[test]
fn trailing_idle_after_the_last_operation_is_included() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    circuit.push(quantumforge::Gate::H, 1);
    // Qubit 0 is free from t=1 to the makespan at t=3; qubit 1 is busy throughout.
    let noisy = model()
        .apply_scheduled(&circuit, &[timing(0.0, 1.0), timing(0.0, 3.0)])
        .expect("valid schedule");
    assert_eq!(noisy.summary.idle_thermal_channels, 1);
    assert_eq!(noisy.summary.idle_dephasing_channels, 1);

    let idle_down = -(-(3.0 - 1.0) / T1).exp_m1() / (1.0 + (-BETA_DELTA).exp());
    let channels = channel_ops(&noisy.circuit);
    let found = channels.iter().any(|(target, channel)| {
        *target == 0
            && matches!(channel, NoiseChannel::ThermalDamping { .. })
            && same_value(thermal_rate(channel), idle_down)
    });
    assert!(
        found,
        "expected one idle thermal channel on qubit 0 with gamma_down={idle_down}"
    );
}

#[test]
fn measurements_receive_no_gate_noise_but_still_collect_idle_noise() {
    let mut circuit = Circuit::new();
    circuit.measure(0, 0);
    let noisy = model()
        .apply_scheduled(&circuit, &[timing(1.0, 1.0)])
        .expect("valid schedule");
    // No gate channels; the [0,1) idle window still produces both channel kinds.
    assert_eq!(noisy.summary.thermal_channels, 1);
    assert_eq!(noisy.summary.dephasing_channels, 1);
    assert_eq!(noisy.summary.idle_thermal_channels, 1);
    assert_eq!(noisy.summary.idle_dephasing_channels, 1);
    assert_eq!(plain_ops(&noisy.circuit).len(), 1);
}

#[test]
fn scheduled_preserves_program_order_and_runs() {
    let mut circuit = Circuit::new();
    circuit.push(quantumforge::Gate::H, 0);
    circuit.cnot(0, 1);
    let noisy = model()
        .apply_scheduled(&circuit, &[timing(0.0, 1.0), timing(1.0, 2.0)])
        .expect("valid schedule");

    assert_eq!(
        plain_ops(&circuit),
        plain_ops(&noisy.circuit),
        "the program order must be untouched"
    );

    // The inserted channels must execute and leave a normalized density matrix.
    let mut rho = DensityMatrix::maximally_mixed(2).expect("valid");
    for operation in noisy.circuit.operations() {
        if let CircuitOp::Channel { target, channel } = *operation {
            rho.apply_channel(channel, target).expect("channel applies");
        }
    }
    assert!(
        (rho.trace().re - 1.0).abs() < 1e-12,
        "trace {}",
        rho.trace().re
    );
    assert!(rho.purity() <= 1.0 + 1e-12);
}
