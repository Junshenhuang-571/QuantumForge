//! Thermal noise channels and von Neumann entropy.

use quantumforge::{DensityMatrix, Gate, NoiseChannel, QuantumError, SmallRng, StateVector};

fn one_state() -> StateVector {
    let mut state = StateVector::zero(1).unwrap();
    state.apply(Gate::X, 0).unwrap();
    state
}

fn plus_state() -> StateVector {
    let mut state = StateVector::zero(1).unwrap();
    state.apply(Gate::H, 0).unwrap();
    state
}

fn thermal_rates(gamma_down: f64, beta_delta: f64) -> (f64, f64) {
    let gamma_up = gamma_down * (-beta_delta).exp();
    (gamma_up, gamma_down)
}

#[test]
fn entropy_of_pure_and_maximally_mixed_states() {
    let pure = DensityMatrix::from_state(&plus_state());
    assert!(pure.von_neumann_entropy().unwrap().abs() < 1e-12);

    let mixed = DensityMatrix::maximally_mixed(1).unwrap();
    assert!((mixed.von_neumann_entropy().unwrap() - std::f64::consts::LN_2).abs() < 1e-12);

    // A non-Hermitian matrix is rejected rather than silently diagonalised.
    let bad = DensityMatrix::from_entries(
        vec![
            quantumforge::Complex64::new(1.0, 0.0),
            quantumforge::Complex64::new(1.0, 0.0),
            quantumforge::Complex64::ZERO,
            quantumforge::Complex64::new(0.0, 0.0),
        ],
        1,
    );
    assert!(matches!(
        bad.von_neumann_entropy(),
        Err(QuantumError::NonHermitianDensityMatrix)
    ));
}

#[test]
fn thermal_damping_prepares_the_gibbs_state_at_the_predicted_rate() {
    let gamma_down = 0.10;
    let beta_delta = 1.0;
    let (gamma_up, gamma_down) = thermal_rates(gamma_down, beta_delta);
    let channel = NoiseChannel::ThermalDamping {
        gamma_down,
        beta_delta,
    };
    let target = (-beta_delta).exp() / (1.0 + (-beta_delta).exp());
    let rate = 1.0 - gamma_up - gamma_down;

    let mut rho = DensityMatrix::from_state(&one_state());
    for step in 0..=20 {
        let p1 = rho.probability_one(0).unwrap();
        let predicted = target + (1.0 - target) * rate.powi(step);
        assert!(
            (p1 - predicted).abs() < 1e-12,
            "step {step}: p1 {p1} vs predicted {predicted}"
        );
        assert!(
            (rho.trace().re - 1.0).abs() < 1e-12,
            "trace drifted at {step}"
        );
        assert!(rho.purity() <= 1.0 + 1e-12, "purity exceeded 1 at {step}");
        rho.apply_channel(channel, 0).unwrap();
    }

    // At the fixed point the entropy is the binary entropy of the Gibbs weight.
    for _ in 0..200 {
        rho.apply_channel(channel, 0).unwrap();
    }
    let fixed_p1 = rho.probability_one(0).unwrap();
    assert!((fixed_p1 - target).abs() < 1e-12);
    let expected_entropy = -(target * target.ln() + (1.0 - target) * (1.0 - target).ln());
    assert!((rho.von_neumann_entropy().unwrap() - expected_entropy).abs() < 1e-12);
}

#[test]
fn thermal_damping_fixed_point_is_independent_of_the_starting_state() {
    let channel = NoiseChannel::ThermalDamping {
        gamma_down: 0.15,
        beta_delta: 0.8,
    };
    let target = (-0.8f64).exp() / (1.0 + (-0.8f64).exp());
    for initial in [one_state(), plus_state()] {
        let mut rho = DensityMatrix::from_state(&initial);
        for _ in 0..400 {
            rho.apply_channel(channel, 0).unwrap();
        }
        assert!((rho.probability_one(0).unwrap() - target).abs() < 1e-12);
        // The coherences die out, leaving the diagonal Gibbs state.
        assert!(rho.entries()[1].norm_sqr().sqrt() < 1e-12);
    }
}

#[test]
fn thermal_damping_interpolates_between_amplitude_damping_and_infinite_temperature() {
    // beta_delta -> 0 gives the infinite-temperature fixed point p1 = 1/2.
    let hot = NoiseChannel::ThermalDamping {
        gamma_down: 0.3,
        beta_delta: 0.0,
    };
    let mut rho = DensityMatrix::from_state(&one_state());
    for _ in 0..50 {
        rho.apply_channel(hot, 0).unwrap();
    }
    assert!((rho.probability_one(0).unwrap() - 0.5).abs() < 1e-12);

    // A large beta_delta drives the fixed point toward |0><0|.
    let cold = NoiseChannel::ThermalDamping {
        gamma_down: 0.3,
        beta_delta: 40.0,
    };
    let mut rho = DensityMatrix::from_state(&one_state());
    for _ in 0..150 {
        rho.apply_channel(cold, 0).unwrap();
    }
    assert!(rho.probability_one(0).unwrap() < 1e-12);
}

#[test]
fn thermal_damping_rejects_invalid_parameters() {
    let bad_rate = NoiseChannel::ThermalDamping {
        gamma_down: 1.5,
        beta_delta: 1.0,
    };
    let mut rho = DensityMatrix::from_state(&one_state());
    assert!(matches!(
        rho.apply_channel(bad_rate, 0),
        Err(QuantumError::InvalidProbability(_))
    ));

    let bad_beta = NoiseChannel::ThermalDamping {
        gamma_down: 0.1,
        beta_delta: -1.0,
    };
    assert!(matches!(
        rho.apply_channel(bad_beta, 0),
        Err(QuantumError::InvalidProbability(_))
    ));
}

#[test]
fn thermal_damping_trajectories_reproduce_the_channel() {
    let gamma_down = 0.20;
    let beta_delta = 0.70;
    let (gamma_up, gamma_down) = thermal_rates(gamma_down, beta_delta);
    let channel = NoiseChannel::ThermalDamping {
        gamma_down,
        beta_delta,
    };
    // From |+> the exact one-step excited population is
    // p1' = p1 (1 - gamma_down) + p0 gamma_up with p0 = p1 = 1/2.
    let expected = 0.5 * (1.0 - gamma_down) + 0.5 * gamma_up;

    let trials = 20_000u64;
    let mut sum = 0.0;
    for seed in 1..=trials {
        let mut state = plus_state();
        let mut rng = SmallRng::seeded(seed);
        state
            .apply_channel_trajectory(channel, 0, &mut rng)
            .unwrap();
        assert!(
            (state.norm_sqr() - 1.0).abs() < 1e-12,
            "trajectory not normalized"
        );
        sum += state.amplitudes[1].norm_sqr();
    }
    let mean = sum / trials as f64;
    // The standard error of the mean is at most 0.5/sqrt(trials) ~ 0.0035.
    assert!(
        (mean - expected).abs() < 0.02,
        "trajectory mean {mean} vs channel {expected}"
    );
}

#[test]
fn thermal_damping_circuit_shots_run() {
    use quantumforge::Circuit;
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.channel(
        0,
        NoiseChannel::ThermalDamping {
            gamma_down: 0.2,
            beta_delta: 1.0,
        },
    );
    circuit.measure(0, 0);
    let result = circuit
        .run_shots_with_noise(&StateVector::zero(1).unwrap(), 256, 7)
        .unwrap();
    assert_eq!(result.shots, 256);
    let total: usize = result.counts.values().sum();
    assert_eq!(total, 256);
}
