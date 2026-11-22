//! Dissipative state preparation with a jump-operator channel.

use quantumforge::dissipative::{fidelity, jump_to_target, lindblad_step, prepare_target};
use quantumforge::{Complex64, DensityMatrix, StateVector};

fn one_qubit_state() -> DensityMatrix {
    DensityMatrix::from_state(&StateVector::zero(1).unwrap())
}

#[test]
fn prepares_the_ground_state_from_the_excited_state() {
    // Start from |1> and dissipate to |0>.
    let mut state = StateVector::zero(1).unwrap();
    state.apply(quantumforge::Gate::X, 0).unwrap();
    let mut rho = DensityMatrix::from_state(&state);
    prepare_target(
        &mut rho,
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        0.05,
        400,
    )
    .unwrap();
    assert!(fidelity(&rho, [Complex64::new(1.0, 0.0), Complex64::ZERO], 0) > 1.0 - 1e-6);
    assert!((rho.trace().re - 1.0).abs() < 1e-12);
}

#[test]
fn prepares_an_arbitrary_superposition() {
    // |t> = cos(0.4) |0> + sin(0.4) |1>
    let target = [
        Complex64::new(0.4f64.cos(), 0.0),
        Complex64::new(0.4f64.sin(), 0.0),
    ];
    let mut rho = one_qubit_state();
    prepare_target(&mut rho, target, 0.05, 400).unwrap();
    assert!(fidelity(&rho, target, 0) > 1.0 - 1e-6);
}

#[test]
fn jump_to_zero_is_the_amplitude_damping_jump() {
    let jump = jump_to_target([Complex64::new(1.0, 0.0), Complex64::ZERO]);
    // L = |0><1| = [[0, 1], [0, 0]]
    assert_eq!(jump[0][0], Complex64::ZERO);
    assert_eq!(jump[0][1], Complex64::ONE);
    assert_eq!(jump[1][0], Complex64::ZERO);
    assert_eq!(jump[1][1], Complex64::ZERO);

    // Applying it to |1> should move population to |0>.
    let mut state = StateVector::zero(1).unwrap();
    state.apply(quantumforge::Gate::X, 0).unwrap();
    let mut rho = DensityMatrix::from_state(&state);
    for _ in 0..500 {
        lindblad_step(&mut rho, jump, 0, 0.05).unwrap();
    }
    assert!(rho.probability_one(0).unwrap() < 1e-9);
}

#[test]
fn lindblad_step_rejects_bad_input() {
    let jump = jump_to_target([Complex64::new(1.0, 0.0), Complex64::ZERO]);
    let mut rho = one_qubit_state();
    assert!(lindblad_step(&mut rho, jump, 1, 0.1).is_err());
    assert!(lindblad_step(&mut rho, jump, 0, -0.1).is_err());
}

#[test]
fn prepares_ghz_and_w_states() {
    use quantumforge::dissipative::{prepare_target_state, state_fidelity};

    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut ghz = vec![Complex64::ZERO; 8];
    ghz[0] = Complex64::new(inv_sqrt2, 0.0);
    ghz[7] = Complex64::new(inv_sqrt2, 0.0);

    let mut w = vec![Complex64::ZERO; 8];
    let inv_sqrt3 = 1.0 / 3.0f64.sqrt();
    for index in [1usize, 2, 4] {
        w[index] = Complex64::new(inv_sqrt3, 0.0);
    }

    for (label, target) in [("GHZ", ghz), ("W", w)] {
        let mut rho = DensityMatrix::from_state(&StateVector::zero(3).unwrap());
        prepare_target_state(&mut rho, &target, 0.05, 400).unwrap();
        let fidelity = state_fidelity(&rho, &target);
        assert!(fidelity > 1.0 - 1e-6, "{label} fidelity {fidelity}");
        assert!((rho.trace().re - 1.0).abs() < 1e-12);
    }
}

#[test]
fn prepares_a_random_entangled_target() {
    use quantumforge::SmallRng;
    use quantumforge::dissipative::{prepare_target_state, state_fidelity};

    let mut rng = SmallRng::seeded(5);
    let mut target = vec![Complex64::ZERO; 8];
    for entry in target.iter_mut() {
        *entry = Complex64::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5);
    }
    let mut rho = DensityMatrix::from_state(&StateVector::zero(3).unwrap());
    prepare_target_state(&mut rho, &target, 0.05, 600).unwrap();
    assert!(state_fidelity(&rho, &target) > 1.0 - 1e-6);
}

#[test]
fn many_qubit_jump_reduces_to_the_single_qubit_one() {
    use quantumforge::dissipative::{jump_to_target_state, lindblad_step_global};

    // |1> = [0, 1]; the jump is |1><0| up to a sign.
    let jump = jump_to_target_state(&[Complex64::ZERO, Complex64::ONE]).unwrap();
    assert_eq!(jump[0], Complex64::ZERO);
    assert_eq!(jump[1], Complex64::ZERO);
    assert!((jump[2].norm_sqr() - 1.0).abs() < 1e-12);
    assert_eq!(jump[3], Complex64::ZERO);

    // Applying it to |0> drives the population to |1>.
    let mut rho = one_qubit_state();
    for _ in 0..400 {
        lindblad_step_global(&mut rho, &jump, 0.05).unwrap();
    }
    assert!(rho.probability_one(0).unwrap() > 1.0 - 1e-6);
}

#[test]
fn target_population_grows_monotonically() {
    use quantumforge::dissipative::{jump_to_target_state, lindblad_step_global, state_fidelity};

    let mut target = vec![Complex64::ZERO; 8];
    let amplitude = 1.0 / 3.0f64.sqrt();
    for index in [1usize, 2, 4] {
        target[index] = Complex64::new(amplitude, 0.0);
    }
    let jump = jump_to_target_state(&target).unwrap();
    let mut rho = DensityMatrix::from_state(&StateVector::zero(3).unwrap());
    let mut previous = state_fidelity(&rho, &target);
    for _ in 0..400 {
        lindblad_step_global(&mut rho, &jump, 0.05).unwrap();
        let current = state_fidelity(&rho, &target);
        assert!(
            current >= previous - 1e-12,
            "target population must not decrease: {previous} -> {current}"
        );
        previous = current;
    }
    assert!(previous > 1.0 - 1e-6);
}

#[test]
fn orthogonal_population_decays_at_the_exact_trotter_rate() {
    use quantumforge::dissipative::{jump_to_target_state, lindblad_step_global, state_fidelity};

    // For L = |t><t_perp| the step gives
    //   p_perp' = (1 - dt/2)^2 p_perp / (1 + (dt^2/4) p_perp),
    // where the denominator is the trace renormalization. The closed form
    // (1 - dt/2)^(2n) is only the leading order.
    let mut target = vec![Complex64::ZERO; 8];
    let amplitude = 1.0 / 3.0f64.sqrt();
    for index in [1usize, 2, 4] {
        target[index] = Complex64::new(amplitude, 0.0);
    }
    let jump = jump_to_target_state(&target).unwrap();
    let mut rho = DensityMatrix::from_state(&StateVector::zero(3).unwrap());
    let dt = 0.05;
    let steps = 50usize;
    let mut expected = 1.0f64;
    for _ in 0..steps {
        lindblad_step_global(&mut rho, &jump, dt).unwrap();
        expected = (1.0 - dt / 2.0).powi(2) * expected / (1.0 + (dt * dt / 4.0) * expected);
    }
    let measured = 1.0 - state_fidelity(&rho, &target);
    assert!(
        (measured - expected).abs() < 1e-12,
        "measured {measured}, expected {expected}"
    );
}

#[test]
fn pauli_expectation_matches_the_analytic_values() {
    use quantumforge::dissipative::pauli_expectation;

    // |+i> = (|0> + i|1>)/sqrt(2): <X> = 0, <Y> = 1, <Z> = 0.
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut amplitudes = vec![Complex64::ZERO; 2];
    amplitudes[0] = Complex64::new(inv_sqrt2, 0.0);
    amplitudes[1] = Complex64::new(0.0, inv_sqrt2);
    let state = StateVector::from_amplitudes(amplitudes).unwrap();
    let rho = DensityMatrix::from_state(&state);

    assert!((pauli_expectation(&rho, &[1]) - 0.0).abs() < 1e-12);
    assert!((pauli_expectation(&rho, &[2]) - 1.0).abs() < 1e-12);
    assert!((pauli_expectation(&rho, &[3]) - 0.0).abs() < 1e-12);
}

#[test]
fn dissipative_code_space_preparation_pumps_the_code_space() {
    use quantumforge::dissipative::{code_jumps, code_space_population, lindblad_step_code};
    use quantumforge::qec::repetition_code_bitflip;

    let code = repetition_code_bitflip(3);
    // |100> is entirely outside the code space span(|000>, |111>).
    let mut amplitudes = vec![Complex64::ZERO; 8];
    amplitudes[1] = Complex64::ONE;
    let state = StateVector::from_amplitudes(amplitudes).unwrap();
    let mut rho = DensityMatrix::from_state(&state);
    assert!(code_space_population(&rho, &code).abs() < 1e-12);

    let jumps = code_jumps(&code).unwrap();
    assert_eq!(jumps.len(), code.stabilizers.len());

    let mut previous = code_space_population(&rho, &code);
    for _ in 0..400 {
        lindblad_step_code(&mut rho, &jumps, 0.05).unwrap();
        let current = code_space_population(&rho, &code);
        assert!(
            current >= previous - 1e-12,
            "code population must be monotone: {previous} -> {current}"
        );
        previous = current;
    }
    assert!(previous > 1.0 - 1e-6, "code population {previous}");
}

#[test]
fn code_space_population_is_one_on_an_encoded_state() {
    use quantumforge::dissipative::code_space_population;
    use quantumforge::qec::repetition_code_bitflip;

    let code = repetition_code_bitflip(3);
    // |000> and |111> are both in the code space.
    for index in [0usize, 7] {
        let mut amplitudes = vec![Complex64::ZERO; 8];
        amplitudes[index] = Complex64::ONE;
        let state = StateVector::from_amplitudes(amplitudes).unwrap();
        let rho = DensityMatrix::from_state(&state);
        assert!(
            (code_space_population(&rho, &code) - 1.0).abs() < 1e-12,
            "index {index}"
        );
    }
}

#[test]
fn projector_jump_only_dephases_and_does_not_pump() {
    use quantumforge::dissipative::{code_space_population, lindblad_step_global};
    use quantumforge::qec::repetition_code_bitflip;

    let code = repetition_code_bitflip(3);
    let mut amplitudes = vec![Complex64::ZERO; 8];
    amplitudes[1] = Complex64::ONE;
    let state = StateVector::from_amplitudes(amplitudes).unwrap();
    let mut rho = DensityMatrix::from_state(&state);

    // A bare projector jump (I - S)/2 preserves the code-space population.
    let projector = {
        // (I - Z0 Z1)/2 as an 8x8 matrix.
        let mut p = vec![Complex64::ZERO; 64];
        for index in 0..8usize {
            let z0 = if index & 1 != 0 { -1.0 } else { 1.0 };
            let z1 = if index & 2 != 0 { -1.0 } else { 1.0 };
            let eigenvalue = z0 * z1;
            p[index * 8 + index] = Complex64::new((1.0 - eigenvalue) / 2.0, 0.0);
        }
        p
    };
    for _ in 0..50 {
        lindblad_step_global(&mut rho, &projector, 0.05).unwrap();
    }
    // The population is unchanged by a projector jump (up to the O(dt) mix).
    assert!(code_space_population(&rho, &code) < 1e-3);
}

#[test]
fn pauli_native_step_scales_to_the_surface_code() {
    use quantumforge::dissipative::{code_space_population, prepare_code_space};
    use quantumforge::qec::surface_code_rotated;

    let code = surface_code_rotated(3);
    let mut amplitudes = vec![Complex64::ZERO; 1 << code.n_physical];
    amplitudes[1] = Complex64::ONE;
    let state = StateVector::from_amplitudes(amplitudes).unwrap();
    let mut rho = DensityMatrix::from_state(&state);
    prepare_code_space(&code, &mut rho, 0.05, 100).unwrap();
    let population = code_space_population(&rho, &code);
    assert!(population > 0.8, "surface code population {population}");
}

#[test]
fn logical_jump_selects_the_logical_state() {
    use quantumforge::dissipative::{pauli_expectation, prepare_logical_state};
    use quantumforge::qec::repetition_code_bitflip;

    let code = repetition_code_bitflip(3);
    let field = code.logical_z[0].paulis.clone();

    let mut amplitudes = vec![Complex64::ZERO; 8];
    amplitudes[1] = Complex64::ONE;
    let state = StateVector::from_amplitudes(amplitudes).unwrap();

    // Ground logical state: <Z_L> = +1.
    let mut rho = DensityMatrix::from_state(&state);
    prepare_logical_state(&code, &mut rho, 0.05, 400, 0, false).unwrap();
    let positive = pauli_expectation(&rho, &field);
    assert!(positive > 0.99, "<Z_L> = {positive}");

    // Excited logical state: <Z_L> = -1.
    let mut rho = DensityMatrix::from_state(&state);
    prepare_logical_state(&code, &mut rho, 0.05, 400, 0, true).unwrap();
    let negative = pauli_expectation(&rho, &field);
    assert!(negative < -0.99, "<Z_L> = {negative}");
}
