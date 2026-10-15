use quantumforge::Circuit;
use quantumforge::{
    DensityMatrix, Gate, NoiseChannel, ParameterizedCircuit, QmlTrainingConfig, RotationAxis,
    ShotConfig, ShotTrainingConfig, SingleQubitObservable, SmallRng, StateVector, VariationalQnn,
};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a} differs from {b} by more than {tolerance}"
    );
}

#[test]
fn phase_gates_s_t_and_adjoints_behave_correctly() {
    // T on |1> adds phase e^{iπ/4}; T·T† = I; T⁴ = Z; S = T².
    let mut t = StateVector::zero(1).unwrap();
    t.apply(Gate::X, 0).unwrap(); // |1>
    t.apply(Gate::T, 0).unwrap();
    let amp = t.amplitudes()[1];
    close(amp.re, std::f64::consts::FRAC_1_SQRT_2, 1e-12);
    close(amp.im, std::f64::consts::FRAC_1_SQRT_2, 1e-12);

    // T then T† is the identity (|0> stays |0>).
    let mut inv = StateVector::zero(1).unwrap();
    inv.apply(Gate::T, 0).unwrap();
    inv.apply(Gate::Tdg, 0).unwrap();
    close(inv.amplitudes()[0].re, 1.0, 1e-12);
    close(inv.amplitudes()[0].im, 0.0, 1e-12);

    // T⁴ = Z: apply T four times to |1>, expect phase -1.
    let mut z = StateVector::zero(1).unwrap();
    z.apply(Gate::X, 0).unwrap();
    for _ in 0..4 {
        z.apply(Gate::T, 0).unwrap();
    }
    close(z.amplitudes()[1].re, -1.0, 1e-12);
    close(z.amplitudes()[1].im, 0.0, 1e-12);

    // S = T² : S on |1> gives phase i.
    let mut s = StateVector::zero(1).unwrap();
    s.apply(Gate::X, 0).unwrap();
    s.apply(Gate::S, 0).unwrap();
    close(s.amplitudes()[1].re, 0.0, 1e-12);
    close(s.amplitudes()[1].im, 1.0, 1e-12);
}

#[test]
fn bell_state_and_partial_trace_have_expected_statistics() {
    let mut state = StateVector::zero(2).unwrap();
    state.apply(Gate::H, 0).unwrap();
    state.apply_cnot(0, 1).unwrap();
    let probabilities = state.probabilities();
    close(probabilities[0], 0.5, 1e-14);
    close(probabilities[3], 0.5, 1e-14);
    close(probabilities[1], 0.0, 1e-14);
    close(probabilities[2], 0.0, 1e-14);

    let reduced = DensityMatrix::from_state(&state)
        .partial_trace(&[0])
        .unwrap();
    close(reduced.trace().re, 1.0, 1e-14);
    close(reduced.purity(), 0.5, 1e-14);
}

#[test]
fn amplitude_damping_preserves_trace_and_damps_excited_population() {
    let mut excited = StateVector::zero(1).unwrap();
    excited.apply(Gate::X, 0).unwrap();
    let mut density = DensityMatrix::from_state(&excited);
    density
        .apply_channel(NoiseChannel::AmplitudeDamping { gamma: 0.25 }, 0)
        .unwrap();
    close(density.probability_one(0).unwrap(), 0.75, 1e-14);
    close(density.trace().re, 1.0, 1e-14);
}

#[test]
fn shared_rotation_parameter_shift_accumulates_occurrences() {
    let mut circuit = ParameterizedCircuit::new(1);
    circuit.rotation(RotationAxis::Y, 0, 0);
    circuit.rotation(RotationAxis::Y, 0, 0);
    let theta = 0.37;
    let gradient = circuit
        .parameter_shift_gradient_for(&[theta], 1, SingleQubitObservable::Z, 0)
        .unwrap();
    close(gradient[0], -2.0 * (2.0 * theta).sin(), 1e-12);
}

#[test]
fn adjoint_gradient_matches_parameter_shift_with_inputs_and_shared_parameters() {
    let mut circuit = ParameterizedCircuit::with_inputs(2, 1);
    circuit.encode_rotation(RotationAxis::Y, 0, 0);
    circuit.rotation(RotationAxis::X, 1, 0);
    circuit.cnot(0, 1);
    circuit.rotation(RotationAxis::Z, 0, 1);
    circuit.rotation(RotationAxis::Y, 1, 0);
    let inputs = [0.43];
    let parameters = [0.31, -0.82];

    for observable in [
        SingleQubitObservable::X,
        SingleQubitObservable::Y,
        SingleQubitObservable::Z,
    ] {
        let adjoint = circuit
            .adjoint_gradient_with_inputs(&inputs, &parameters, 2, observable, 1)
            .unwrap();
        let shifted = circuit
            .parameter_shift_gradient_with_inputs(&inputs, &parameters, 2, observable, 1)
            .unwrap();
        for (actual, expected) in adjoint.iter().zip(shifted) {
            close(*actual, expected, 1e-12);
        }
    }

    let mut noisy = circuit.clone();
    noisy.channel(0, NoiseChannel::PhaseDamping { gamma: 0.2 });
    assert!(matches!(
        noisy.adjoint_gradient_with_inputs(&inputs, &parameters, 2, SingleQubitObservable::Z, 0),
        Err(quantumforge::QuantumError::OperationRequiresDensityMatrix)
    ));
}

#[test]
fn noisy_parameterized_circuit_uses_density_matrix_parameter_shift() {
    let mut circuit = ParameterizedCircuit::with_inputs(1, 1);
    circuit.encode_rotation(RotationAxis::Y, 0, 0);
    circuit.rotation(RotationAxis::X, 0, 0);
    circuit.channel(0, NoiseChannel::AmplitudeDamping { gamma: 0.23 });
    circuit.channel(0, NoiseChannel::Depolarizing { probability: 0.08 });
    let inputs = [0.39];
    let parameters = [0.71];
    assert!(matches!(
        circuit.run_with_inputs(&inputs, &parameters, 1),
        Err(quantumforge::QuantumError::OperationRequiresDensityMatrix)
    ));

    for observable in [
        SingleQubitObservable::X,
        SingleQubitObservable::Y,
        SingleQubitObservable::Z,
    ] {
        let gradient = circuit
            .parameter_shift_gradient_density_with_inputs(&inputs, &parameters, 1, observable, 0)
            .unwrap()[0];
        let epsilon = 1e-6;
        let mut plus = parameters;
        let mut minus = parameters;
        plus[0] += epsilon;
        minus[0] -= epsilon;
        let value_plus = circuit
            .expectation_density_with_inputs(&inputs, &plus, 1, observable, 0)
            .unwrap();
        let value_minus = circuit
            .expectation_density_with_inputs(&inputs, &minus, 1, observable, 0)
            .unwrap();
        close(gradient, (value_plus - value_minus) / (2.0 * epsilon), 1e-8);
    }

    let training_data = vec![(vec![0.15], 0.62), (vec![0.71], -0.08), (vec![-0.32], 0.91)];
    let (_, gradient) = circuit
        .mse_loss_and_gradient_density(&training_data, &parameters, 1, SingleQubitObservable::Z, 0)
        .unwrap();
    let epsilon = 1e-6;
    let mut plus = parameters;
    let mut minus = parameters;
    plus[0] += epsilon;
    minus[0] -= epsilon;
    let loss_plus = circuit
        .mse_loss_and_gradient_density(&training_data, &plus, 1, SingleQubitObservable::Z, 0)
        .unwrap()
        .0;
    let loss_minus = circuit
        .mse_loss_and_gradient_density(&training_data, &minus, 1, SingleQubitObservable::Z, 0)
        .unwrap()
        .0;
    close(
        gradient[0],
        (loss_plus - loss_minus) / (2.0 * epsilon),
        1e-8,
    );

    let config = quantumforge::QmlTrainingConfig {
        learning_rate: 0.04,
        epochs: 24,
        batch_size: 2,
        seed: 95,
    };
    let trained = circuit
        .train_regression_adam_density(
            &training_data,
            &parameters,
            1,
            SingleQubitObservable::Z,
            0,
            config,
        )
        .unwrap();
    let trained_again = circuit
        .train_regression_adam_density(
            &training_data,
            &parameters,
            1,
            SingleQubitObservable::Z,
            0,
            config,
        )
        .unwrap();
    assert_eq!(trained, trained_again);

    let classification_data = vec![(vec![0.15], 1.0), (vec![0.71], 0.0), (vec![-0.32], 1.0)];
    let (_, classification_gradient) = circuit
        .binary_cross_entropy_and_gradient_density(
            &classification_data,
            &parameters,
            1,
            SingleQubitObservable::Z,
            0,
        )
        .unwrap();
    let plus_loss = circuit
        .binary_cross_entropy_and_gradient_density(
            &classification_data,
            &plus,
            1,
            SingleQubitObservable::Z,
            0,
        )
        .unwrap()
        .0;
    let minus_loss = circuit
        .binary_cross_entropy_and_gradient_density(
            &classification_data,
            &minus,
            1,
            SingleQubitObservable::Z,
            0,
        )
        .unwrap()
        .0;
    close(
        classification_gradient[0],
        (plus_loss - minus_loss) / (2.0 * epsilon),
        1e-8,
    );

    let shot_config = quantumforge::ShotTrainingConfig {
        shots_per_expectation: 256,
        training: quantumforge::QmlTrainingConfig {
            learning_rate: 0.01,
            epochs: 2,
            batch_size: 2,
            seed: 881,
        },
    };
    let shot_params = circuit
        .train_regression_adam_density_shots(
            &training_data,
            &parameters,
            1,
            SingleQubitObservable::Z,
            0,
            shot_config,
        )
        .unwrap();
    assert_eq!(
        shot_params,
        circuit
            .train_regression_adam_density_shots(
                &training_data,
                &parameters,
                1,
                SingleQubitObservable::Z,
                0,
                shot_config,
            )
            .unwrap()
    );
    let shot_classifier = circuit
        .train_binary_classifier_adam_density_shots(
            &classification_data,
            &parameters,
            1,
            SingleQubitObservable::Z,
            0,
            shot_config,
        )
        .unwrap();
    assert_eq!(
        shot_classifier,
        circuit
            .train_binary_classifier_adam_density_shots(
                &classification_data,
                &parameters,
                1,
                SingleQubitObservable::Z,
                0,
                shot_config,
            )
            .unwrap()
    );
}

#[test]
fn qnn_binary_cross_entropy_gradient_matches_finite_difference() {
    let mut model = ParameterizedCircuit::with_inputs(1, 1);
    model.encode_rotation(RotationAxis::Y, 0, 0);
    model.rotation(RotationAxis::Y, 0, 0);
    let data = vec![(vec![0.2], 1.0), (vec![-0.8], 0.0)];
    let parameters = [0.43];
    let (_, gradient) = model
        .binary_cross_entropy_and_gradient(&data, &parameters, 1, SingleQubitObservable::Z, 0)
        .unwrap();
    let epsilon = 1e-6;
    let mut plus = parameters;
    let mut minus = parameters;
    plus[0] += epsilon;
    minus[0] -= epsilon;
    let loss_plus = model
        .binary_cross_entropy_and_gradient(&data, &plus, 1, SingleQubitObservable::Z, 0)
        .unwrap()
        .0;
    let loss_minus = model
        .binary_cross_entropy_and_gradient(&data, &minus, 1, SingleQubitObservable::Z, 0)
        .unwrap()
        .0;
    close(
        gradient[0],
        (loss_plus - loss_minus) / (2.0 * epsilon),
        1e-6,
    );
    let probability = model
        .binary_class_probability(&[0.2], &parameters, 1, SingleQubitObservable::Z, 0)
        .unwrap();
    assert!((0.0..1.0).contains(&probability));
    assert!(
        model
            .binary_classify(&[0.2], &parameters, 1, SingleQubitObservable::Z, 0,)
            .is_ok()
    );
}

#[test]
fn qnn_binary_classifier_training_reduces_cross_entropy() {
    let mut model = ParameterizedCircuit::with_inputs(1, 1);
    model.encode_rotation(RotationAxis::Y, 0, 0);
    model.rotation(RotationAxis::Y, 0, 0);
    let data = vec![(vec![0.0], 0.0), (vec![std::f64::consts::PI], 1.0)];
    let initial_parameters = [0.2];
    let initial_loss = model
        .binary_cross_entropy_and_gradient(
            &data,
            &initial_parameters,
            1,
            SingleQubitObservable::Z,
            0,
        )
        .unwrap()
        .0;
    let trained = model
        .train_binary_classifier_adam(
            &data,
            &initial_parameters,
            1,
            SingleQubitObservable::Z,
            0,
            QmlTrainingConfig {
                learning_rate: 0.08,
                epochs: 30,
                batch_size: 2,
                seed: 17,
            },
        )
        .unwrap();
    let final_loss = model
        .binary_cross_entropy_and_gradient(&data, &trained, 1, SingleQubitObservable::Z, 0)
        .unwrap()
        .0;
    assert!(final_loss < initial_loss, "{initial_loss} -> {final_loss}");
}

#[test]
fn mini_batch_qnn_training_reduces_loss() {
    let qnn = VariationalQnn;
    let target_weight: f64 = 0.7;
    let data: Vec<(f64, f64)> = [-1.0, -0.5, 0.0, 0.5, 1.0]
        .into_iter()
        .map(|x| (x, (x + target_weight).cos()))
        .collect();
    let initial = -1.2;
    let initial_loss = qnn.loss_and_gradient(&data, initial).unwrap().0;
    let trained = qnn.train_adam(&data, initial, 0.05, 100, 2, 42).unwrap();
    let final_loss = qnn.loss_and_gradient(&data, trained).unwrap().0;
    assert!(
        final_loss < initial_loss,
        "training loss did not improve: {initial_loss} -> {final_loss}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ry_matches_cpu_reference() {
    let mut gpu = StateVector::zero(5).unwrap();
    gpu.apply(Gate::H, 0).unwrap();
    gpu.apply(Gate::Ry(-0.2), 3).unwrap();
    let mut cpu = gpu.clone();
    let backend = quantumforge::cuda::CudaBackend::new(0).unwrap();
    let gates = [
        (RotationAxis::X, 2, 0.73),
        (RotationAxis::Y, 4, -0.2),
        (RotationAxis::Z, 1, 1.1),
    ];
    backend.apply_rotation_chain(&mut gpu, &gates).unwrap();
    cpu.apply(Gate::Rx(0.73), 2).unwrap();
    cpu.apply(Gate::Ry(-0.2), 4).unwrap();
    cpu.apply(Gate::Rz(1.1), 1).unwrap();
    for (actual, expected) in gpu.amplitudes().iter().zip(cpu.amplitudes()) {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_device_state_stays_resident_between_dispatches() {
    let mut initial = StateVector::zero(4).unwrap();
    initial.apply(Gate::H, 0).unwrap();
    let mut expected = initial.clone();
    let backend = quantumforge::cuda::CudaBackend::new(0).unwrap();
    let mut device = backend.upload(&initial).unwrap();
    device.apply_rotation(RotationAxis::X, 2, 0.31).unwrap();
    device
        .apply_rotation_chain(&[(RotationAxis::Y, 1, -0.83), (RotationAxis::Z, 3, 1.27)])
        .unwrap();
    drop(backend);

    expected.apply(Gate::Rx(0.31), 2).unwrap();
    expected.apply(Gate::Ry(-0.83), 1).unwrap();
    expected.apply(Gate::Rz(1.27), 3).unwrap();
    let actual = device.download().unwrap();
    for (actual, expected) in actual.amplitudes().iter().zip(expected.amplitudes()) {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_entangling_gates_match_cpu_reference() {
    let mut initial = StateVector::zero(5).unwrap();
    for qubit in 0..5 {
        initial.apply(Gate::H, qubit).unwrap();
    }
    initial.apply(Gate::Rz(0.37), 2).unwrap();
    let mut expected = initial.clone();
    let backend = quantumforge::cuda::CudaBackend::new(0).unwrap();
    let mut device = backend.upload(&initial).unwrap();
    device.apply_cnot(0, 4).unwrap();
    expected.apply_cnot(0, 4).unwrap();
    for (actual, expected) in device
        .download()
        .unwrap()
        .amplitudes()
        .iter()
        .zip(expected.amplitudes())
    {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
    device.apply_cz(1, 3).unwrap();
    expected.apply_cz(1, 3).unwrap();
    for (actual, expected) in device
        .download()
        .unwrap()
        .amplitudes()
        .iter()
        .zip(expected.amplitudes())
    {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
    device.apply_swap(0, 2).unwrap();

    expected.apply_swap(0, 2).unwrap();
    let actual = device.download().unwrap();
    for (actual, expected) in actual.amplitudes().iter().zip(expected.amplitudes()) {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_executes_the_unitary_circuit_api() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.push(Gate::X, 1);
    circuit.push(Gate::Y, 2);
    circuit.push(Gate::Z, 3);
    circuit.push(Gate::Rx(0.19), 0);
    circuit.push(Gate::Ry(-0.57), 1);
    circuit.push(Gate::Rz(1.03), 2);
    circuit.cnot(0, 3);
    circuit.cz(1, 2);
    circuit.swap(0, 1);
    let initial = StateVector::zero(4).unwrap();
    let expected = circuit.run(initial.clone()).unwrap();
    let actual = quantumforge::cuda::CudaBackend::new(0)
        .unwrap()
        .run_circuit(initial, &circuit)
        .unwrap();
    for (actual, expected) in actual.amplitudes().iter().zip(expected.amplitudes()) {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_circuit_matches_independent_cpu_runs() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.push(Gate::Rx(0.19), 2);
    circuit.push(Gate::Ry(-0.53), 1);
    circuit.cnot(0, 2);
    circuit.cz(1, 2);
    circuit.swap(0, 1);
    circuit.push(Gate::Rz(std::f64::consts::FRAC_PI_2), 2);

    let mut inputs = Vec::new();
    inputs.push(StateVector::zero(3).unwrap());
    let mut second = StateVector::zero(3).unwrap();
    second.apply(Gate::X, 0).unwrap();
    second.apply(Gate::Ry(0.31), 2).unwrap();
    inputs.push(second);
    let mut third = StateVector::zero(3).unwrap();
    third.apply(Gate::H, 1).unwrap();
    third
        .apply(Gate::Rz(std::f64::consts::FRAC_PI_2), 2)
        .unwrap();
    inputs.push(third);
    let mut fourth = StateVector::zero(3).unwrap();
    fourth.apply(Gate::X, 1).unwrap();
    fourth.apply(Gate::H, 2).unwrap();
    inputs.push(fourth);

    let expected: Vec<_> = inputs
        .iter()
        .map(|state| circuit.run(state.clone()).unwrap())
        .collect();
    let backend = quantumforge::cuda::CudaBackend::new(0).unwrap();
    assert!(matches!(
        backend.upload_batch(&[]),
        Err(quantumforge::cuda::CudaError::EmptyBatch)
    ));
    assert!(matches!(
        backend.upload_batch(&[StateVector::zero(2).unwrap(), StateVector::zero(3).unwrap(),]),
        Err(quantumforge::cuda::CudaError::IncompatibleBatch)
    ));
    let actual = backend.run_circuit_batch(&inputs, &circuit).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual_state, expected_state) in actual.iter().zip(expected.iter()) {
        for (actual, expected) in actual_state
            .amplitudes()
            .iter()
            .zip(expected_state.amplitudes())
        {
            close(actual.re, expected.re, 1e-12);
            close(actual.im, expected.im, 1e-12);
        }
    }

    let mut measurement_inputs = inputs.clone();
    for state in &mut measurement_inputs {
        state.apply(Gate::H, 0).unwrap();
    }
    let mut cpu_measurement_states = measurement_inputs.clone();
    let mut cpu_rng = SmallRng::seeded(7721);
    let expected_outcomes: Vec<_> = cpu_measurement_states
        .iter_mut()
        .map(|state| state.measure(0, &mut cpu_rng).unwrap())
        .collect();
    let mut gpu_rng = SmallRng::seeded(7721);
    let mut device_batch = backend.upload_batch(&measurement_inputs).unwrap();
    let actual_outcomes = device_batch.measure(0, &mut gpu_rng).unwrap();
    assert_eq!(actual_outcomes, expected_outcomes);
    let collapsed = device_batch.download().unwrap();
    for (actual_state, expected_state) in collapsed.iter().zip(cpu_measurement_states.iter()) {
        for (actual, expected) in actual_state
            .amplitudes()
            .iter()
            .zip(expected_state.amplitudes())
        {
            close(actual.re, expected.re, 1e-12);
            close(actual.im, expected.im, 1e-12);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_measurement_matches_the_cpu_reference() {
    let mut initial = StateVector::zero(2).unwrap();
    initial.apply(Gate::H, 1).unwrap();
    let mut cpu = initial.clone();
    let mut cpu_rng = SmallRng::seeded(342);
    let mut gpu_rng = SmallRng::seeded(342);
    let expected = cpu.measure(1, &mut cpu_rng).unwrap();

    let backend = quantumforge::cuda::CudaBackend::new(0).unwrap();
    let mut device = backend.upload(&initial).unwrap();
    let actual = device.measure(1, &mut gpu_rng).unwrap();
    assert_eq!(actual, expected);
    let collapsed = device.download().unwrap();
    for (actual, expected) in collapsed.amplitudes().iter().zip(cpu.amplitudes()) {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_resident_density_measurement_matches_cpu_collapse() {
    let mut state = StateVector::zero(3).unwrap();
    state.apply(Gate::H, 0).unwrap();
    state.apply(Gate::Ry(0.71), 1).unwrap();
    state.apply_cnot(1, 2).unwrap();
    let initial = DensityMatrix::from_state(&state);
    let mut cpu = initial.clone();
    let mut cpu_rng = SmallRng::seeded(8831);
    let mut gpu_rng = SmallRng::seeded(8831);
    let expected = cpu.measure(1, &mut cpu_rng).unwrap();

    let backend = quantumforge::cuda::CudaBackend::new(0).unwrap();
    let mut device = backend.upload_density(&initial).unwrap();
    let actual = device.measure(1, &mut gpu_rng).unwrap();
    assert_eq!(actual, expected);
    let collapsed = device.download().unwrap();
    for (actual, expected) in collapsed.entries().iter().zip(cpu.entries()) {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
    close(collapsed.trace().re, 1.0, 1e-12);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_runs_seeded_measurement_and_feedforward_circuits_on_device() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.push(Gate::H, 1);
    circuit.measure(0, 0);
    circuit.if_classical_bit(0, 1, Gate::X, 2);
    circuit.measure(1, 1);
    circuit.measure(2, 2);

    let initial = StateVector::zero(3).unwrap();
    let backend = quantumforge::cuda::CudaBackend::new(0).unwrap();
    let result = backend
        .run_circuit_shots(&initial, &circuit, 64, 721)
        .unwrap();
    let batched = backend
        .run_circuit_shots_batched(&initial, &circuit, 64, 721)
        .unwrap();
    let cpu_reference = circuit.run_shots(&initial, 64, 721).unwrap();
    assert_eq!(result, cpu_reference);
    assert_eq!(batched, cpu_reference);
    assert_eq!(result.shots, 64);
    assert_eq!(result.counts.values().sum::<usize>(), 64);
    assert!(result.counts.contains_key(&vec![0, 0, 0]));
    assert!(result.counts.contains_key(&vec![1, 1, 1]));
    assert!(result.counts.contains_key(&vec![0, 1, 0]));
    assert!(result.counts.contains_key(&vec![1, 0, 1]));
    assert_eq!(result.counts.len(), 4);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_batched_noisy_shots_match_cpu_kraus_trajectories() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.push(Gate::Ry(0.54), 1);
    circuit.cnot(0, 1);
    circuit.channel(0, NoiseChannel::AmplitudeDamping { gamma: 0.23 });
    circuit.channel(1, NoiseChannel::PhaseDamping { gamma: 0.13 });
    circuit.channel(0, NoiseChannel::Depolarizing { probability: 0.17 });
    circuit.measure(0, 0);
    circuit.if_classical_bit(0, 1, Gate::X, 2);
    circuit.measure(2, 1);
    circuit.measure(1, 2);

    let initial = StateVector::zero(3).unwrap();
    let expected = circuit.run_shots_with_noise(&initial, 256, 9981).unwrap();
    let actual = quantumforge::cuda::CudaBackend::new(0)
        .unwrap()
        .run_circuit_shots_batched(&initial, &circuit, 256, 9981)
        .unwrap();
    assert_eq!(actual, expected);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_density_circuit_matches_cpu_gates_and_noise_channels() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.push(Gate::Rx(0.37), 1);
    circuit.push(Gate::Ry(-0.81), 2);
    circuit.push(Gate::Y, 1);
    circuit.push(Gate::Z, 2);
    circuit.push(Gate::Rz(-0.44), 0);
    circuit.cnot(0, 2);
    circuit.cz(1, 2);
    circuit.swap(0, 1);
    circuit.channel(0, NoiseChannel::AmplitudeDamping { gamma: 0.17 });
    circuit.channel(1, NoiseChannel::PhaseDamping { gamma: 0.29 });
    circuit.channel(2, NoiseChannel::Depolarizing { probability: 0.13 });

    let initial = DensityMatrix::from_state(&StateVector::zero(3).unwrap());
    let expected = circuit.run_density(initial.clone()).unwrap();
    let actual = quantumforge::cuda::CudaBackend::new(0)
        .unwrap()
        .run_density_circuit(&initial, &circuit)
        .unwrap();
    for (actual, expected) in actual.entries().iter().zip(expected.entries()) {
        close(actual.re, expected.re, 1e-12);
        close(actual.im, expected.im, 1e-12);
    }
    close(actual.trace().re, 1.0, 1e-12);
    close(actual.purity(), expected.purity(), 1e-12);
}

#[test]
fn seeded_measurement_is_reproducible() {
    let mut a = StateVector::zero(1).unwrap();
    a.apply(Gate::H, 0).unwrap();
    let mut b = a.clone();
    let mut rng_a = SmallRng::seeded(1234);
    let mut rng_b = SmallRng::seeded(1234);
    assert_eq!(
        a.measure(0, &mut rng_a).unwrap(),
        b.measure(0, &mut rng_b).unwrap()
    );
}

#[test]
fn circuit_shots_record_bits_and_execute_classical_conditions() {
    let mut circuit = Circuit::new();
    circuit.push(Gate::H, 0);
    circuit.measure(0, 0);
    circuit.if_classical_bit(0, 1, Gate::X, 1);
    circuit.measure(1, 1);

    let initial = StateVector::zero(2).unwrap();
    let result = circuit.run_shots(&initial, 512, 991).unwrap();
    let repeated = circuit.run_shots(&initial, 512, 991).unwrap();
    assert_eq!(result, repeated);
    assert_eq!(result.counts.values().sum::<usize>(), 512);
    assert_eq!(result.counts.len(), 2);
    assert!(result.counts.contains_key(&vec![0, 0]));
    assert!(result.counts.contains_key(&vec![1, 1]));
    assert!(matches!(
        circuit.run(initial),
        Err(quantumforge::QuantumError::OperationRequiresShotExecution)
    ));
}

#[test]
fn noisy_shots_sample_kraus_trajectories_against_density_matrix_reference() {
    let cases = [
        (Gate::X, NoiseChannel::AmplitudeDamping { gamma: 0.31 }),
        (Gate::H, NoiseChannel::PhaseDamping { gamma: 0.73 }),
        (Gate::I, NoiseChannel::Depolarizing { probability: 0.6 }),
    ];
    for (initial_gate, channel) in cases {
        let mut initial = StateVector::zero(2).unwrap();
        initial.apply(initial_gate, 0).unwrap();
        let mut noise_circuit = Circuit::new();
        noise_circuit.channel(0, channel);
        let exact = noise_circuit
            .run_density(DensityMatrix::from_state(&initial))
            .unwrap()
            .probability_one(0)
            .unwrap();
        noise_circuit.measure(0, 0);
        noise_circuit.if_classical_bit(0, 1, Gate::X, 1);
        noise_circuit.measure(1, 1);

        assert!(matches!(
            noise_circuit.run_shots(&initial, 1, 5),
            Err(quantumforge::QuantumError::OperationRequiresDensityMatrix)
        ));
        let sampled = noise_circuit
            .run_shots_with_noise(&initial, 12_000, 5)
            .unwrap();
        let repeated = noise_circuit
            .run_shots_with_noise(&initial, 12_000, 5)
            .unwrap();
        assert_eq!(sampled, repeated);
        let ones: usize = sampled
            .counts
            .iter()
            .filter(|(bits, _)| bits[0] == 1)
            .map(|(_, count)| count)
            .sum();
        close(ones as f64 / sampled.shots as f64, exact, 0.025);
        assert!(sampled.counts.keys().all(|bits| bits[0] == bits[1]));
    }
}

#[test]
fn finite_shot_qml_samples_y_and_training_is_reproducible() {
    let mut y_eigenstate = ParameterizedCircuit::new(0);
    y_eigenstate.push(Gate::Rx(-std::f64::consts::FRAC_PI_2), 0);
    let estimate = y_eigenstate
        .expectation_shots(
            &[],
            1,
            SingleQubitObservable::Y,
            0,
            ShotConfig {
                shots: 128,
                seed: 7,
            },
        )
        .unwrap();
    close(estimate, 1.0, 1e-14);

    let mut model = ParameterizedCircuit::new(1);
    model.rotation(RotationAxis::Y, 0, 0);
    let data = vec![(vec![], 0.5)];
    let training = ShotTrainingConfig {
        shots_per_expectation: 64,
        training: QmlTrainingConfig {
            learning_rate: 0.01,
            epochs: 1,
            batch_size: 1,
            seed: 42,
        },
    };
    let first = model
        .train_adam_shots(&data, &[0.2], 1, SingleQubitObservable::Z, 0, training)
        .unwrap();
    let second = model
        .train_adam_shots(&data, &[0.2], 1, SingleQubitObservable::Z, 0, training)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 1);

    let classifier_data = vec![(vec![], 1.0)];
    let classification = model
        .train_binary_classifier_adam_shots(
            &classifier_data,
            &[0.2],
            1,
            SingleQubitObservable::Z,
            0,
            ShotTrainingConfig {
                shots_per_expectation: 64,
                training: QmlTrainingConfig {
                    learning_rate: 0.01,
                    epochs: 1,
                    batch_size: 1,
                    seed: 84,
                },
            },
        )
        .unwrap();
    assert_eq!(classification.len(), 1);
    assert!(classification[0].is_finite());
}
