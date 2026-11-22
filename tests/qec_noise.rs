//! Noise simulation and threshold estimation for stabilizer codes.

use quantumforge::qec::{
    Decoder, DecodingTable, NoiseModel, PauliNoise, PauliString, bitflip_sweep, depolarizing_sweep,
    is_logical_error, logical_error_rate, multiply_pauli_strings, repetition_code_bitflip,
    surface_code_rotated, syndrome_of, threshold_crossing,
};

#[test]
fn decoder_corrects_every_single_qubit_error() {
    // Repetition code under bit-flip decoding: every single X error is
    // corrected, so the net is never a logical operator.
    let repetition = repetition_code_bitflip(5);
    let table = DecodingTable::new(&repetition, Decoder::BitFlip);
    for qubit in 0..5 {
        let mut error = PauliString::new(5);
        error.set(qubit, 1);
        let syndrome = syndrome_of(&error, &repetition);
        let net = multiply_pauli_strings(&error, &table.decode(syndrome));
        assert!(
            !is_logical_error(&net, &repetition),
            "single X on qubit {qubit} was not corrected"
        );
    }

    // Surface code under depolarizing decoding: every single X/Y/Z error is
    // corrected.
    let surface = surface_code_rotated(3);
    let table = DecodingTable::new(&surface, Decoder::Depolarizing);
    for qubit in 0..surface.n_physical {
        for pauli in 1..4u8 {
            let mut error = PauliString::new(surface.n_physical);
            error.set(qubit, pauli);
            let syndrome = syndrome_of(&error, &surface);
            let net = multiply_pauli_strings(&error, &table.decode(syndrome));
            assert!(
                !is_logical_error(&net, &surface),
                "single Pauli {pauli} on qubit {qubit} was not corrected"
            );
        }
    }
}

#[test]
fn identity_is_not_a_logical_error() {
    let code = surface_code_rotated(3);
    let identity = PauliString::new(code.n_physical);
    assert!(identity.commutes(&code.logical_x[0]));
    assert!(identity.commutes(&code.logical_z[0]));
    assert!(!is_logical_error(&identity, &code));
    assert_eq!(syndrome_of(&identity, &code), 0);
}

#[test]
fn quiet_rate_is_zero_and_repetition_threshold_is_one_half() {
    let surface = surface_code_rotated(3);
    let quiet = logical_error_rate(
        &surface,
        &NoiseModel {
            data: PauliNoise::depolarizing(0.0),
            measurement: 0.0,
        },
        &DecodingTable::new(&surface, Decoder::Depolarizing),
        100,
        1,
    );
    assert_eq!(quiet, 0.0);

    // The bit-flip repetition family has an exact threshold at p = 1/2.
    let rates = [0.1, 0.3, 0.5, 0.7, 0.9];
    let trials = 4_000;
    let mut campaigns = Vec::new();
    for d in [3usize, 5, 7, 9] {
        campaigns.push(bitflip_sweep(
            &repetition_code_bitflip(d),
            &rates,
            trials,
            1_000 + d as u64,
        ));
    }
    let crossing = threshold_crossing(&campaigns).expect("repetition family must cross");
    assert!(
        (crossing - 0.5).abs() < 0.05,
        "threshold crossing {crossing} vs exact 0.5"
    );
}

#[test]
fn surface_code_logical_error_grows_with_noise() {
    let surface = surface_code_rotated(3);
    let low = depolarizing_sweep(&surface, &[0.001], 4_000, 11);
    let high = depolarizing_sweep(&surface, &[0.1], 4_000, 12);
    assert!(
        high.logical_error_rates[0] > low.logical_error_rates[0],
        "logical error must grow with the physical rate: {} vs {}",
        high.logical_error_rates[0],
        low.logical_error_rates[0]
    );
}

#[test]
fn pauli_commutation_matches_the_anticommutation_parity() {
    // Independent re-derivation: two Paulis commute iff they anticommute on an
    // even number of qubits.
    let mut x = PauliString::new(4);
    x.set(0, 1);
    let mut z = PauliString::new(4);
    z.set(0, 3);
    assert!(!x.commutes(&z)); // X vs Z on one qubit: anticommute
    let mut xx = PauliString::new(4);
    xx.set(0, 1);
    xx.set(1, 1);
    assert!(xx.commutes(&xx));
    let mut xz = PauliString::new(4);
    xz.set(0, 1);
    xz.set(1, 3);
    let mut zx = PauliString::new(4);
    zx.set(0, 3);
    zx.set(1, 1);
    assert!(xz.commutes(&zx)); // two anticommuting positions: even
    let mut y = PauliString::new(4);
    y.set(0, 2);
    assert!(!xz.commutes(&y)); // X vs Y at qubit 0 only: odd
}

#[test]
fn decoding_table_handles_every_syndrome_without_panicking() {
    for code in [repetition_code_bitflip(5), surface_code_rotated(3)] {
        let table = DecodingTable::new(&code, Decoder::Depolarizing);
        let states = 1u32 << code.stabilizers.len();
        for syndrome in 0..states {
            let decoded = table.decode(syndrome);
            assert_eq!(decoded.paulis.len(), code.n_physical);
        }
    }
}
