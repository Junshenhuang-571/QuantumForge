//! Autonomous dissipative quantum error correction.
//!
//! A jump `L_i = A_i P_-^{(i)}` per stabilizer (with `A_i` anticommuting with
//! `S_i` and commuting with the other stabilizers) pumps the register into the
//! code space. A bare projector jump `(I - S_i)/2` only dephases.
//!
//! ```bash
//! cargo run --release --example dissipative_qec
//! ```

use quantumforge::dissipative::{
    code_jumps, code_space_population, lindblad_step_code, pauli_expectation,
};
use quantumforge::qec::repetition_code_bitflip;
use quantumforge::{Complex64, DensityMatrix, StateVector};

fn main() {
    let code = repetition_code_bitflip(3);
    println!(
        "bit-flip repetition code, {} physical qubits, {} stabilizers",
        code.n_physical,
        code.stabilizers.len()
    );
    for stabilizer in &code.stabilizers {
        println!("  S = {stabilizer}");
    }
    println!("  code space = span(|000>, |111>)\n");

    // |100> is entirely outside the code space.
    let mut amplitudes = vec![Complex64::ZERO; 8];
    amplitudes[1] = Complex64::ONE;
    let state = StateVector::from_amplitudes(amplitudes).expect("valid state");
    let mut rho = DensityMatrix::from_state(&state);

    let jumps = code_jumps(&code).expect("code jumps");
    println!(
        "  {:>6} {:>16} {:>14} {:>14}",
        "step", "code population", "<Z0Z1>", "<Z1Z2>"
    );
    for step in 0..=400 {
        if matches!(step, 0 | 1 | 5 | 20 | 50 | 100 | 200 | 400) {
            let population = code_space_population(&rho, &code);
            let first = pauli_expectation(&rho, &[3, 3, 0]);
            let second = pauli_expectation(&rho, &[0, 3, 3]);
            println!("  {step:>6} {population:>16.9} {first:>14.6} {second:>14.6}");
        }
        if step < 400 {
            lindblad_step_code(&mut rho, &jumps, 0.05).expect("Lindblad step");
        }
    }

    println!("\nthe register ends in the code space; the stabilizers are pinned to +1.");
    println!("the code-space block itself is preserved by the channel, so which code");
    println!("state is reached depends on the input and the jump choice:");
    let logical_z = pauli_expectation(&rho, &[3, 0, 0]);
    println!("  <Z0> = {logical_z:.9}");
    println!("prepare_logical_state adds L = X_L (I -+ Z_L)/2 to select |0_L> or |1_L>.");
}
