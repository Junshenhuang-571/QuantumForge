//! Dissipative preparation of entangled target states, with the entropy
//! trajectory of the refrigerator.
//!
//! ```bash
//! cargo run --release --example dissipative_entangled
//! ```

use quantumforge::dissipative::{jump_to_target_state, lindblad_step_global, state_fidelity};
use quantumforge::{Complex64, DensityMatrix, StateVector};

fn ghz(n: usize) -> Vec<Complex64> {
    let mut target = vec![Complex64::ZERO; 1 << n];
    let amplitude = 1.0 / 2.0f64.sqrt();
    target[0] = Complex64::new(amplitude, 0.0);
    target[(1 << n) - 1] = Complex64::new(amplitude, 0.0);
    target
}

fn w(n: usize) -> Vec<Complex64> {
    let mut target = vec![Complex64::ZERO; 1 << n];
    let amplitude = 1.0 / (n as f64).sqrt();
    for qubit in 0..n {
        target[1 << qubit] = Complex64::new(amplitude, 0.0);
    }
    target
}

fn main() {
    let cases: [(&str, usize, Vec<Complex64>); 4] = [
        ("GHZ", 3, ghz(3)),
        ("W", 3, w(3)),
        ("GHZ", 4, ghz(4)),
        ("W", 4, w(4)),
    ];

    for (label, qubits, target) in cases {
        let jump = jump_to_target_state(&target).unwrap();
        let mut rho = DensityMatrix::from_state(&StateVector::zero(qubits).unwrap());
        println!("{label}_{qubits}  target dimension {}", target.len());
        println!(
            "  {:>6} {:>14} {:>12} {:>12}",
            "step", "fidelity", "S", "purity"
        );
        let mut peak_entropy = 0.0f64;
        for step in 1..=400 {
            lindblad_step_global(&mut rho, &jump, 0.05).unwrap();
            let entropy = rho.von_neumann_entropy().unwrap();
            peak_entropy = peak_entropy.max(entropy);
            if matches!(step, 1 | 5 | 20 | 50 | 100 | 200 | 400) {
                println!(
                    "  {step:>6} {:>14.9} {:>12.6} {:>12.6}",
                    state_fidelity(&rho, &target),
                    entropy,
                    rho.purity()
                );
            }
        }
        println!(
            "  peak entropy {peak_entropy:.6}, final entropy {:.6}",
            rho.von_neumann_entropy().unwrap()
        );
        println!();
    }

    println!("the system starts and ends pure, so its net entropy change is zero;");
    println!("the bath absorbs the entropy generated on the way. The channel is a");
    println!("refrigerator: it purifies by dumping heat, exactly like amplitude");
    println!("damping, not like a unital thermalizing channel.");
}
