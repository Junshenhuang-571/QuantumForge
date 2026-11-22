//! Dissipative state preparation with a jump-operator channel.
//!
//! ```bash
//! cargo run --release --example dissipative_prep
//! ```

use quantumforge::dissipative::{fidelity, prepare_target};
use quantumforge::{Complex64, DensityMatrix, StateVector};

fn one_qubit_state() -> DensityMatrix {
    DensityMatrix::from_state(&StateVector::zero(1).unwrap())
}

fn main() {
    println!("dissipative preparation of a single-qubit target |t>");
    println!("jump L = |t><t_perp|, step dt = 0.05\n");

    let targets: [(&str, [Complex64; 2]); 3] = [
        ("|0>", [Complex64::new(1.0, 0.0), Complex64::ZERO]),
        ("|1>", [Complex64::ZERO, Complex64::new(1.0, 0.0)]),
        (
            "|+>",
            [
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        ),
    ];

    for (label, target) in targets {
        let mut rho = one_qubit_state();
        for step in 1..=400 {
            prepare_target(&mut rho, target, 0.05, 1).unwrap();
            if step == 1 || step == 10 || step == 100 || step == 400 {
                println!(
                    "  target {label:>3}  step {step:>3}  fidelity = {:.9}",
                    fidelity(&rho, target, 0)
                );
            }
        }
    }

    // The amplitude-damping channel is the |0> case with an exact (non-Trotter)
    // channel; compare the two fixed points.
    let mut rho = one_qubit_state();
    for _ in 0..400 {
        rho.apply_channel(
            quantumforge::NoiseChannel::AmplitudeDamping { gamma: 0.05 },
            0,
        )
        .unwrap();
    }
    println!(
        "\namplitude damping (gamma = 0.05, 400 steps): p1 = {:.9}",
        rho.probability_one(0).unwrap()
    );
}
