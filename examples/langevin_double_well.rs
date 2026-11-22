//! Noise-driven optimization: annealed Langevin against plain gradient descent.
//!
//! ```bash
//! cargo run --release --example langevin_double_well
//! ```

use quantumforge::noisy_optimizer::{LangevinOptions, annealed_langevin, gradient_descent};

fn double_well(x: &[f64]) -> f64 {
    let x = x[0];
    (x * x - 1.0).powi(2) + 0.5 * x
}

fn double_well_gradient(x: &[f64], out: &mut [f64]) {
    out[0] = 4.0 * x[0].powi(3) - 4.0 * x[0] + 0.5;
}

fn main() {
    println!("asymmetric double well f(x) = (x^2 - 1)^2 + x/2");
    println!("deep minimum near x = -1.07, shallow near x = +0.93\n");

    let (plain, plain_value) =
        gradient_descent(double_well, double_well_gradient, &[0.5], 0.01, 2_000);
    println!(
        "plain gradient descent: x = {:.4}, f = {:.4}",
        plain[0], plain_value
    );

    let trials = 400;
    let mut escapes = 0usize;
    let mut best = f64::INFINITY;
    for seed in 0..trials {
        let options = LangevinOptions {
            learning_rate: 0.01,
            steps: 2_000,
            initial_temperature: 1.0,
            final_temperature: 0.01,
            seed,
        };
        let outcome = annealed_langevin(double_well, double_well_gradient, &[0.5], &options);
        if outcome.best[0] < 0.0 {
            escapes += 1;
        }
        best = best.min(outcome.best_value);
    }
    println!("annealed Langevin: {escapes}/{trials} runs reached the deep well");
    println!("best value found: {best:.4}");

    println!("\nthe thermal kick obeys fluctuation-dissipation, sigma = sqrt(2 lr T);");
    println!("it is a resource only when the objective is barrier-limited. On a");
    println!("convex objective the same kick just adds variance.");
}
