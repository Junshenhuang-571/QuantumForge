//! Readout-noise modeling and mitigation in a VQE measurement.
//!
//! ```bash
//! cargo run --release --example readout_vqe
//! ```

use quantumforge::readout::{ConfusionMatrix, apply_readout_error};
use quantumforge::{Circuit, Gate, ShotResult, SmallRng, StateVector};

fn expectation_of_z(counts: &std::collections::BTreeMap<Vec<u8>, usize>, shots: usize) -> f64 {
    let ones: usize = counts.get(&vec![1u8]).copied().unwrap_or(0);
    1.0 - 2.0 * ones as f64 / shots as f64
}

fn main() {
    let theta: f64 = 0.7;
    let shots = 200_000;
    let exact = theta.cos();
    println!("Ry({theta})|0> measured in Z: <Z> = cos({theta}) = {exact:.6}");
    println!("{shots} shots\n");

    let mut circuit = Circuit::new();
    circuit.push(Gate::Ry(theta), 0);
    circuit.measure(0, 0);
    let ideal = circuit
        .run_shots(&StateVector::zero(1).unwrap(), shots, 5)
        .unwrap();
    println!(
        "ideal counts            : <Z> = {:.6}",
        expectation_of_z(&ideal.counts, shots)
    );

    let matrix = ConfusionMatrix::symmetric(0.10).unwrap();
    let mut rng = SmallRng::seeded(11);
    let noisy = apply_readout_error(&ideal.counts, &[matrix], &mut rng);
    println!(
        "after 10% readout error : <Z> = {:.6}  (expected {:.6})",
        expectation_of_z(&noisy, shots),
        0.8 * exact
    );

    let mitigated = ShotResult {
        shots,
        counts: noisy,
    }
    .mitigate_readout(&[matrix])
    .unwrap();
    println!(
        "after mitigation        : <Z> = {:.6}  (distribution sums to {:.6})",
        1.0 - 2.0 * mitigated[1],
        mitigated.iter().sum::<f64>()
    );

    // Applying the inverse to counts that never had the error is a misuse: it
    // biases an unbiased estimate. Mitigation is only correct on data that
    // actually passed through the channel.
    let unchanged = ShotResult {
        shots,
        counts: ideal.counts.clone(),
    }
    .mitigate_readout(&[matrix])
    .unwrap();
    println!(
        "mitigate(ideal), wrong use: <Z> = {:.6} (biased, as expected)",
        1.0 - 2.0 * unchanged[1]
    );
}
