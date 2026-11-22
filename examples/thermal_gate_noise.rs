//! Apply a T1/T2 thermal noise model to a gate circuit and inspect the result.

use quantumforge::noise_model::ThermalGateNoiseModel;
use quantumforge::{Circuit, DensityMatrix, Gate, StateVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ideal = Circuit::new();
    ideal.push(Gate::H, 0);
    ideal.cnot(0, 1);

    // All time values use the same unit; beta_delta is dimensionless.
    let noise = ThermalGateNoiseModel::new(
        30.0, // T1
        20.0, // T2
        1.0,  // beta * energy gap
        0.02, // single-qubit gate duration
        0.2,  // two-qubit gate duration
    )?;
    let noisy = noise.apply(&ideal);
    let initial = DensityMatrix::from_state(&StateVector::zero(2)?);
    let final_state = noisy.circuit.run_density(initial)?;

    println!(
        "Inserted {} thermal and {} dephasing channels",
        noisy.summary.thermal_channels, noisy.summary.dephasing_channels
    );
    println!("Final density-matrix entries: {:?}", final_state.entries());
    Ok(())
}
