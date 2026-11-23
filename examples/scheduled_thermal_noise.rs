//! Add thermal noise to gates and to idle gaps in a caller-supplied schedule.

use quantumforge::noise_model::{OperationTiming, ThermalGateNoiseModel};
use quantumforge::{Circuit, DensityMatrix, Gate, StateVector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ideal = Circuit::new();
    ideal.push(Gate::H, 0);
    ideal.cnot(0, 1);

    // All values use the same unit as T1/T2. The CNOT starts after a real idle
    // gap, so qubit 0 and qubit 1 receive separate thermal noise channels there.
    let schedule = [
        OperationTiming {
            start: 0.0,
            duration: 0.02,
        },
        OperationTiming {
            start: 0.10,
            duration: 0.20,
        },
    ];
    let noise = ThermalGateNoiseModel::for_schedule(30.0, 20.0, 1.0)?;
    let noisy = noise.apply_scheduled(&ideal, &schedule)?;
    let initial = DensityMatrix::from_state(&StateVector::zero(2)?);
    let final_state = noisy.circuit.run_density(initial)?;

    println!(
        "Inserted {} total thermal and {} total dephasing channels",
        noisy.summary.thermal_channels, noisy.summary.dephasing_channels
    );
    println!(
        "Of those, {} thermal and {} dephasing channels model idle periods",
        noisy.summary.idle_thermal_channels, noisy.summary.idle_dephasing_channels
    );
    println!("Final density-matrix entries: {:?}", final_state.entries());
    Ok(())
}
