//! QEC noise simulation and threshold estimation.
//!
//! ```bash
//! cargo run --release --example qec_threshold
//! ```
//!
//! Two studies on the same machinery:
//!
//! 1. The bit-flip repetition-code family `d = 3, 5, 7, 9` under bit-flip noise.
//!    Its threshold is exactly `p = 1/2`, so the crossing of the logical-error
//!    curves is a validation of the simulator and the estimator.
//! 2. The rotated `d = 3` surface code under depolarizing noise with noisy
//!    syndrome readout. Only one distance exists (the rotated boundary
//!    construction is verified only for `d = 3`), so this is a pseudo-threshold
//!    curve, not a finite-size-scaling threshold.

use quantumforge::qec::{
    NoiseModel, PauliNoise, bitflip_sweep, depolarizing_sweep, depolarizing_sweep_with_measurement,
    logical_error_rate, repetition_code_bitflip, surface_code_rotated, threshold_crossing,
};

fn main() {
    println!("QEC noise simulation and threshold estimation\n");

    // 1. Repetition-code family: exact threshold 1/2.
    let rates = [0.1, 0.3, 0.5, 0.7, 0.9];
    let trials = 8_000;
    let mut campaigns = Vec::new();
    println!("bit-flip repetition codes ({trials} trials per point)");
    println!("{:<6} {:>10}", "d", "campaign");
    for d in [3usize, 5, 7, 9] {
        let code = repetition_code_bitflip(d);
        let campaign = bitflip_sweep(&code, &rates, trials, 1_000 + d as u64);
        println!(
            "{d:<6} {:>10}",
            format!(
                "L({:.2})={:.4} L({:.2})={:.4}",
                rates[0],
                campaign.logical_error_rates[0],
                rates[2],
                campaign.logical_error_rates[2]
            )
        );
        campaigns.push(campaign);
    }
    match threshold_crossing(&campaigns) {
        Some(threshold) => println!("crossing estimate: p = {threshold:.4} (exact: 0.5)"),
        None => println!("no crossing found"),
    }

    // 2. Rotated d=3 surface code under depolarizing noise.
    println!("\nrotated d=3 surface code, depolarizing data + readout noise");
    let code = surface_code_rotated(3);
    println!(
        "n_physical = {}, stabilizers = {}, logicals = {}",
        code.n_physical,
        code.stabilizers.len(),
        code.logical_x.len()
    );
    let surface_rates = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2];
    let surface_trials = 4_000;
    // Perfect readout: the logical error rate scales as p^2 at low p and the
    // pseudo-threshold (L = p) is a few percent.
    let campaign =
        depolarizing_sweep_with_measurement(&code, &surface_rates, 0.0, surface_trials, 7_003);
    println!("{:<10} {:>12}", "p", "logical error");
    for (rate, logical) in campaign
        .rates
        .iter()
        .zip(campaign.logical_error_rates.iter())
    {
        println!("{rate:<10.3} {logical:>12.4}");
    }
    let pseudo = campaign
        .rates
        .iter()
        .zip(campaign.logical_error_rates.iter())
        .find(|(rate, logical)| **logical >= **rate)
        .map(|(rate, _)| *rate);
    match pseudo {
        Some(rate) => println!("pseudo-threshold (L = p), perfect readout: p ~ {rate:.3}"),
        None => println!("pseudo-threshold not reached in the scanned range"),
    }

    // Single-round noisy readout: one measurement error can cause a logical
    // failure, so the curve has a linear term and no threshold. A real
    // threshold study needs several rounds so measurement errors are corrected
    // in time as well as space.
    let noisy = depolarizing_sweep(&code, &surface_rates, surface_trials, 7_004);
    println!("\nsingle-round noisy readout (measurement = p):");
    for (rate, logical) in noisy.rates.iter().zip(noisy.logical_error_rates.iter()) {
        println!("{rate:<10.3} {logical:>12.4}");
    }

    // Sanity: no noise means no logical errors.
    let quiet = logical_error_rate(
        &code,
        &NoiseModel {
            data: PauliNoise::depolarizing(0.0),
            measurement: 0.0,
        },
        &quantumforge::qec::DecodingTable::new(&code, quantumforge::qec::Decoder::Depolarizing),
        100,
        1,
    );
    println!("logical error rate at p = 0: {quiet}");
}
