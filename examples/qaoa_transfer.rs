//! Parameter-transfer initialization with targeted-single-layer optimization
//! for QAOA MaxCut (Patel & Mishra, arXiv:2601.15760).
//!
//! Run in release mode — every objective evaluation rebuilds and runs the full
//! QAOA circuit:
//!
//! ```bash
//! cargo run --release --example qaoa_transfer
//! ```
//!
//! The pipeline: optimize a small 8-vertex donor 3-regular graph, copy its
//! angles to a larger acceptor, pick the layer whose two angles are worth
//! optimizing, and optimize only those. The script prints the transferred ratio
//! `r_n`, the targeted-single-layer ratio `r_s`, and — when
//! `compare_full_optimization` is enabled — the full re-optimization ratio
//! `r_f` for comparison.

use quantumforge::mps_qaoa::MaxCutProblem;
use quantumforge::qaoa_transfer::{
    TargetedLayerOptions, TransferOptions, run_parameter_transfer, select_target_layer_majority,
};

fn main() {
    let donor_nodes = 8;
    let acceptor_nodes = 12;

    let donor = MaxCutProblem::random_3_regular(donor_nodes, 20261117);
    let acceptor = MaxCutProblem::random_3_regular(acceptor_nodes, 20261118);

    // A small ensemble of acceptors to vote on the target layer, mirroring
    // Algorithm 1 of the paper.
    let ensemble: Vec<MaxCutProblem> = (0..8)
        .map(|seed| MaxCutProblem::random_3_regular(acceptor_nodes, 9000 + seed))
        .collect();

    let options = TransferOptions {
        depth: 8,
        donor_iterations: 40,
        targeted: TargetedLayerOptions {
            max_iterations: 200,
            ..TargetedLayerOptions::default()
        },
        compare_full_optimization: true,
        ..TransferOptions::default()
    };

    println!("QAOA parameter transfer + targeted-single-layer optimization");
    println!(
        "donor: {donor_nodes} vertices, acceptor: {acceptor_nodes} vertices, p = {}",
        options.depth
    );
    println!(
        "acceptor: {} edges, exact MaxCut = {:.4}",
        acceptor.edges.len(),
        acceptor.exact_maxcut().unwrap_or(f64::NAN)
    );

    let outcome = match run_parameter_transfer(&donor, &acceptor, &options) {
        Ok(outcome) => outcome,
        Err(error) => {
            eprintln!("parameter transfer failed: {error}");
            std::process::exit(1);
        }
    };

    let (voted_layer, votes) =
        select_target_layer_majority(&ensemble, &outcome.transferred, &options.targeted)
            .expect("layer vote failed");

    println!();
    println!("donor angles (first 4 layers):");
    for layer in 0..4.min(outcome.donor_parameters.depth()) {
        println!(
            "  layer {layer}: gamma = {:+.5}, beta = {:+.5}",
            outcome.donor_parameters.gammas[layer], outcome.donor_parameters.betas[layer]
        );
    }

    println!();
    println!("refinement (cut expectation after each targeted sweep):");
    for (sweep, energy) in outcome.refinement_energies.iter().enumerate() {
        println!("  sweep {sweep}: {energy:.5}");
    }

    println!();
    println!(
        "targeted block           : layers {}..{} (ensemble vote for a single layer: {}, votes {:?})",
        outcome.target_layer,
        outcome.optimized.depth() - 1,
        voted_layer,
        votes
    );
    println!(
        "transferred  r_n         : {:.5}",
        outcome.transferred_ratio
    );
    println!(
        "targeted     r_s         : {:.5}",
        outcome.approximation_ratio
    );
    if let Some(full) = outcome.full_optimization_ratio {
        println!("full         r_f         : {full:.5}");
        println!(
            "r_s / r_f                 : {:.2}%",
            100.0 * outcome.approximation_ratio / full
        );
    }
    println!(
        "energy / reference       : {:.5} / {:.5}",
        outcome.energy, outcome.reference
    );
}
