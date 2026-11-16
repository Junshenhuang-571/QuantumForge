//! MPS-based QAOA for MaxCut.
//!
//! Runs the QAOA circuit on a matrix product state with a bounded bond
//! dimension instead of a `2^n` state vector, then sweeps the bond dimension
//! to show the accuracy/memory tradeoff. The cost layer is applied as exact
//! bond-2 MPO gates and the cost expectation is a transfer contraction, so
//! nothing expands the state vector.
//!
//! Run in release mode (the compression is ~50x faster than a debug build):
//!
//! ```text
//! cargo run --release --example mps_qaoa
//! ```

use quantumforge::mps_qaoa::{MaxCutProblem, MpsQaoaConfig, find_angles_mps, run_maxcut_mps};

fn main() {
    let n = 20;
    let problem = MaxCutProblem::random_3_regular(n, 42);
    let best = problem.exact_maxcut();
    println!(
        "3-regular MaxCut instance: {n} vertices, {} edges, total weight {:.1}",
        problem.edges.len(),
        problem.total_weight()
    );
    match best {
        Some(b) => println!("exact MaxCut (brute force): {b:.1}"),
        None => println!("exact MaxCut: skipped (instance too large to brute force)"),
    }

    // --- Find angles on the MPS -------------------------------------------------
    let config = MpsQaoaConfig {
        layers: 2,
        max_bond: 16,
        truncation_tol: 1e-12,
        optimizer_steps: 25,
        restarts: 2,
        seed: 7,
    };
    let found = find_angles_mps(&problem, &config).expect("angle finding must succeed");
    println!(
        "\nfound angles: betas = {:.4?}, gammas = {:.4?}",
        found.betas, found.gammas
    );
    println!(
        "  expected cut = {:.4} (ratio {:.4} of exact)",
        found.energy,
        found.approximation_ratio(&problem)
    );

    // --- Bond-dimension sweep at the found angles ------------------------------
    println!("\nchi   expected cut   ratio    max bond   time");
    for &chi in &[4usize, 8, 16, 32] {
        let outcome = run_maxcut_mps(&problem, &found.betas, &found.gammas, chi, 1e-12)
            .expect("run must succeed");
        println!(
            "{chi:4}  {:>12.4}  {:>6.4}  {:>9}  {:?}",
            outcome.energy,
            outcome.approximation_ratio(&problem),
            outcome.max_bond_reached,
            outcome.elapsed
        );
    }

    println!(
        "\nnote: the bond cap is respected at every chi; larger chi keeps more of the\n\
         entangled QAOA state and converges to the exact state-vector result.\n\
         The state vector itself is never formed."
    );
}
