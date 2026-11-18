//! Parameter-transfer QAOA across the graph families studied in
//! arXiv:2601.15760 (3-regular, Erdős–Rényi, Barabási–Albert, and weighted
//! variants).
//!
//! ```bash
//! cargo run --release --example qaoa_transfer_families
//! ```
//!
//! The paper reports that transfer plus targeted-single-layer optimization is
//! strongest on unweighted 3-regular graphs and degrades on weighted families.
//! This reproduces that comparison with the crate's defaults (a 3-layer block,
//! two sweeps), so the two can be compared directly.

use quantumforge::mps_qaoa::MaxCutProblem;
use quantumforge::qaoa_transfer::{TransferOptions, run_parameter_transfer};

/// Build a donor/acceptor pair from the same family and seed.
fn pair(
    family: &str,
    donor_nodes: usize,
    acceptor_nodes: usize,
    seed: u64,
) -> (MaxCutProblem, MaxCutProblem) {
    let build = |vertices: usize, salt: u64| -> MaxCutProblem {
        match family {
            "3r" => MaxCutProblem::random_3_regular(vertices, salt),
            "er" => MaxCutProblem::random_er(vertices, 0.4, salt),
            "ba" => MaxCutProblem::random_ba(vertices, 2, salt),
            "w3r" => MaxCutProblem::random_3_regular(vertices, salt).with_random_weights(
                salt ^ 0x5EED,
                0.5,
                2.0,
            ),
            "wer" => MaxCutProblem::random_er(vertices, 0.4, salt).with_random_weights(
                salt ^ 0x5EED,
                0.5,
                2.0,
            ),
            other => panic!("unknown family {other}"),
        }
    };
    (build(donor_nodes, seed), build(acceptor_nodes, seed + 1))
}

fn main() {
    let acceptor_nodes = 12;
    let options = TransferOptions {
        depth: 6,
        donor_iterations: 100,
        compare_full_optimization: true,
        ..TransferOptions::default()
    };

    println!("parameter transfer across graph families");
    println!(
        "donor 8 -> acceptor {acceptor_nodes} vertices, p={}, lr=0.05, iters=100, \
         layers=3, sweeps=2",
        options.depth
    );
    println!();
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8}",
        "family", "edges", "r_n", "r_s", "r_f", "r_s/r_f", "time"
    );

    for family in ["3r", "er", "ba", "w3r", "wer"] {
        let (donor, acceptor) = pair(family, 8, acceptor_nodes, 42);
        let start = std::time::Instant::now();
        let outcome = match run_parameter_transfer(&donor, &acceptor, &options) {
            Ok(outcome) => outcome,
            Err(error) => {
                eprintln!("{family}: {error}");
                continue;
            }
        };
        let full = outcome.full_optimization_ratio.unwrap_or(f64::NAN);
        println!(
            "{family:<6} {:>8} {:>8.4} {:>8.4} {:>8.4} {:>8.2}% {:>7.1}s",
            acceptor.edges.len(),
            outcome.transferred_ratio,
            outcome.approximation_ratio,
            full,
            100.0 * outcome.approximation_ratio / full,
            start.elapsed().as_secs_f64()
        );
    }

    println!();
    println!("weighted families are included to reproduce the paper's finding that");
    println!("parameter transfer degrades there; raise `layers`/`sweeps` for them.");
}
