//! Integration tests for the interactive REPL binary (`src/bin/qf.rs`).
//!
//! Each test pipes a command sequence through the real `qf` binary on stdin
//! and asserts on stdout/stderr, so the interactive behaviour (live state,
//! validation, math) is locked in against regressions.

use std::io::Write;
use std::process::{Command, Stdio};

/// Run the `qf` binary with the given stdin lines; return combined stdout.
fn run_qf(lines: &[&str]) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_qf"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn qf");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        for line in lines {
            writeln!(stdin, "{line}").expect("write stdin");
        }
    }
    let output = child.wait_with_output().expect("wait qf");
    // Combine stdout + stderr: some messages (validation errors) go to stderr.
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    combined
}

#[test]
fn bell_state_and_bloch() {
    let out = run_qf(&["qubits 2", "add H[0]", "add CNOT[0,1]", "run", "quit"]);
    assert!(out.contains("|00>  +0.7071"), "Bell |00> missing:\n{out}");
    assert!(out.contains("|11>  +0.7071"), "Bell |11> missing:\n{out}");
}

#[test]
fn observables_and_bloch_vector() {
    let out = run_qf(&[
        "qubits 1",
        "add H[0]",
        "run",
        "bloch 0",
        "expect X 0",
        "expect Y 0",
        "quit",
    ]);
    assert!(
        out.contains("<X>=+1.0000 <Y>=+0.0000 <Z>=+0.0000"),
        "bloch:\n{out}"
    );
    assert!(out.contains("<X_0> = +1.000000"), "expect X:\n{out}");
    assert!(out.contains("<Y_0> = +0.000000"), "expect Y:\n{out}");
}

#[test]
fn phase_damping_decoheres_density_state() {
    let out = run_qf(&[
        "qubits 1",
        "add Ry[Pi/2,0]",
        "add PhaseDamping[0,0.5]",
        "noisy",
        "bloch 0",
        "quit",
    ]);
    assert!(out.contains("purity=0.750000"), "purity after PD:\n{out}");
    // Coherence (off-diagonal) halved: <X> drops from 1.0 to ~0.7071.
    assert!(out.contains("<X>=+0.7071"), "decohered bloch:\n{out}");
}

#[test]
fn out_of_range_op_is_rejected() {
    let out = run_qf(&["qubits 2", "add H[0]", "add X[9]", "circuit", "quit"]);
    assert!(
        out.contains("invalid op: qubit index 9 is out of range for 2 qubits"),
        "rejection message:\n{out}"
    );
    // The circuit must NOT contain the rejected op.
    assert!(out.contains("Circuit[H[0]]"), "circuit polluted:\n{out}");
    assert!(!out.contains("X[9]"), "bad op was committed:\n{out}");
}

#[test]
fn mixed_channel_and_measurement_shots() {
    let out = run_qf(&[
        "qubits 1",
        "add H[0]",
        "add AmplitudeDamping[0,0.3]",
        "add Measure[0,0]",
        "shots-noise 400 7",
        "quit",
    ]);
    // Measurement op is accepted and shots produce both outcomes.
    assert!(
        out.contains("Circuit[H[0], AmplitudeDamping[0,0.3], Measure[0,0]]"),
        "circuit:\n{out}"
    );
    assert!(out.contains("shots: 400 (seed 7)"), "shots header:\n{out}");
    assert!(
        out.contains("|0>") && out.contains("|1>"),
        "both outcomes:\n{out}"
    );
}

#[test]
fn draw_renders_circuit_diagram() {
    let out = run_qf(&["qubits 2", "add H[0]", "add CNOT[0,1]", "draw", "quit"]);
    assert!(out.contains('•'), "control dot missing:\n{out}");
    assert!(out.contains('⊕'), "target symbol missing:\n{out}");
    // Wire stays continuous (dashes); H and CNOT markers sit in their columns.
    assert!(out.contains("q0 ───H"), "H in column 0:\n{out}");
    assert!(out.contains("──────•"), "CNOT control in column 1:\n{out}");
    assert!(
        out.contains("q1 ──────────⊕"),
        "target aligned with control:\n{out}"
    );
}

#[test]
fn draw_aligns_multi_char_labels() {
    // Multi-char labels (Ry0.50, channels) occupy their own CELL; the wire
    // stays continuous and the qubit-label prefix is never clobbered.
    let out = run_qf(&[
        "qubits 2",
        "add AmplitudeDamping[1,0.3]",
        "add PhaseDamping[0,0.5]",
        "draw",
        "quit",
    ]);
    // AD on q1 (col 0), PD on q0 (col 1): each keeps its prefix and column.
    assert!(
        out.contains("q1 AD0.30"),
        "AD in column 0 keeps q1 prefix:\n{out}"
    );
    assert!(out.contains("q0 ───────PD0.50"), "PD in column 1:\n{out}");
}

#[test]
fn draw_rotations_and_swap_align() {
    let out = run_qf(&[
        "qubits 2",
        "add X[0]",
        "add Y[1]",
        "add Z[0]",
        "add SWAP[0,1]",
        "draw",
        "quit",
    ]);
    // Each gate in its own CELL column; SWAP x markers aligned in column 3.
    assert!(out.contains("q0 ───X"), "X in column 0:\n{out}");
    assert!(out.contains("Z──────x"), "Z then SWAP aligned:\n{out}");
    assert!(out.contains("q1 ──────────Y"), "Y in column 1:\n{out}");
}

#[test]
fn draw_connects_nonadjacent_two_qubit_gates() {
    // CNOT[0,3] and SWAP[1,2] on 4 qubits: markers align in their column and a
    // vertical │ connector spans the wires between control and target.
    let out = run_qf(&["qubits 4", "add CNOT[0,3]", "add SWAP[1,2]", "draw", "quit"]);
    // CNOT •/⊕ in the same column (col 0), aligned across wires.
    assert!(out.contains("q0 ───•"), "CNOT control in column 0:\n{out}");
    assert!(
        out.contains("q3 ───⊕"),
        "CNOT target aligned with control:\n{out}"
    );
    // q1 and q2 (between control and target) carry the │ connector.
    assert!(out.contains("q1 ───│"), "connector on q1:\n{out}");
    assert!(out.contains("q2 ───│"), "connector on q2:\n{out}");
    // SWAP x markers in the next column, after the connector.
    assert!(
        out.contains("│──────x"),
        "SWAP after CNOT connector:\n{out}"
    );
}

#[test]
fn gradient_adjoint_matches_parameter_shift() {
    let out = run_qf(&["qubits 1", "add Ry[0.5,0]", "gradient Z 0", "quit"]);
    // d<Z>/dθ for Ry(θ) on |0> is -sin(θ); both methods must agree.
    assert!(out.contains("adjoint=-0.479426"), "adjoint value:\n{out}");
    assert!(out.contains("shift=-0.479426"), "shift value:\n{out}");
    assert!(out.contains("✓ agree"), "agreement marker:\n{out}");
}

#[test]
fn vqe_ising_converges_to_exact_ground() {
    // 3-qubit Ising H = -Σ Z_i Z_{i+1}; exact ground energy = -2.
    let out = run_qf(&[
        "qubits 3",
        "add Ry[0.1,0]",
        "add Ry[0.1,1]",
        "add Ry[0.1,2]",
        "add CNOT[0,1]",
        "add CNOT[1,2]",
        "add Ry[0.2,0]",
        "add Ry[0.2,1]",
        "add Ry[0.2,2]",
        "vqe ising",
        "quit",
    ]);
    assert!(
        out.contains("exact ground energy = -2.000000"),
        "exact:\n{out}"
    );
    assert!(out.contains("✓ converged"), "convergence:\n{out}");
}

#[test]
fn vqe_heisenberg_converges_to_exact_ground() {
    // 2-qubit XXX Heisenberg; exact ground (singlet) energy = -3.
    let out = run_qf(&[
        "qubits 2",
        "add Ry[0.3,0]",
        "add Ry[0.3,1]",
        "add CNOT[0,1]",
        "add Ry[0.4,0]",
        "add Ry[0.4,1]",
        "vqe heisenberg",
        "quit",
    ]);
    assert!(
        out.contains("exact ground energy = -3.000000"),
        "exact:\n{out}"
    );
    assert!(out.contains("✓ converged"), "convergence:\n{out}");
}

#[test]
fn vqe_tfim_converges_to_exact_ground() {
    // 2-qubit transverse-field Ising; exact ground energy = -(1 + sqrt(2)) ≈ -2.236068.
    let out = run_qf(&[
        "qubits 2",
        "add Ry[0.2,0]",
        "add Ry[0.2,1]",
        "add CNOT[0,1]",
        "add Ry[0.3,0]",
        "add Ry[0.3,1]",
        "vqe tfim",
        "quit",
    ]);
    assert!(
        out.contains("exact ground energy = -2.236068"),
        "exact:\n{out}"
    );
    assert!(out.contains("✓ converged"), "convergence:\n{out}");
}

#[test]
fn vqe_rejects_unknown_hamiltonian() {
    let out = run_qf(&[
        "qubits 2",
        "add Ry[0.2,0]",
        "add Ry[0.2,1]",
        "vqe bogus",
        "quit",
    ]);
    assert!(
        out.contains("unknown or ill-sized hamiltonian"),
        "rejection:\n{out}"
    );
}

// Deep Ry+CNOT-ring ansatz: 3 Ry layers separated by a full CNOT ring (0-1,1-2,0-2).
fn deep_ansatz() -> Vec<&'static str> {
    vec![
        "qubits 3",
        "add Ry[0.3,0]",
        "add Ry[0.3,1]",
        "add Ry[0.3,2]",
        "add CNOT[0,1]",
        "add CNOT[1,2]",
        "add CNOT[0,2]",
        "add Ry[0.4,0]",
        "add Ry[0.4,1]",
        "add Ry[0.4,2]",
        "add CNOT[0,1]",
        "add CNOT[1,2]",
        "add CNOT[0,2]",
        "add Ry[0.5,0]",
        "add Ry[0.5,1]",
        "add Ry[0.5,2]",
    ]
}

#[test]
fn vqe_xxz_converges() {
    let mut cmds = deep_ansatz();
    cmds.extend(["vqe xxz", "quit"]);
    let out = run_qf(&cmds);
    assert!(
        out.contains("exact ground energy = -2.372281"),
        "exact:\n{out}"
    );
    assert!(out.contains("✓ converged"), "convergence:\n{out}");
}

#[test]
fn vqe_hubbard_converges() {
    let mut cmds = deep_ansatz();
    cmds.extend(["vqe hubbard", "quit"]);
    let out = run_qf(&cmds);
    assert!(
        out.contains("exact ground energy = -0.686141"),
        "exact:\n{out}"
    );
    assert!(out.contains("✓ converged"), "convergence:\n{out}");
}

#[test]
fn train_regression_fits_target() {
    // Fit cos(x+0.7): trained weight should recover 0.7 with near-zero MSE.
    let out = run_qf(&["train regression", "quit"]);
    assert!(out.contains("R² = 1.0000"), "R²:\n{out}");
    // Trained weight ≈ 0.7 (target); printed as "weight = 0.70...".
    assert!(
        out.contains("weight = 0.7") || out.contains("weight = 0.6"),
        "trained weight near 0.7:\n{out}"
    );
}

#[test]
fn train_classification_reaches_full_accuracy() {
    // Well-separated 1-D blobs must be classified perfectly.
    let out = run_qf(&["train classify", "quit"]);
    assert!(out.contains("accuracy = 100% (8/8)"), "accuracy:\n{out}");
}

#[test]
fn train_rejects_unknown_task() {
    let out = run_qf(&["train bogus", "quit"]);
    assert!(out.contains("unknown task 'bogus'"), "rejection:\n{out}");
}

#[test]
fn train_regression_custom_data() {
    // Bounded target y=x on [-1,1] must fit well (R² near 1).
    let out = run_qf(&["train regression {-1,-1, 0,0, 1,1} nfeat=1", "quit"]);
    assert!(out.contains("R² = 0.9"), "R²:\n{out}");
    assert!(!out.contains("note: R²<0"), "should not warn:\n{out}");
}

#[test]
fn train_regression_custom_data_unbounded_warns() {
    // Unbounded target y=x+1 exceeds the <Z>∈[-1,1] readout; R²<0 + note.
    let out = run_qf(&["train regression {0,1, 1,2, 2,3} nfeat=1", "quit"]);
    assert!(
        out.contains("note: R²<0"),
        "expected range-limit note:\n{out}"
    );
}

#[test]
fn train_classify_custom_data() {
    let out = run_qf(&["train classify {-1,0, -0.5,0, 0.5,1, 1,1} nfeat=1", "quit"]);
    assert!(out.contains("accuracy = 100% (4/4)"), "accuracy:\n{out}");
}

#[test]
fn train_data_evaluates_kernel_expressions() {
    // Pi/2 and Pi in the dataset must evaluate via the scalar parser.
    let out = run_qf(&["train regression {1,Pi/2, 2,Pi} nfeat=1", "quit"]);
    assert!(out.contains("2 samples"), "parsed 2 rows:\n{out}");
}

#[test]
fn train_data_rejects_bad_shape() {
    let out = run_qf(&["train regression {1,2,3} nfeat=1", "quit"]);
    assert!(
        out.contains("not a multiple of row length"),
        "shape error:\n{out}"
    );
}

#[test]
fn qnn_demo_recovers_target() {
    // Bare demo fits cos(x+0.7): trained weight ≈ 0.7, MSE → 0.
    let out = run_qf(&["qnn", "quit"]);
    assert!(out.contains("-> 0.000000"), "mse→0:\n{out}");
    assert!(out.contains("weight = 0.700000"), "weight≈0.7:\n{out}");
}

#[test]
fn qnn_custom_data_and_gd() {
    // Fit cos data (target 0): weight ≈ 0; Adam and GD both run.
    let adam = run_qf(&["qnn {-1,0.5403, 0,1, 1,0.5403}", "quit"]);
    assert!(adam.contains("Adam"), "adam label:\n{adam}");
    assert!(adam.contains("3 samples"), "parsed 3 pairs:\n{adam}");
    let gd = run_qf(&["qnn {-1,0.5403, 0,1, 1,0.5403} opt=gd", "quit"]);
    assert!(gd.contains("GD"), "gd label:\n{gd}");
    assert!(!gd.contains("Adam"), "gd not adam:\n{gd}");
}

#[test]
fn qnn_rejects_bad_shape() {
    let out = run_qf(&["qnn {1,2,3}", "quit"]);
    assert!(
        out.contains("not a multiple of row length"),
        "shape error:\n{out}"
    );
}

#[test]
fn scalar_math_commands() {
    let out = run_qf(&[
        "eval 2 + 3*Pi",
        "simplify x + x + 2 + 3",
        "diff Sin[x^2]",
        "quit",
    ]);
    assert!(out.contains("11.42477"), "eval:\n{out}");
    assert!(out.contains("5 + 2 * x"), "simplify:\n{out}");
    assert!(out.contains("2 * x * Cos[x^2]"), "diff:\n{out}");
}

#[test]
fn seeded_shots_are_reproducible() {
    let a = run_qf(&[
        "qubits 1",
        "add H[0]",
        "add Measure[0,0]",
        "shots 512 42",
        "quit",
    ]);
    let b = run_qf(&[
        "qubits 1",
        "add H[0]",
        "add Measure[0,0]",
        "shots 512 42",
        "quit",
    ]);
    // Same seed -> identical counts lines.
    let counts = |s: &str| -> Vec<String> {
        s.lines()
            .filter(|l| l.starts_with("  |"))
            .map(str::to_owned)
            .collect()
    };
    assert_eq!(counts(&a), counts(&b), "seeded shots not reproducible");
}

#[test]
fn qtm_command_runs_reversible_step() {
    let out = run_qf(&["qtm", "quit"]);
    assert!(out.contains("counter machine"), "qtm header:\n{out}");
    assert!(
        out.contains("reversibility gap 0"),
        "counter is a bijection:\n{out}"
    );
    assert!(
        out.contains("classical step: state 0 -> 1"),
        "classical toggle:\n{out}"
    );
    assert!(
        out.contains("measured internal state after 1 step = 1"),
        "measured:\n{out}"
    );
}

#[test]
fn qtm_erase_machine_uses_ancilla() {
    // The `erase` machine is non-injective on the plain register; the command
    // must reject it there and then build a valid bijection via ancilla.
    let out = run_qf(&["qtm machine=erase cells=4", "quit"]);
    assert!(
        out.contains("reversibility gap 64"),
        "erase has a gap:\n{out}"
    );
    assert!(
        out.contains("step_quantum rejected the plain register"),
        "plain register rejected:\n{out}"
    );
    assert!(
        out.contains("permutation is a bijection"),
        "ancilla fixes it:\n{out}"
    );
}

#[test]
fn mbqc_command_runs_measurement_pattern() {
    let out = run_qf(&["mbqc", "quit"]);
    assert!(
        out.contains("3-qubit cluster state"),
        "cluster header:\n{out}"
    );
    assert!(out.contains("2 edges"), "edge count:\n{out}");
    assert!(
        out.contains("corrected angles:"),
        "by-product angles:\n{out}"
    );
    assert!(out.contains("output bit ="), "output bit:\n{out}");
}

#[test]
fn anneal_command_reaches_ground_state() {
    let out = run_qf(&["anneal 20", "quit"]);
    assert!(out.contains("-(X0+X1) -> Z0Z1"), "schedule header:\n{out}");
    assert!(
        out.contains("exact ground energy    = -1.000000"),
        "exact energy:\n{out}"
    );
    assert!(out.contains("✓ adiabatic"), "adiabatic verdict:\n{out}");
}

#[test]
fn anneal_named_schedule_and_errors() {
    // Named schedule on >=2 qubits reaches the problem ground state.
    let out = run_qf(&["qubits 3", "anneal ising T=15", "quit"]);
    assert!(out.contains("-ΣX -> ising"), "named schedule:\n{out}");
    assert!(out.contains("✓ adiabatic"), "ising converged:\n{out}");
    // Unknown schedule and bad mbqc edges are rejected with guidance.
    let bad = run_qf(&["anneal bogus", "quit"]);
    assert!(
        bad.contains("unknown or ill-sized schedule"),
        "bad schedule:\n{bad}"
    );
    let bad_edge = run_qf(&["mbqc edges=0-9 qubits=3", "quit"]);
    assert!(bad_edge.contains("bad edge"), "bad edge:\n{bad_edge}");
}
