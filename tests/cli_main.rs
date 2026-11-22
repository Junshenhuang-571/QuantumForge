//! Tests for the `main` binary (`src/bin/main.rs`), which had no test at all.
//!
//! `src/bin/main.rs` is 553 lines and sat at 0/259 covered lines. Driving it as
//! a subprocess is the only way to test a binary, so this file does that via
//! `CARGO_BIN_EXE_main`. It found two problems on the first run:
//!
//! - `shots` printed `": 200"` for a circuit with no `Measure` operations — one
//!   empty bitstring carrying every shot, because `Circuit::run_shots` reports
//!   whatever classical bits the circuit writes and the CLI added none.
//! - `demo` printed a hard-coded `QuantumForge 0.1` banner on a 0.2.9 crate.

use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_main")
}

fn run(args: &[&str]) -> (String, String, i32) {
    let output = Command::new(binary())
        .args(args)
        .output()
        .expect("spawns the quantumforge binary");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.code().unwrap_or(-1),
    )
}

#[test]
fn help_lists_the_subcommands() {
    let (stdout, _, code) = run(&["help"]);
    assert_eq!(code, 0);
    for subcommand in ["demo", "eval", "simplify", "shots"] {
        assert!(stdout.contains(subcommand), "help omits `{subcommand}`");
    }
    assert!(
        stdout.contains("Measure"),
        "help should document the mapping"
    );
}

#[test]
fn no_arguments_runs_the_demo() {
    let (stdout, _, code) = run(&[]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains(concat!("QuantumForge ", env!("CARGO_PKG_VERSION"))),
        "demo banner should carry the crate version, got:\n{stdout}"
    );
    assert!(stdout.contains("Bell state"));
}

#[test]
fn eval_prints_the_value() {
    let (stdout, _, code) = run(&["eval", "2+3*4"]);
    assert_eq!(code, 0);
    assert!(stdout.contains("14"), "got: {stdout}");
}

#[test]
fn eval_reports_unbound_symbols_instead_of_panicking() {
    let (stdout, stderr, code) = run(&["eval", "x+1"]);
    assert_eq!(code, 2, "an evaluation error should exit 2");
    assert!(stdout.is_empty());
    assert!(
        stderr.contains('x'),
        "stderr should name the symbol: {stderr}"
    );
}

#[test]
fn eval_rejects_a_zero_denominator() {
    let (_, stderr, code) = run(&["eval", "1/0"]);
    assert_eq!(code, 2);
    assert!(!stderr.is_empty());
}

#[test]
fn simplify_prints_the_simplified_expression() {
    let (stdout, _, code) = run(&["simplify", "x+x+2+3"]);
    assert_eq!(code, 0);
    assert!(stdout.contains('x'), "got: {stdout}");
    assert!(stdout.contains('5'), "got: {stdout}");
}

#[test]
fn shots_samples_every_qubit_when_the_circuit_measures_none() {
    let (stdout, stderr, code) = run(&["shots", "2", "200", "7", "Circuit[H[0],CNOT[0,1]]"]);
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains("shots: 200"), "got: {stdout}");
    // A Bell state: both qubits must appear, in both outcomes, and nothing else.
    let mut outcomes: Vec<&str> = stdout
        .lines()
        .skip(1)
        .filter_map(|line| line.split(':').next())
        .filter(|bits| !bits.is_empty())
        .collect();
    outcomes.sort_unstable();
    assert_eq!(outcomes, vec!["00", "11"], "got: {stdout}");
}

#[test]
fn shots_uses_the_circuits_own_measurement_mapping() {
    let (stdout, stderr, code) = run(&[
        "shots",
        "2",
        "200",
        "7",
        "Circuit[H[0],CNOT[0,1],Measure[0,0],Measure[1,1]]",
    ]);
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains("11:"), "got: {stdout}");
}

#[test]
fn shots_rejects_an_out_of_range_qubit() {
    let (_, stderr, code) = run(&["shots", "2", "200", "7", "Circuit[H[9]]"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("out of range"), "got: {stderr}");
}

#[test]
fn shots_requires_the_expected_arguments() {
    let (_, stderr, code) = run(&["shots", "2", "200"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("usage"), "got: {stderr}");
}

#[test]
fn unknown_command_is_rejected() {
    let (stdout, stderr, code) = run(&["frobnicate"]);
    assert_eq!(code, 2);
    assert!(stderr.contains("unknown command"), "got: {stderr}");
    assert!(
        stdout.contains("Usage"),
        "it should print the usage: {stdout}"
    );
}
