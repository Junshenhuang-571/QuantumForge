//! Tests for the `qaoa-transfer` REPL commands.

use quantumforge::repl::Session;

#[test]
fn transfer_runs_and_stores_state() {
    let mut session = Session::new(4);
    assert!(session.dispatch("qaoa-transfer donor=8 acceptor=10 p=3 iters=20 full=0"));
    let stored = session
        .qaoa_transfer
        .as_ref()
        .expect("the run should be stored on the session");
    assert_eq!(stored.donor.num_qubits, 8);
    assert_eq!(stored.acceptor.num_qubits, 10);
    assert_eq!(stored.outcome.optimized.depth(), 3);
    assert!(stored.outcome.approximation_ratio > 0.0);
    assert!(stored.outcome.full_optimization_ratio.is_none());
}

#[test]
fn inspect_commands_read_the_stored_run() {
    let mut session = Session::new(4);
    // Before any run the inspect commands report and do nothing.
    assert!(session.dispatch("qaoa-angles"));
    assert!(session.dispatch("qaoa-layers"));
    assert!(session.dispatch("qaoa-layer 0"));
    assert!(session.qaoa_transfer.is_none());

    assert!(session.dispatch("qaoa-transfer donor=8 acceptor=10 p=2 iters=10 full=0"));
    assert!(session.dispatch("qaoa-angles donor"));
    assert!(session.dispatch("qaoa-angles transferred"));
    assert!(session.dispatch("qaoa-angles optimized"));
    assert!(session.dispatch("qaoa-angles nonsense"));
    assert!(session.dispatch("qaoa-layers"));
    assert!(session.dispatch("qaoa-layer 1"));
    assert!(session.dispatch("qaoa-layer not-a-number"));
}

#[test]
fn full_comparison_is_opt_in() {
    let mut session = Session::new(4);
    assert!(session.dispatch("qaoa-transfer donor=8 acceptor=10 p=2 iters=10 full=1"));
    let stored = session.qaoa_transfer.as_ref().unwrap();
    assert!(stored.outcome.full_optimization_ratio.is_some());
}

#[test]
fn bad_input_leaves_previous_run_untouched() {
    let mut session = Session::new(4);
    assert!(session.dispatch("qaoa-transfer donor=8 acceptor=10 p=2 iters=10 full=0"));
    let before = session
        .qaoa_transfer
        .as_ref()
        .unwrap()
        .outcome
        .approximation_ratio;

    // Odd donor, donor larger than acceptor, and p=0 are all rejected.
    assert!(session.dispatch("qaoa-transfer donor=7"));
    assert!(session.dispatch("qaoa-transfer donor=10 acceptor=8"));
    assert!(session.dispatch("qaoa-transfer p=0"));
    assert!(session.dispatch("qaoa-transfer iters=abc"));

    let after = session
        .qaoa_transfer
        .as_ref()
        .unwrap()
        .outcome
        .approximation_ratio;
    assert_eq!(
        before, after,
        "a rejected run must not clobber the stored one"
    );
}

#[test]
fn acceptor_defaults_above_the_donor() {
    let mut session = Session::new(4);
    assert!(session.dispatch("qaoa-transfer p=2 iters=10 full=0"));
    let stored = session.qaoa_transfer.as_ref().unwrap();
    assert_eq!(stored.donor.num_qubits, 8);
    assert!(stored.acceptor.num_qubits > stored.donor.num_qubits);
}

#[test]
fn help_lists_the_new_commands() {
    let mut session = Session::new(4);
    assert!(session.dispatch("help"));
    assert!(quantumforge::repl::HELP.contains("qaoa-transfer"));
    assert!(quantumforge::repl::HELP.contains("qaoa-angles"));
    assert!(quantumforge::repl::HELP.contains("qaoa-layers"));
    assert!(quantumforge::repl::HELP.contains("qaoa-layer"));
}
