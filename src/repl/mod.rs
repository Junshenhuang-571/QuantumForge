//! Interactive REPL building blocks shared by the `qf` binary.
//!
//! The command-dispatch logic still lives in `src/bin/qf.rs`; the session
//! state and help text have been extracted here so both the binary and any
//! future front-ends (TUI, tests) can share them.

pub mod help;
pub mod state;

pub use help::HELP;
pub use state::Session;
