//! REPL module split into submodules.

pub mod commands;
pub mod state;

pub use commands::dispatch;
pub use state::Session;