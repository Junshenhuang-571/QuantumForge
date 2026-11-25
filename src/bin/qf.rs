//! Interactive QuantumForge REPL binary.
//!
//! Session state and command dispatch live in `crate::repl`.

use quantumforge::repl::Session;
use std::io::Write;

fn main() {
    let mut session = Session::new(4);
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║               Welcome to QuantumForge!               ║");
    println!("║         A local quantum simulator and toolkit.       ║");
    println!("║     Try: qubits 2; add H[0]; add CNOT[0,1]; draw     ║");
    println!("║     Type 'help' for commands, 'quit' to exit.        ║");
    println!("╚══════════════════════════════════════════════════════╝");
    let mut line = String::new();
    loop {
        line.clear();
        print!("qf> ");
        let _ = std::io::stdout().flush();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !session.dispatch(trimmed) {
                    break;
                }
            }
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        }
    }
    println!("bye.");
}
