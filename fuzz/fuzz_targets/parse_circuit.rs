#![no_main]

use libfuzzer_sys::fuzz_target;
use quantumforge::parse_circuit;

// The circuit DSL parser is the largest untrusted-input surface in the crate:
// it is the front end for the REPL, the Python `parse_circuit_py` binding and
// user scripts. A panic here is a reachable, user-triggered crash.
fuzz_target!(|data: &[u8]| {
    if let Ok(source) = core::str::from_utf8(data) {
        // Any outcome is fine; a panic, overflow or hang is not.
        let _ = parse_circuit(source);
    }
});
