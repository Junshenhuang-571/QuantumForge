#![no_main]

use libfuzzer_sys::fuzz_target;
use quantumforge::qasm::parse_qasm;

// OpenQASM input reaches the library from files and network sources, so the
// parser must reject malformed input rather than panic on it.
fuzz_target!(|data: &[u8]| {
    if let Ok(source) = core::str::from_utf8(data) {
        let _ = parse_qasm(source);
    }
});
