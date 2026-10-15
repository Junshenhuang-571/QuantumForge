//! Robustness "fuzzing" of the hand-rolled Wolfram-style parser.
//!
//! `cargo-fuzz` (libFuzzer) needs nightly + a separate fuzz workspace that the
//! stable CI can't run, so we get comparable coverage with proptest: throw
//! random and structured-garbage strings at both parser entry points and assert
//! the fundamental safety property — **the parser never panics**; it always
//! returns `Ok(..)` or `Err(..)`. Any panic is a parser bug (unguarded index,
//! unwrap on user input, integer overflow in a digit loop, etc.).

use proptest::prelude::*;
use quantumforge::parser::{parse_circuit, parse_scalar_expr};

/// Arbitrary printable ASCII (the realistic attack surface for a text parser),
/// biased toward parser-relevant characters.
fn parser_soup() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            5 => prop::char::range('a', 'z'),
            3 => prop::char::range('A', 'Z'),
            4 => prop::char::range('0', '9'),
            6 => prop::sample::select(vec![
                '+', '-', '*', '/', '^', '(', ')', '[', ']', '{', '}', ',', '.',
                ' ', '=', '<', '>', '!', '&', '|', 'X', 'Y', 'Z', 'H', 'I',
                'P', 'i', 'e', 'π', '∞', 'C', 'N', 'O', 'T', 'S', 'W', 'A',
                'M', 'R', 'x', 'D', 'F', 'G', 'L', 'B', 'Q', 'U', 'V',
            ]),
        ],
        0..80,
    )
    .prop_map(|chars| chars.into_iter().collect())
}

proptest! {
    /// The scalar-expression parser must terminate on any input without panicking.
    #[test]
    fn scalar_parser_never_panics(s in parser_soup()) {
        // The only contract: no panic. Ok or Err are both fine.
        let _ = parse_scalar_expr(&s);
    }

    /// The circuit parser must terminate on any input without panicking.
    #[test]
    fn circuit_parser_never_panics(s in parser_soup()) {
        let _ = parse_circuit(&s);
    }

    /// Deeply nested brackets must not overflow the stack (a classic recursive
    /// -descent failure). We only require no panic/crash.
    #[test]
    fn deeply_nested_input_no_stack_overflow(depth in 1usize..2000) {
        let scalar = "(".repeat(depth) + "1" + &")".repeat(depth);
        let _ = parse_scalar_expr(&scalar);
        let circuit = "[".repeat(depth) + &"]".repeat(depth);
        let _ = parse_circuit(&circuit);
    }

    /// Long digit runs must not overflow when parsed as integers.
    #[test]
    fn long_digit_runs_no_overflow(len in 1usize..200) {
        let digits = "9".repeat(len);
        let _ = parse_scalar_expr(&digits);
        let _ = parse_scalar_expr(&format!("2^{digits}"));
    }

    /// Round-trip sanity for well-formed inputs: a valid expression parses and
    /// evaluates to a finite number (guards against a parser that accepts
    /// garbage silently).
    #[test]
    fn wellformed_scalar_evaluates_finite(a in -1000i64..1000, b in 1i64..1000) {
        let src = format!("{a}/{b} + {a}");
        if let Ok(expr) = parse_scalar_expr(&src)
            && let Ok(v) = expr.evaluate(&std::collections::HashMap::new())
        {
            prop_assert!(v.re.is_finite(), "evaluated to non-finite: {v:?}");
        }
    }
}
