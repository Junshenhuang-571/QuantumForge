# Quantum Error Correction Codes

## Supported Codes

### Repetition Codes
- Bit-flip repetition code
- Phase-flip repetition code

### Stabilizer Codes
- **Shor's 9-qubit code** — 8 stabilizers (4 X + 4 Z)
- **Steane's 7-qubit code** — 6 stabilizers (3 X + 3 Z)

### Surface Codes
- **Planar surface code** — d=3, d=5 verified
- **Rotated surface code** — d=3, d=5 verified

## Features

- Stabilizer commutation verification
- Syndrome extraction circuits
- Recovery tables (single-error lookup)
- MWPM decoder (branch-and-bound + greedy fallback)
- End-to-end correction tests

## Verification Status

| Code | Stabilizers Commute | Syndrome Distinct | E2E Correction |
|------|---------------------|-------------------|----------------|
| Repetition (bit-flip) | ✅ | ✅ | ✅ |
| Repetition (phase-flip) | ✅ | ✅ | ✅ |
| Shor 9-qubit | ✅ | ✅ | — |
| Steane 7-qubit | ✅ | ✅ | — |
| Surface planar d=3 | ✅ | ✅ (27 syndromes) | — |
| Surface rotated d=3 | ✅ | ✅ (27 syndromes) | — |
| Surface d=5 | ✅ | — | — |

## MWPM Decoder

The MWPM (Minimum Weight Perfect Matching) decoder supports:
- Exhaustive search up to 4 errors
- Greedy/branch-and-bound fallback for larger codes
- Works with any stabilizer code

## Usage

```rust
use quantumforge::qec::*;

let code = repetition_code_bitflip(3);
let syndrome = code.syndrome(&error_state);
let recovery = code.recovery(&syndrome);
```
