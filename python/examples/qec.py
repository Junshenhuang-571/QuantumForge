"""QEC: inspect surface codes and encode a logical qubit.

Run with the quantumforge wheel installed:
    python python/examples/qec.py
"""

from quantumforge import (
    QecCode,
    encode_repetition_bitflip,
    repetition_code_bitflip,
    surface_code_planar,
    surface_code_rotated,
    StateVector,
)


def describe(name: str, code: QecCode) -> None:
    print(
        f"{name}: {code.n_physical()} physical qubits, "
        f"{code.n_logical()} logical, {len(code.stabilizers())} stabilizers"
    )


def main() -> None:
    describe("rotated surface code d=3", surface_code_rotated(3))
    # Planar codes use face stabilizers only, so a d x d planar code encodes
    # 2d - 1 logical qubits rather than one.
    describe("planar surface code d=3", surface_code_planar(3))
    describe("repetition code n=3", repetition_code_bitflip(3))

    # Encode a logical |0> with the 3-qubit repetition code.
    logical = StateVector.zero(1)
    encoded = encode_repetition_bitflip(logical, 3)
    probs = encoded.probabilities()
    print("encoded |0> -> |000> probabilities:", [round(p, 6) for p in probs])
    assert probs[0] > 0.99
    print("OK: logical |0> encoded across 3 physical qubits")


if __name__ == "__main__":
    main()
