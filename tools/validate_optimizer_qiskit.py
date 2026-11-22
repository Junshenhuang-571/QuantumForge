#!/usr/bin/env python3
"""Cross-check QuantumForge's optimizer against Qiskit's independent unitary model.

Install the optional dependency from tools/requirements-validation.txt, build the
QuantumForge CLI, then run this script with the CLI binary path. The suite uses
small deterministic OpenQASM 2 circuits and compares the input and optimized
unitary operators up to global phase.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator


def build_corpus() -> list[tuple[str, QuantumCircuit]]:
    circuits: list[tuple[str, QuantumCircuit]] = []

    cancellation = QuantumCircuit(1)
    cancellation.h(0)
    cancellation.h(0)
    cancellation.rx(math.pi / 7, 0)
    cancellation.rx(-math.pi / 7, 0)
    circuits.append(("single-qubit-cancellation", cancellation))

    fusion = QuantumCircuit(2)
    fusion.ry(math.pi / 5, 0)
    fusion.ry(math.pi / 7, 0)
    fusion.rz(-math.pi / 9, 1)
    fusion.rz(math.pi / 9, 1)
    fusion.cx(0, 1)
    fusion.cx(0, 1)
    circuits.append(("rotation-fusion-and-entangler-cancellation", fusion))

    routed = QuantumCircuit(4)
    routed.h(0)
    routed.cx(0, 3)
    routed.rz(math.pi / 4, 1)
    routed.cz(1, 3)
    routed.swap(0, 2)
    routed.t(2)
    routed.tdg(2)
    circuits.append(("nonlocal-routing-and-mixed-gates", routed))

    return circuits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        required=True,
        type=Path,
        help="path to the built quantumforge executable",
    )
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="skip the linear-topology routing pass",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"QuantumForge executable does not exist: {binary}")

    failures = 0
    for name, source_circuit in build_corpus():
        command = [
            str(binary),
            "optimize",
            str(source_circuit.num_qubits),
            "all",
            "--verify",
            "--input",
            "-",
            "--format",
            "qasm",
        ]
        if not args.no_routing:
            command.append("--line")
        completed = subprocess.run(
            command,
            input=qasm2.dumps(source_circuit),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if completed.returncode != 0:
            failures += 1
            print(f"FAIL {name}: QuantumForge exited {completed.returncode}", file=sys.stderr)
            print(completed.stderr.rstrip(), file=sys.stderr)
            continue

        try:
            optimized_circuit = qasm2.loads(completed.stdout)
            equivalent = Operator(source_circuit).equiv(Operator(optimized_circuit))
        except Exception as error:  # Qiskit's parser/operator may reject invalid output.
            failures += 1
            print(f"FAIL {name}: could not validate optimized QASM: {error}", file=sys.stderr)
            continue

        if not equivalent:
            failures += 1
            print(f"FAIL {name}: Qiskit found a unitary mismatch", file=sys.stderr)
            continue
        print(f"PASS {name}: Qiskit operators agree up to global phase")

    if failures:
        print(f"{failures} cross-tool validation case(s) failed", file=sys.stderr)
        return 1
    print("All cross-tool validation cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
