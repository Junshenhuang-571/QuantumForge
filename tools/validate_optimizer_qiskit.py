#!/usr/bin/env python3
"""Cross-check QuantumForge's optimizer against Qiskit's independent unitary model.

Install the optional dependency from tools/requirements-validation.txt, build the
QuantumForge CLI, then run this script with the CLI binary path. The suite uses
small deterministic OpenQASM 2 circuits and compares the input and optimized
unitary operators up to global phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap


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

    # Seeded synthetic circuits broaden independent equivalence coverage and
    # make relative compiler metrics repeatable. These are not a substitute for
    # running a named public benchmark suite.
    rng = random.Random(20261123)
    one_qubit_gates = ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz")
    angles = (math.pi / 2, math.pi / 3, math.pi / 5, -math.pi / 7)
    for width in (2, 3, 4, 5):
        for case in range(4):
            circuit = QuantumCircuit(width)
            for _ in range(8 + case * 2):
                if rng.random() < 0.65:
                    name = rng.choice(one_qubit_gates)
                    qubit = rng.randrange(width)
                    if name in ("rx", "ry", "rz"):
                        getattr(circuit, name)(rng.choice(angles), qubit)
                    else:
                        getattr(circuit, name)(qubit)
                else:
                    first, second = rng.sample(range(width), 2)
                    getattr(circuit, rng.choice(("cx", "cz", "swap")))(first, second)
            circuits.append((f"seeded-random-{width}q-{case}", circuit))

    return circuits


def resource_metrics(circuit: QuantumCircuit) -> dict[str, int]:
    counts = circuit.count_ops()
    return {
        "operations": int(sum(counts.values())),
        "two_qubit_gates": int(sum(counts.get(name, 0) for name in ("cx", "cz", "swap"))),
        "cx_gates": int(counts.get("cx", 0)),
        "depth": int(circuit.depth() or 0),
    }


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
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        help="write a resource and compile-time comparison against Qiskit",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of compile-time trials per case (default: 3)",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"QuantumForge executable does not exist: {binary}")
    if args.repeats < 1 or args.repeats > 25:
        parser.error("--repeats must be between 1 and 25")

    repository_root = Path(__file__).resolve().parent.parent
    manifest = tomllib.loads((repository_root / "Cargo.toml").read_text(encoding="utf-8"))
    revision = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    source_revision = revision.stdout.strip() if revision.returncode == 0 else None

    failures = 0
    benchmark_rows = []
    basis_gates = ["id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cz", "swap"]
    for name, source_circuit in build_corpus():
        source_qasm = qasm2.dumps(source_circuit)
        command = [
            str(binary),
            "optimize",
            str(source_circuit.num_qubits),
            "all",
            "--input",
            "-",
            "--format",
            "qasm",
        ]
        if not args.no_routing:
            command.append("--line")
        qf_compile_trials = []
        completed = None
        for _ in range(args.repeats):
            qf_started = time.perf_counter()
            completed = subprocess.run(
                command,
                input=source_qasm,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            qf_compile_trials.append((time.perf_counter() - qf_started) * 1000.0)
            if completed.returncode != 0:
                break
        assert completed is not None
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
        qiskit_compile_trials = []
        qiskit_circuit = None
        for _ in range(args.repeats):
            qiskit_started = time.perf_counter()
            qiskit_circuit = transpile(
                source_circuit,
                basis_gates=basis_gates,
                coupling_map=(
                    CouplingMap.from_line(source_circuit.num_qubits)
                    if not args.no_routing and source_circuit.num_qubits > 1
                    else None
                ),
                # Level 1, not 3: Qiskit 2.6.x drops `swap` gates outright at
                # optimization levels 2 and 3, which would make every routed
                # comparison meaningless.
                optimization_level=1,
                seed_transpiler=20261123,
            )
            qiskit_compile_trials.append((time.perf_counter() - qiskit_started) * 1000.0)
        assert qiskit_circuit is not None

        # Sanity-check the reference itself on an UNROUTED transpile. With a
        # coupling map Qiskit returns a circuit expressed in a permuted qubit
        # order (`TranspileLayout.final_index_layout()` is non-trivial), so
        # `Operator(routed)` is not directly comparable to `Operator(source)`.
        # Without routing the layout is trivial and the comparison is exact.
        qiskit_baseline = transpile(
            source_circuit,
            basis_gates=basis_gates,
            optimization_level=1,
            seed_transpiler=20261123,
        )
        if not Operator(source_circuit).equiv(Operator(qiskit_baseline)):
            failures += 1
            print(f"FAIL {name}: Qiskit baseline changed the unitary", file=sys.stderr)
            continue

        qf_metrics = resource_metrics(optimized_circuit)
        qiskit_metrics = resource_metrics(qiskit_circuit)
        qf_median_ms = statistics.median(qf_compile_trials)
        qiskit_median_ms = statistics.median(qiskit_compile_trials)
        qf_qasm = completed.stdout
        qiskit_qasm = qasm2.dumps(qiskit_circuit)
        benchmark_rows.append(
            {
                "name": name,
                "qubits": source_circuit.num_qubits,
                "quantumforge": {
                    **qf_metrics,
                    "compile_ms_median": round(qf_median_ms, 3),
                    "compile_ms_min": round(min(qf_compile_trials), 3),
                    "compile_ms_max": round(max(qf_compile_trials), 3),
                    "compile_ms_trials": [round(value, 3) for value in qf_compile_trials],
                },
                "qiskit": {
                    **qiskit_metrics,
                    "compile_ms_median": round(qiskit_median_ms, 3),
                    "compile_ms_min": round(min(qiskit_compile_trials), 3),
                    "compile_ms_max": round(max(qiskit_compile_trials), 3),
                    "compile_ms_trials": [round(value, 3) for value in qiskit_compile_trials],
                },
                "artifacts": {
                    "input_qasm": source_qasm,
                    "quantumforge_output_qasm": qf_qasm,
                    "qiskit_output_qasm": qiskit_qasm,
                    "sha256": {
                        "input_qasm": hashlib.sha256(source_qasm.encode()).hexdigest(),
                        "quantumforge_output_qasm": hashlib.sha256(qf_qasm.encode()).hexdigest(),
                        "qiskit_output_qasm": hashlib.sha256(qiskit_qasm.encode()).hexdigest(),
                    },
                },
                "two_qubit_gate_delta_qf_minus_qiskit": (
                    qf_metrics["two_qubit_gates"] - qiskit_metrics["two_qubit_gates"]
                ),
                "depth_delta_qf_minus_qiskit": qf_metrics["depth"] - qiskit_metrics["depth"],
                "equivalent_to_input": True,
            }
        )
        print(
            f"PASS {name}: equivalent; two-qubit gates QF={qf_metrics['two_qubit_gates']} "
            f"Qiskit={qiskit_metrics['two_qubit_gates']}; depth QF={qf_metrics['depth']} "
            f"Qiskit={qiskit_metrics['depth']}"
        )

    if args.benchmark_json:
        document = {
            "schema_version": 1,
            "benchmark": "synthetic_seeded_optimizer_comparison",
            "random_seed": 20261123,
            "compile_repeats": args.repeats,
            "transpiler": {
                "quantumforge": "all + optional linear routing",
                "qiskit_optimization_level": 1,
                "qiskit_seed_transpiler": 20261123,
                "qiskit_basis_gates": basis_gates,
            },
            "quantumforge_package_version": manifest["package"]["version"],
            "quantumforge_source_revision": source_revision,
            "quantumforge_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "routing": "disabled" if args.no_routing else "linear",
            "qiskit_version": qiskit.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "metric_notes": [
                "Compile times exclude independent equivalence checking and are repeated wall-clock observations; compare only on equivalent machines and run conditions.",
                "QuantumForge timing includes its CLI process startup; Qiskit timing measures the in-process transpile call.",
                "The synthetic corpus is regression evidence, not a replacement for MQT Bench or hardware workloads.",
                "Both compiler outputs were checked against the same input unitary up to global phase.",
                "QuantumForge's routed output is compared directly in the input qubit order. Qiskit's baseline self-check uses an UNROUTED transpile, because a routed Qiskit circuit is returned in a permuted qubit order (see TranspileLayout.final_index_layout) and is not directly comparable.",
                "Qiskit runs at optimization level 1; levels 2 and 3 drop `swap` gates in Qiskit 2.6.x.",
            ],
            "cases": benchmark_rows,
        }
        args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    if failures:
        print(f"{failures} cross-tool validation case(s) failed", file=sys.stderr)
        return 1
    print(f"All {len(benchmark_rows)} cross-tool validation and comparison cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
