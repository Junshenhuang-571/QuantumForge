"""
QuantumForge Python bindings.

The compiled extension module is named `quantumforge` (built with maturin).
CUDA and LLVM acceleration are compile-time cargo features: when built with
`--features cuda` or `--features llvm` the accelerated paths are used from
within this same module. There are no separate `quantumforge_cuda` /
`quantumforge_llvm` Python packages.

Names below match exactly what the Rust `#[pymodule]` registers
(see `src/python.rs`); importing is smoke-tested by `python/smoke_test.py`.
"""

from quantumforge import (
    # Core
    Circuit,
    StateVector,
    DensityMatrix,
    ShotResult,
    PauliString,
    QecCode,
    # Parsing
    parse_circuit_py,
    parse_qasm_py,
    export_qasm_py,
    parse_gate_py,
    # Optimizer
    optimize_circuit_py,
    OptimizeConfig,
    # Algorithms
    Grover,
    QAOABuilder,
    QAOAAnsatzBuilder,
    HEABuilder,
    ParameterizedCircuit,
    qft_circuit_py,
    hhl_2x2_py,
    hhl_solve_py,
    hhl_solve_general_py,
    hhl_solve_quantum_general_py,
    shor_period_finding_py,
    shor_factor_py,
    shor_factor_large_py,
    shor_factor_quantum_py,
    # QEC
    repetition_code_bitflip,
    repetition_code_phaseflip,
    shor_code,
    steane_code,
    surface_code_planar,
    surface_code_rotated,
    encode_repetition_bitflip,
    # Tensor networks
    MPS,
    Tensor,
    TEBD,
    DMRG,
)

__all__ = [
    "Circuit",
    "StateVector",
    "DensityMatrix",
    "ShotResult",
    "PauliString",
    "QecCode",
    "parse_circuit_py",
    "parse_qasm_py",
    "export_qasm_py",
    "parse_gate_py",
    "optimize_circuit_py",
    "OptimizeConfig",
    "Grover",
    "QAOABuilder",
    "QAOAAnsatzBuilder",
    "HEABuilder",
    "ParameterizedCircuit",
    "qft_circuit_py",
    "hhl_2x2_py",
    "hhl_solve_py",
    "hhl_solve_general_py",
    "hhl_solve_quantum_general_py",
    "shor_period_finding_py",
    "shor_factor_py",
    "shor_factor_large_py",
    "shor_factor_quantum_py",
    "repetition_code_bitflip",
    "repetition_code_phaseflip",
    "shor_code",
    "steane_code",
    "surface_code_planar",
    "surface_code_rotated",
    "encode_repetition_bitflip",
    "MPS",
    "Tensor",
    "TEBD",
    "DMRG",
]


def backend_info() -> dict:
    """Return information about available backends."""
    return {"core": True}
