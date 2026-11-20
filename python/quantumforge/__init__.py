"""
QuantumForge Python bindings.

The compiled extension module is `quantumforge` (built with maturin). Maturin
places the extension as `quantumforge.quantumforge`, so imports here are
relative. CUDA and LLVM acceleration are compile-time cargo features used from
within this same module; there are no separate `quantumforge_cuda` /
`quantumforge_llvm` packages.

The names below match exactly what the Rust `#[pymodule]` exports — pyclasses
keep their `Py*` names and functions their `*_py` names there; this module
aliases them to the clean public names. Import is smoke-tested by
`python/smoke_test.py`.
"""

from .quantumforge import (
    PyCircuit as Circuit,
    PyGradientResult as GradientResult,
    PyCircuitProfile as CircuitProfile,
    gradient_strategies,
    PyRational as Rational,
    PyExactComplex as ExactComplex,
    PyPauliSum as PauliSum,
    PySubsCoReOutcome as SubsCoReOutcome,
    vqe_subscore_py as vqe_subscore,
    PyStateVector as StateVector,
    PyDensityMatrix as DensityMatrix,
    PyShotResult as ShotResult,
    PyPauliString as PauliString,
    PyQecCode as QecCode,
    parse_circuit_py,
    parse_qasm_py,
    export_qasm_py,
    parse_gate_py,
    optimize_circuit_py,
    PyOptimizeConfig as OptimizeConfig,
    PyGrover as Grover,
    PyQAOABuilder as QAOABuilder,
    PyQAOAAnsatzBuilder as QAOAAnsatzBuilder,
    PyHEABuilder as HEABuilder,
    PyParameterizedCircuit as ParameterizedCircuit,
    qft_circuit_py,
    hhl_2x2_py,
    hhl_solve_py,
    hhl_solve_general_py,
    hhl_solve_quantum_general_py,
    shor_period_finding_py,
    shor_factor_py,
    shor_factor_large_py,
    shor_factor_quantum_py,
    repetition_code_bitflip,
    repetition_code_phaseflip,
    shor_code,
    steane_code,
    surface_code_planar,
    surface_code_rotated,
    encode_repetition_bitflip,
    PyMPS as MPS,
    PyTensor as Tensor,
    PyTEBD as TEBD,
    PyDMRG as DMRG,
)

__all__ = [
    "Circuit",
    "GradientResult",
    "CircuitProfile",
    "gradient_strategies",
    "Rational",
    "ExactComplex",
    "PauliSum",
    "SubsCoReOutcome",
    "vqe_subscore",
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
