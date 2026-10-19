"""
QuantumForge Python bindings.

Conditional imports for optional backends (CUDA, LLVM).
"""

# Core (always available)
from quantumforge_core import (
    Circuit,
    Gate,
    StateVector,
    DensityMatrix,
    parse_circuit,
    parse_qasm,
    export_qasm,
    # Algorithms
    grover_circuit,
    qaoa_maxcut,
    qpe_circuit,
    hea_ansatz,
    qaoa_ansatz,
    # Optimizer
    optimize_circuit,
    OptimizeConfig,
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

# Optional backends
try:
    from quantumforge_cuda import (
        CUDAStateVector,
        CUDADensityMatrix,
        CUDABackend,
    )
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    CUDAStateVector = None
    CUDADensityMatrix = None
    CUDABackend = None

try:
    from quantumforge_llvm import (
        LLVMBackend,
        LLVMJIT,
    )
    HAS_LLVM = True
except ImportError:
    HAS_LLVM = False
    LLVMBackend = None
    LLVMJIT = None

__all__ = [
    # Core
    "Circuit",
    "Gate",
    "StateVector",
    "DensityMatrix",
    "parse_circuit",
    "parse_qasm",
    "export_qasm",
    # Algorithms
    "grover_circuit",
    "qaoa_maxcut",
    "qpe_circuit",
    "hea_ansatz",
    "qaoa_ansatz",
    # Optimizer
    "optimize_circuit",
    "OptimizeConfig",
    # QEC
    "repetition_code_bitflip",
    "repetition_code_phaseflip",
    "shor_code",
    "steane_code",
    "surface_code_planar",
    "surface_code_rotated",
    "encode_repetition_bitflip",
    # Tensor networks
    "MPS",
    "Tensor",
    "TEBD",
    "DMRG",
    # Optional
    "HAS_CUDA",
    "HAS_LLVM",
    "CUDAStateVector",
    "CUDADensityMatrix",
    "CUDABackend",
    "LLVMBackend",
    "LLVMJIT",
]

def backend_info() -> dict:
    """Return information about available backends."""
    return {
        "cuda": HAS_CUDA,
        "llvm": HAS_LLVM,
        "core": True,
    }