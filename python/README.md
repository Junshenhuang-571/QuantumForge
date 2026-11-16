# QuantumForge Python bindings

QuantumForge ships an abi3 wheel (`quantumforge`) exposing the circuit
simulator, QEC code builders, parameterized circuits, and algorithm helpers.
Everything runs locally on your CPU; GPU/LLVM acceleration are compile-time
Rust features, not separate packages.

## Install

From PyPI once the project is published there:

```bash
pip install quantumforge
```

Or build from a checkout (requires Rust and
[maturin](https://github.com/PyO3/maturin)):

```bash
git clone https://github.com/Junshenhuang-571/QuantumForge.git
cd QuantumForge
maturin build --release          # writes a wheel to target/wheels/
pip install target/wheels/*.whl
```

## Quick start

```python
from quantumforge import Circuit, StateVector

circuit = Circuit()
circuit.add_gate("H", 0)
circuit.add_cnot(0, 1)

state = circuit.run(StateVector.zero(2))
print(state.probabilities())      # [0.5, 0.0, 0.0, 0.5]
```

Seeded sampling with reproducible counts:

```python
from quantumforge import Circuit, StateVector

circuit = Circuit()
circuit.add_gate("H", 0)
circuit.add_cnot(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)

result = circuit.run_shots(StateVector.zero(2), shots=1024, seed=42)
print(result.counts())            # only |00> and |11>
```

## Examples

Worked scripts, runnable against an installed wheel:

- [`examples/bell.py`](examples/bell.py) — Bell state, exact amplitudes and
  seeded shot counts
- [`examples/qec.py`](examples/qec.py) — surface/repetition code construction
  and repetition encoding
- [`examples/vqe.py`](examples/vqe.py) — variational minimization of `<Z>`
  with the parameter-shift gradient

Run them with the wheel importable, e.g.:

```bash
pip install target/wheels/*.whl
python python/examples/bell.py
```

## What is exposed

| Area | Names |
|---|---|
| Circuits | `Circuit`, `parse_circuit_py`, `parse_qasm_py`, `export_qasm_py`, `parse_gate_py`, `optimize_circuit_py`, `OptimizeConfig` |
| States | `StateVector`, `DensityMatrix`, `ShotResult` |
| QEC | `QecCode`, `PauliString`, `repetition_code_bitflip`, `repetition_code_phaseflip`, `shor_code`, `steane_code`, `surface_code_planar`, `surface_code_rotated`, `encode_repetition_bitflip` |
| Variational | `ParameterizedCircuit` |
| Algorithms | `Grover`, `QAOABuilder`, `QAOAAnsatzBuilder`, `HEABuilder`, `qft_circuit_py`, `hhl_*`, `shor_*` |
| Tensor networks | `MPS`, `Tensor`, `TEBD`, `DMRG` |

The authoritative list is `quantumforge/__init__.py` in the repository; the
Rust doc comments on each type carry usage examples.

## Notes

- Wheels are abi3 (`cp38-abi3`), so one wheel serves Python 3.8+.
- - Measurement outcomes are classical-register bit vectors, least-significant
  bit first; `ShotResult.counts()` returns a list of `(bits, count)` pairs.
- The Python API mirrors the Rust API one-to-one; the Rust documentation at
  <https://docs.rs/quantumforge> applies, with `snake_case` method names.
