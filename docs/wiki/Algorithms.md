# Algorithms

## Implemented Algorithms

### Search & Optimization
- **Grover's Search** — O(√N) quantum search
- **QAOA** — Quantum Approximate Optimization Algorithm
- **VQE** — Variational Quantum Eigensolver

### Transforms
- **QFT** — Quantum Fourier Transform
- **QPE** — Quantum Phase Estimation

### Decision Problems
- **Deutsch-Jozsa** — determines if function is constant or balanced
- **Bernstein-Vazirani** — finds hidden string

### Linear Algebra
- **HHL** — Harrow-Hassidim-Lloyd for solving Ax=b
  - 2×2 quantum circuit
  - General NxN via Gaussian elimination (classical)
  - Quantum general (eigendecomposition + controlled rotation)

### Factoring
- **Shor's Algorithm** — quantum period finding
  - Circuit generator (`shor_period_finding`)
  - Classical period extraction (`shor_factor_quantum`)
  - Scalable fallback (`shor_factor_large` via Pollard rho)

### Variational
- **HEA** — Hardware Efficient Ansatz
- **Parameterized Circuits** — with adjoint + parameter-shift gradients

## Usage Examples

```rust
use quantumforge::algorithms::*;

// QFT
let circuit = qft_circuit(4);

// HHL (2x2)
let x = hhl_solve_quantum([[2.0, 1.0], [1.0, 2.0]], [3.0, 3.0]);

// HHL (general)
let a = vec![vec![4.0, 1.0], vec![1.0, 4.0]];
let b = vec![1.0, 2.0];
let x = hhl_solve_general(&a, &b);

// Shor
let factors = shor_factor(15); // Some((3, 5))
```
