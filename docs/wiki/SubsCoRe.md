# SubsCoRe: adaptive observation cost control for VQE

Implementation of Anders, Nicoli, Wu, Elosegui, Pedrielli, Funcke, Jansen, Kühn
and Nakajima, *Adaptive Observation Cost Control for Variational Quantum
Eigensolvers* ([arXiv:2502.01704](https://arxiv.org/abs/2502.01704), ICML 2024).

Module: [`quantumforge::subscore`](https://docs.rs/quantumforge/latest/quantumforge/subscore/)
· Example: `cargo run --release --example subscore`
· Tests: `tests/subscore.rs`

## The problem

VQE minimizes `f(x) = <psi_x|H|psi_x>` over `x` in `[0, 2pi)^D`. Every
*observation* of `f` is an average over many measurement shots, so it carries
shot noise with variance `eta^2 / N_shots`. Quantum time is the scarce
resource, so the cost that matters is the **total number of shots**, not the
number of classical iterations.

Because the circuit is rotations plus fixed gates, `f` is a low-order
trigonometric polynomial along each axis, so a 1D subspace is identified from
`1 + 2 V_d` observations (NFT / sequential minimal optimization). SubsCoRe runs
that subspace optimization and chooses **how many shots to spend at each
observed point** so the whole updated axis stays inside the *confident region*
— the set where the GP surrogate's posterior variance is at most `kappa^2`.

`kappa` adapts: loose while the energy is dropping fast, tightening to
`core_min` as progress stalls, which is where accurate observations pay off.

## API

| Item | Role |
|---|---|
| [`vqe_kernel`](https://docs.rs/quantumforge/latest/quantumforge/subscore/fn.vqe_kernel.html) | VQE kernel (Eq. 10): product over axes of a truncated Fourier sum |
| [`GaussianProcess`](https://docs.rs/quantumforge/latest/quantumforge/subscore/struct.GaussianProcess.html) | Heteroscedastic GP regression with that kernel (Eqs. 4–5) |
| [`axis_points`](https://docs.rs/quantumforge/latest/quantumforge/subscore/fn.axis_points.html) | The `1 + 2 V_d` equidistant points along one axis |
| [`choose_shots_bound`](https://docs.rs/quantumforge/latest/quantumforge/subscore/fn.choose_shots_bound.html) | SubsCoRe-Bound: same variance at every point |
| [`choose_shots_center`](https://docs.rs/quantumforge/latest/quantumforge/subscore/fn.choose_shots_center.html) | SubsCoRe-Center: separate variances, fewest shots (Eq. 15) |
| [`run_subscore`](https://docs.rs/quantumforge/latest/quantumforge/subscore/fn.run_subscore.html) | The full loop (Algorithm 1) |
| [`ShotHamiltonian`](https://docs.rs/quantumforge/latest/quantumforge/subscore/struct.ShotHamiltonian.html) | Shot-based `PauliSum` energy objective for an ansatz |

```rust
use quantumforge::subscore::{run_subscore, ReadoutStrategy, ShotHamiltonian, SubsCoReOptions};

// `objective.observe(params, shots, seed)` estimates <H> from measurement shots.
let options = SubsCoReOptions {
    total_shot_budget: 20_000,
    strategy: ReadoutStrategy::Center,
    ..SubsCoReOptions::default()
};
let outcome = run_subscore(
    |params, shots, seed| objective.observe(params, shots, seed),
    &initial_params,
    &parameter_gate_counts,
    &options,
);
println!("E = {:.4} after {} shots", outcome.best_value, outcome.shots_used);
```

`parameter_gate_counts[d]` is `V_d` — how many gates parameter `d` drives — and
sets how many observations each subspace iteration needs. For an ansatz where
every parameter drives exactly one gate, it is all ones.

## What the example shows

`cargo run --release --example subscore` runs a 3-qubit transverse-field Ising
VQE three ways under the same 20,000-shot budget: a fixed-shot NFT baseline,
SubsCoRe-Bound, and SubsCoRe-Center.

Measured on this machine:

| method | iterations | shots | final E | vs start |
|---|---|---|---|---|
| fixed-shot NFT | 20 | 20000 | -3.4232 | -1.0804 |
| SubsCoRe-Bound | 13 | 19999 | -3.4152 | -1.0724 |
| SubsCoRe-Center | 14 | 19999 | -3.3917 | -1.0490 |

The mechanism is visible even where the ranking is not: SubsCoRe-Center spent
**1,536 shots on its first iteration and 10,719 on its last**, moving budget
from the rapidly-improving phase to the converging phase, exactly as the paper
describes. On this small and easy instance all three land within 0.03 of each
other; the paper's reported advantage appears on harder instances with larger
budgets (it uses `2.5 * 10^6` shots). Treat the ranking here as a demonstration
of the mechanism, not as a reproduction of the paper's benchmark.

## Deviations from the paper

- **Direction of the shot search.** The paper's stated objective (Eq. 15) is to
  *maximize* the total observation variance — spend the fewest shots — subject
  to the confident-region constraint, but its pseudocode scans variance upwards
  from `eta` and keeps the first feasible value, which is the opposite
  direction. This module implements Eq. 15: feasibility is monotone (less
  observation noise can only help), so it scans upward from `kappa^2` and keeps
  the largest feasible variance.
- **`kappa` units.** The paper uses `kappa` both as a variance threshold
  (Eqs. 12, 15) and as a shot count (Table 1). Here `kappa_sq` is always a
  variance.
- **GP size.** The paper caps the GP at about 100 points for classical cost.
  This implementation stores every observation; `push` is `O(n^2)` via an
  incremental Cholesky update, so a few hundred points is fine, but a
  multi-million-shot run will accumulate many observations.

## Related

- [QAOA parameter transfer](QAOA-Parameter-Transfer.md) — the other
  shot-budget-aware optimization method in the crate
- [Algorithms](Algorithms.md) — the state-vector VQE builders
