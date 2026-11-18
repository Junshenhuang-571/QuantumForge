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
| [`Observation`](https://docs.rs/quantumforge/latest/quantumforge/subscore/struct.Observation.html) | One noisy reading: value, its variance, and the shots spent |
| [`UpdateRule`](https://docs.rs/quantumforge/latest/quantumforge/subscore/enum.UpdateRule.html) | Mean (quench) or Thompson (finite temperature) update |

```rust
use quantumforge::subscore::{
    Observation, ReadoutStrategy, ShotHamiltonian, SubsCoReOptions, UpdateRule, run_subscore,
};

let options = SubsCoReOptions {
    total_shot_budget: 40_000,
    eta_sq: objective.single_shot_variance(), // the noise scale is the thermostat
    initial_core_variance: objective.single_shot_variance() / 512.0,
    core_min: objective.single_shot_variance() / 4096.0,
    strategy: ReadoutStrategy::Bound,
    ..SubsCoReOptions::default()
};
let outcome = run_subscore(
    |params, shots, seed| -> Observation { objective.observe(params, shots, seed) },
    &initial_params,
    &parameter_gate_counts,
    &options,
);
println!("E = {:.4} after {} shots", outcome.best_value, outcome.shots_used);
```

`eta_sq` and the two variance thresholds must be set from the *objective's* noise
scale. `ShotHamiltonian::single_shot_variance()` reports it: a Pauli word with
coefficient `c` contributes `c^2`, so a 7-word unit-coefficient Hamiltonian has
`eta^2 = 7`. The paper's Table 1 quotes the threshold as a shot count (512 to
start, 4096 at the floor); divide by `eta_sq` to get the variance form.

`parameter_gate_counts[d]` is `V_d` — how many gates parameter `d` drives — and
sets how many observations each subspace iteration needs. For an ansatz where
every parameter drives exactly one gate, it is all ones.

## What the example shows

`cargo run --release --example subscore` runs a 4-qubit, 2-layer transverse-field
Ising VQE under one 40,000-shot budget and averages over 6 seeds. Measured on
this machine:

| method | mean E | best E | vs start | iterations |
|---|---|---|---|---|
| fixed-shot NFT | -3.9977 | -4.3878 | -0.6077 | 28 |
| SubsCoRe-Bound + mean | -4.4908 | -4.6201 | -1.1009 | 20 |
| SubsCoRe-Center + mean | -3.7495 | -4.5788 | -0.3596 | 17 |
| Center + Thompson | -0.4071 | -0.8298 | +2.9829 | 13 |
| Center + guarded Thompson | -2.2377 | -3.4107 | +1.1523 | 15 |
| Center + guarded Thompson x1.5 | -2.4584 | -3.6681 | +0.9316 | 14 |

Three things are worth reading off that table.

**Adaptive beats fixed-shot once the threshold is annealed.** Bound + mean
reaches -4.49 against the fixed-shot -4.00 at the same budget. That gap appeared
only after the threshold was made monotone (see below); before that the run
could make itself hotter than it started and starved its own observations.

**Center underperforms Bound here.** The paper reports Center >= Bound. On this
instance it is worse, because Center spends the center point (the point the
optimizer actually acts on) very accurately and the shifted points very
loosely, and with `V_d = 1` the three axis observations jointly identify the
subspace. This is a measured disagreement with the paper, not a tuning result.

**Thompson sampling does not accelerate here — it is catastrophic.** The
posterior variance is large away from the observed points, so a sampled
function's minimizer is essentially arbitrary, and a hot step can destroy a
solution the optimizer was happy with. The guarded variant refuses those steps
but still explores too hard at this budget. The mechanism is implemented and
correct (see below); the honest conclusion is that on this landscape, at this
budget, the zero-temperature step is better. Reported as a negative result
rather than quietly dropped.

## The shot noise is a temperature

An observation averaged over `N` shots has variance `eta^2 / N`. Write
`T = eta^2 / N` with `k_B = 1` and the shot budget *is* a thermostat.

**Fluctuation-dissipation.** The noise strength of the estimator is fixed by
the shot count, so a step that trusts a noisy surrogate is a hot step and one
that trusts an accurate surrogate is cold. The confident region is exactly the
statement "the surrogate is cold enough here to act on". SubsCoRe's adaptive
threshold is therefore a cooling schedule: loose while the energy falls, tight
near convergence. It is clamped to be monotone non-increasing, because annealing
that gets hotter than its own starting point is not annealing.

**Finite-temperature updates.** `GaussianProcess::sample` draws a whole
function from the posterior (a joint draw, so the sampled curve keeps the
posterior's correlations), and `UpdateRule::Thompson` moves to *its* minimizer.
That is Boltzmann exploration: the posterior variance plays the role of `T`, so
the temperature falls automatically as shots accumulate. Hajek's condition for
guaranteed convergence to a global minimum is `T(t) >= c / ln(t)`; the schedule
here is whatever the budget imposes, so a run that ends with many shots per
point has quenched.

`UpdateRule::GuardedThompson` adds a safeguard: a kick is refused when it is
uphill by more than the surrogate's own standard deviation there. (Comparing
against the mean alone refuses almost every kick and reduces the rule to
`Mean`.) The thermal step may cross a barrier, but it may not discard a solution
the surrogate already trusts. On the demo instance above even the guarded
variant loses to the plain quench — the exploration it buys is not worth the
shots at this budget.

`ShotHamiltonian::observe` reports the estimator's *actual* variance
`sum_w c_w^2 (1 - <W_w>^2) / N_w` with every reading, so the GP is never told
the noise is smaller than it is. An earlier version assumed `eta^2 / N` for a
single word, which understated the noise by the word count.

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
