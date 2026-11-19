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
`T = eta^2 / N` with `k_B = 1` and the shot budget *is* a thermostat. Three
consequences follow, and they are not equally useful.

**Fluctuation-dissipation.** The noise strength of the estimator is fixed by
the shot count, so a step that trusts a noisy surrogate is hot and one that
trusts an accurate surrogate is cold. The confident region is exactly the
statement "the surrogate is cold enough here to act on".

**The useful direction is cooling, and it is measured.** You cannot delete the
noise, only choose its size. The gain comes from spending it where accuracy does
not matter rather than uniformly across the run — which is what the adaptive
threshold does. On the demo instance that reaches a lower energy than a uniform
shot count at the same total budget (Bound + mean -4.4908 against fixed-shot
-3.9977). *That* is the thermodynamic benefit, and it is real. Hajek's condition
says a cooling schedule may not fall faster than `T(t) >= c / ln(t)` if you want
a guarantee of reaching the global minimum; the slope rule here cools faster
than that, so it converges to a good minimum rather than provably the best one.

**Adding thermal noise to the step is a different mechanism with a narrow
domain.** Annealing escapes a local minimum only when the optimizer is
*barrier-limited*. It is easy to confirm that it works there: on a 1D double
well, plain gradient descent stays in the shallow well 400/400 times, while a
Langevin step with an annealed temperature reaches the deep well 199/400 times.
But VQE at a practical shot budget is *information-limited*, not
barrier-limited — it has not measured enough to know where to go — and step noise
then destroys information. Measured on the demo instance, every thermal scale
tried does worse than the quench:

| update rule | mean E |
|---|---|
| mean (quench) | -4.4020 |
| guarded Thompson x0.5 | -3.4310 |
| guarded Thompson x0.25 | -3.4532 |
| guarded Thompson x0.1 | -3.6045 |

**Do not confuse model uncertainty with temperature.** `GaussianProcess::sample`
draws from the posterior, whose variance is large far from the observed points
because the *model* is ignorant there. That is not `T = eta^2 / N`: different
quantity, different units, and a sampled minimizer is a jump to an arbitrary
point rather than a Boltzmann fluctuation around the incumbent. Scaling the
deviation by `exploration` makes the step colder; it does not make it thermal
physics. `UpdateRule::Thompson` and `::GuardedThompson` are kept because they are
a legitimate finite-temperature update and a documented negative result, not
because they accelerate anything here.

## Which statistical-mechanics models actually apply

The repertoire is larger than the one distribution everyone reaches for, so here
is the whole mapping with a verdict for each.

| model | what it says | verdict here |
|---|---|---|
| Boltzmann / Gibbs, `p ∝ exp(-f/T)` | the equilibrium distribution minimizes the free energy `F = <E> - T S` | **applies** — this is the right target, and it gives the acquisition below |
| two-level (Boltzmann) single shot | `P(±1) = (1 ± <W>)/2`; `N` shots give a binomial, Gaussian only as `N → ∞` | exact, but **a dead end**: the GP uses only the noise *variance*, which is already exact. Measured `KL(binomial ‖ Gaussian)` is 0.02 nats at `N=1`, 0.005 at `N=16`, 0.0004 at `N=256` |
| Maxwell-Boltzmann + fluctuation-dissipation | the step noise is `sigma = sqrt(2 * lr * T)`; noise amplitude and dissipation are tied | **correct**, and it is what makes the annealing demo work: gradient descent stays in the shallow well 400/400, an annealed Langevin step reaches the deep well 199/400 |
| free energy `F = <E> - T S` | equilibrium is `argmin F`; entropy `S` is the exploration bonus | **implemented** as `UpdateRule::FreeEnergy`, with `T = eta/sqrt(N)` in energy units. Measured -4.3534 against the quench's -4.4020 over 16 seeds |
| Metropolis acceptance | accept an uphill move with probability `exp(-df/T)` | equivalent to the free-energy rule; not implemented separately |
| Bose-Einstein, Planck | bosonic occupancy, blackbody spectrum | no analogue — the estimator is not a bosonic mode |
| Jarzynski / Crooks | free-energy differences from a nonequilibrium work ensemble | would need a trajectory ensemble you do not have; noted, not built |
| H-theorem, `KL(p \|\| p_eq)` | a Lyapunov function: the divergence to equilibrium decreases | a plausible stopping criterion, not implemented |

Two readings worth keeping separate:

**Temperature is a standard deviation, not a variance.** The shot noise is a
variance, `eta^2/N`, so the temperature in `exp(-f/T)` is `T = eta/sqrt(N)`. For
the 4-qubit demo `eta = 2.65`, so 512 shots per point is `T = 0.117` against an
energy scale of about 4 — a cold bath. Getting hot enough to cross a barrier of
height `dE` needs `N ~ eta^2 / dE^2`, which for `dE = 1` is about 7 shots per
point. That is the thermometer: the shot count that can cross a barrier is
readable off the barrier height.

**The free-energy rule is the one that survives contact.** It uses the physical
temperature, so its exploration bonus `T * sigma` is at most `0.117` at 512
shots — small next to the energy differences the optimizer works with, and it
vanishes automatically as the run cools. That is why it tracks the quench
instead of collapsing the way posterior sampling does.

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
