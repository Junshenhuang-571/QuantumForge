# QAOA parameter transfer and targeted-single-layer optimization

Implementation of Shubham Patel and Utkarsh Mishra, *Improving the efficiency of
QAOA using efficient parameter transfer initialization and targeted-single-layer
regularized optimization with minimal performance degradation*
([arXiv:2601.15760](https://arxiv.org/abs/2601.15760), January 2026).

Module: [`quantumforge::qaoa_transfer`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/)
· Example: `cargo run --release --example qaoa_transfer`
· Tests: `tests/qaoa_transfer.rs`

## The idea

A depth-`p` QAOA ansatz has `2p` angles and a non-convex landscape that grows
with the graph. The paper's answer is to optimize the angles once, on a small
graph, and reuse them:

1. **Donor optimization.** Take an 8-vertex graph from the same family as the
   target. Initialize its `2p` angles with the trotterized-quantum-annealing
   (TQA) schedule and optimize them fully with Adagrad.
2. **Transfer.** Copy the optimized angles onto the larger acceptor graph of the
   same family, unchanged.
3. **Targeted-single-layer (TSL) optimization.** Try each layer on its own and
   optimize only that layer's two angles, ridge-regularized.

Step 3 optimizes 2 parameters instead of `2p`, which is where the reported
`~8x` speedup comes from. On unweighted graphs the approximation ratio stays
close to that of a full re-optimization.

## API walkthrough

```rust
use quantumforge::mps_qaoa::MaxCutProblem;
use quantumforge::qaoa_transfer::{run_parameter_transfer, TransferOptions};

let donor = MaxCutProblem::random_3_regular(8, 20261117);
let acceptor = MaxCutProblem::random_3_regular(12, 20261118);

let mut options = TransferOptions::default();
options.depth = 8;              // p
options.donor_iterations = 40;  // Adagrad steps on the donor
options.compare_full_optimization = true;

let outcome = run_parameter_transfer(&donor, &acceptor, &options)?;
println!("r_n = {:.5}", outcome.transferred_ratio);
println!("r_s = {:.5}", outcome.approximation_ratio);
println!("r_f = {:?}", outcome.full_optimization_ratio);
# Ok::<(), quantumforge::qaoa_transfer::TransferError>(())
```

Individual pieces, if you want to drive the pipeline yourself:

| Function | Role |
|---|---|
| [`tqa_initialization`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.tqa_initialization.html) | Eq. (1) of the paper: `gamma_i = (i/p) dt`, `beta_i = (1 - i/p) dt` |
| [`adagrad`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.adagrad.html) | Full `2p`-angle donor optimization (finite-difference gradients) |
| [`maxcut_qaoa_expectation`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.maxcut_qaoa_expectation.html) | `<H_C>` of a QAOA state |
| [`select_target_layer`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.select_target_layer.html) | One pass of Algorithm 1; also returns every layer's outcome |
| [`select_target_layer_majority`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.select_target_layer_majority.html) | Algorithm 1 as published: majority vote over several acceptor instances |
| [`optimize_targeted_layer`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.optimize_targeted_layer.html) | The ridge-regularized two-angle step |
| [`nelder_mead`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.nelder_mead.html) | Derivative-free minimizer used for the targeted step |

## What the example prints

On the seeded pair used by `examples/qaoa_transfer.rs` (8 → 12 vertices,
`p = 8`):

```
transferred  r_n : 0.91602
targeted     r_s : 0.91795
full         r_f : 0.95571
r_s / r_f       : 96.05%
```

The paper reports `r_s / r_f = 98.88%` at `p = 15` with 100 Adagrad iterations
on the donor. Raising `depth` and `donor_iterations` moves this implementation
in the same direction; the example keeps them low so it finishes in seconds.

## Two things worth knowing

**The donor learning rate.** The paper uses `lr = 0.1` with exact autograd
gradients. This module uses finite differences, and at `0.1` Adagrad overshoots
and stalls in a poor local optimum: on a seeded 8-vertex 3-regular donor at
`p = 6`, the donor reaches r = 0.825 at `lr = 0.1` but 0.980 at `lr = 0.05`, and
the transferred acceptor ratio goes from 0.630 to 0.878. The default is
therefore `0.05`. With that setting the pipeline reproduces the paper's headline
closely — 8 → 10 vertices at `p = 6`:

```
iters=100:  r_n = 0.9065   r_s = 0.9068   r_f = 0.9459   r_s/r_f = 95.9%
iters=300:  r_n = 0.9239   r_s = 0.9255   r_f = 0.9459   r_s/r_f = 97.8%
```

**Sign convention.** The paper defines the approximation ratio as
`r = <H_C> / C_max` with `<H_C>` the cut weight to be *maximized*, but writes
the TSL objective as `<H_C> + lambda (gamma^2 + beta^2)` and says it is
minimized. Those cannot both hold. This crate maximizes the cut weight, so the
minimized objective is `-<H_cut> + lambda (gamma^2 + beta^2)` — see
[`targeted_layer_objective`](https://docs.rs/quantumforge/latest/quantumforge/qaoa_transfer/fn.targeted_layer_objective.html).
The physics is unchanged; only the sign is made consistent. (The `(gamma, beta)
-> (-gamma, -beta)` symmetry of the landscape means the *circuit's* sign
convention is not the issue.)

**Where this stops paying off.** Every objective evaluation rebuilds and runs
the full state-vector circuit, costing `O(2^n (n + E) p)`. The paper itself
finds the method weakest on *weighted* families at larger vertex counts, where
the transferred angles are no longer near-optimal. For acceptors beyond roughly
14 qubits, transfer the same idea to
[`mps_qaoa`](https://docs.rs/quantumforge/latest/quantumforge/mps_qaoa/)
instead of the state-vector path.

## In the REPL

```
qf> qaoa-transfer                 # 8 -> 10 vertices, p=6, 100 Adagrad steps
qf> qaoa-layers                   # cut expectation for every targeted layer
qf> qaoa-angles donor             # donor | transferred | optimized
qf> qaoa-layer 3                  # what if only layer 3 were optimized?
```

`qaoa-transfer` takes `donor=`, `acceptor=`, `p=`, `dt=`, `lr=`, `iters=`,
`lambda=`, `full=` and `seed=`. The run is stored on the session, so the other
three commands work on it afterwards. `full=1` adds the `r_f` comparison, which
roughly triples the runtime.

Expect ~7 s per run in a release build and tens of seconds in a debug build —
the command says so before it starts.

## Related

- [MPS-based QAOA](MPS-QAOA.md) — QAOA on matrix product states for large MaxCut
- [Algorithms](Algorithms.md) — the state-vector QAOA builders this builds on
