# Gradient strategies

Module: [`quantumforge::gradients`](https://docs.rs/quantumforge/latest/quantumforge/gradients/)
· Tests: `tests/gradients.rs` · REPL: `gradient X|Y|Z Q [method=...]`

## The four rules

A variational stack differentiates `f(x) = <psi_x|O|psi_x>` in more than one
way, and the right choice depends on the circuit and on whether the expectation
itself is sampled.

| strategy | exact | needs | cost | notes |
|---|---|---|---|---|
| `ParameterShift` | yes | generator with two eigenvalues | 2 evals per parameterized occurrence | the only exact rule that supports finite shots |
| `Adjoint` | yes | state vector | 1 forward + 1 backward sweep | cheapest exact rule for one observable |
| `Backprop` | yes | state vector | 1 forward + 1 backward sweep | independent reverse-mode implementation |
| `FiniteDifference` | no, `O(h^2)` | anything | 2 evals per parameterized occurrence | fallback when no shift rule is known |

Every rule differentiates **per gate occurrence** and sums into the parameter, so
a parameter shared by `k` rotations gets the sum of its `k` occurrence-wise
derivatives. That matches the existing exact API and is what `tests/gradients.rs`
asserts.

## Automatic selection

`GradientStrategy::select(&CircuitProfile)` returns the cheapest applicable rule:

- a noise channel → `None` (the gradient lives on the density-matrix path);
- a shot budget → `ParameterShift`;
- otherwise → `Adjoint`.

`GradientOptions::default()` is `strategy: Auto` with a finite-difference step
of `1e-3` and `shots: 0`. `gradient_auto(...)` is the same call with those
defaults, and `GradientEstimate` reports which rule actually ran and how many
state-vector sweeps it consumed — the number that matters when you are choosing
between them.

`GradientOptions::shots` does not make this exact API sample anything. It tells
`Auto` that the gradient will ultimately be taken on sampled values, which is
the condition under which `ParameterShift` is the right answer. The gradient you
get back is still exact; take the noisy one from the `*_gradient_shots`
variants.

`GradientStrategy::parse` accepts the long names plus `shift` and `fd`, and
returns `None` for anything else — the REPL uses it rather than keeping a
second copy of the name table.

## What is actually new

Before this module the crate had `parameter_shift_gradient*` and
`adjoint_gradient*`. Both are unchanged. Added:

- **`GradientStrategy` / `CircuitProfile` / `GradientOptions` /
  `GradientEstimate`** — the dispatch layer.
- **`FiniteDifference`** — central differences with a validated step. Valid for
  any differentiable gate; in this gate set nothing needs it, but it is the
  anchor the exact rules are checked against.
- **`Backprop`** — reverse-mode backpropagation written from the definition
  (carry the costate `O|psi>` backwards, accumulate
  `2 Re <lambda| (dU/dtheta) |psi_before>`), rather than reusing the adjoint
  shortcut.

`Backprop` and `Adjoint` are algebraically the same computation for a Pauli
observable, and `tests/gradients.rs` asserts they agree to `1e-9`. They are both
kept because an independent implementation is the only real check on either, and
because backprop is the generic reverse-mode path to extend when a gate type
appears whose adjoint shortcut has not been hand-derived. Do not present them as
different mathematics.

## Tests and how to extend

`tests/gradients.rs` cross-validates rather than self-checks:

- an analytic anchor: `Ry(theta)|0>` has `<Z> = cos(theta)`, so the derivative
  is `-sin(theta)`, checked to `1e-10` for all three exact rules;
- every exact rule against every other on a 42-parameter, 7-qubit circuit;
- `Backprop` the free function against `Backprop` the method;
- shared parameters summing occurrence-wise;
- finite differences converging at second order (100x error reduction per 10x
  step reduction) and then losing to round-off near `h = 1e-5`;
- `Auto` resolving correctly, and rejecting a circuit with a channel;
- the finite-difference step validation.

A new rule should slot in as another `GradientStrategy` variant, a branch in
`gradient_with_inputs`, a row in `select`, and a line in the cross-validation
test. If it cannot be checked against the others, it is not ready.

## Where each rule is used elsewhere

- `cmd_vqe` (REPL) uses the adjoint gradient of `<H>` for its gradient-descent
  step, summing per-Pauli-term contributions.
- `mse_loss_and_gradient` and the QML trainers use parameter-shift derivatives.
- `subscore` does not differentiate at all: it measures `<H>` directly and
  differentiates a GP surrogate of it instead.
