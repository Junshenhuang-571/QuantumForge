# Test coverage

Two different questions, two different tools. Line coverage says how much of a
module is exercised; the module check says whether a module is exercised **at
all**. The second question is the expensive one: `stabilizer` had no test file
referencing it and shipped seven bugs, four of them silent wrong answers rather
than panics.

## Module coverage gate

`scripts/module_test_coverage.py` lists public modules that nothing under
`tests/`, `examples/` or `python/` mentions, and CI fails on any:

```bash
python scripts/module_test_coverage.py --fail
```

It is a proxy — a mention is not an assertion — but it catches the zero case
cheaply. Current state: **33 public modules, none at zero**, with `serialize` and
`python` reached through `Circuit::{to_json, from_json}` and the Python binding
scripts respectively (both listed explicitly in the script rather than implied).

## Line coverage

One `cargo tarpaulin --lib --tests` run on `codex/optimizer-professional-workflow`,
2026-11-24 (after the `noise_model` test work). Toolchain rustc 1.100.0, Windows,
debug profile, default features. `--lib --tests` covers the library unit tests
and the integration suite; examples and doctests are **not** included.

**Overall: 71.71% — 12157 of 16952 lines.**

> Earlier revisions of this page reported 72.89% (11511/15793) on `main` and
> 70.49% (11950/16952) before the `noise_model` tests. The 72.89% figure is
> **not comparable**: this branch adds roughly 1160 lines of source
> (`noise_model`, readout validation, thermal circuit noise, MPS observables).
> Compare trends within one codebase, not across branches.

### Lowest coverage

| file | covered/total | |
|---|---|---|
| `src/qec/mwpm.rs` | 94/243 | the heuristic decoder the docs already warn about |
| `src/hardware.rs` | 113/234 | provider scaffolds behind feature flags |
| `src/repl/state.rs` | 1179/2207 | 2207 lines of dispatch |
| `src/bin/main.rs` | 527/965 | see the correction below |
| `src/mbqc.rs` | 281/513 | measurement-based QC |

`src/algorithms.rs` is no longer in this table: it stood at 176/604 in the run
above, which was taken **before** `tests/algorithms.rs` was written. That file
adds 15 tests over the builders that had none. Its two worst paths —
`qft_circuit` and `qpe_circuit` — remain uncovered **on purpose**, because both
are broken (see the Known issues section in `CHANGELOG.md`); a test pinning
them would lock the defect in place.

### Closed: `noise_model.rs`

`src/noise_model.rs` went from **3/245 (1.2%) to 214/245 (87.3%)** by adding
`tests/noise_model.rs` (21 tests). It is the subsystem every noisy result depends
on, so the tests pin the physics as well as the plumbing: the `T1` relaxation
probability, detailed balance (`p1/p0 -> exp(-beta*Delta)`), the documented
`exp(-duration/T2)` coherence contract, and the full `apply_scheduled`
validation and idle-placement surface.

### Correction: `main.rs`

An earlier revision of this page claimed `src/bin/main.rs` at **252/265 (95.1%)**.
That was true *then*. This branch grew the file from 265 to 965 lines — batch
`optimize`, `--report`, `--line`/`--topology`, `estimate`, Wilson intervals,
SHA-256 digests — and most of that new surface has no test. It now sits at
**527/965 (54.6%)**. `tests/cli_main.rs` drives `demo`, `eval`, `simplify`,
`shots`, `estimate`, `help` and `optimize`, but only along the paths those tests
happen to hit.

### Highest coverage

`qec/gf2.rs` 86/86, `simulation_plan.rs` 23/23, `subscore.rs` 464/469 (98.9%),
`qaoa_transfer.rs` 293/299 (98.0%), `qec/codes.rs` 131/134,
`noisy_optimizer.rs` 145/149, `collisional.rs` 31/32, `mps_qaoa.rs` 213/223,
`thermal.rs` 134/142, `algorithms_templates.rs` 192/204,
`stabilizer.rs` 116/130 (89.2%), `numeric_ir.rs` 157/183 (85.8%),
`optimizer.rs` 479/641 (74.7%), `qasm.rs` 373/520 (71.7%).

### Reading the number

`src/main.rs` at 54.6% is the other large gap, and unlike the earlier `main.rs`
result it is not a small, contained fix.

Current doctest total: **16**, including worked chemistry and QPE examples in
the crate-level rustdoc. The line-coverage figures above were measured before
`tests/qft_qpe.rs` and the three chemistry suites; those 15 new tests are not
reflected in the tarpaulin numbers yet.

The `qec/mwpm.rs` figure is consistent with its documented status as a
surface-code-oriented heuristic that should not be used as a general decoder.

`NumericIrError::UnsupportedScalarExpression` is documented as unreachable:
`ScalarExpr` is a closed enum of thirteen variants and `lower_scalar` handles all
thirteen. Do not write a test asserting it.

Reproduce with:

```bash
cargo install cargo-tarpaulin --locked
cargo tarpaulin --lib --tests --out Stdout
```

The run takes roughly 25 minutes because every test binary is rebuilt with
instrumentation.
