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
cheaply. Current state: **31/31 public modules covered**, with `serialize` and
`python` reached through `Circuit::{to_json, from_json}` and the Python binding
scripts respectively (both listed explicitly in the script rather than implied).

## Line coverage

One `cargo tarpaulin --lib --tests` run, 2026-11-22 (after the `main.rs` and
`numeric_ir` work). Toolchain rustc 1.100.0, Windows, debug profile, default
features. `--lib --tests` covers the library unit tests and the integration
suite; examples and doctests are **not** included.

**Overall: 72.89% — 11511 of 15793 lines**, up from 69.80% before this pass.

### Lowest coverage

| file | covered/total | |
|---|---|---|
| `src/algorithms.rs` | 177/604 | HHL/Shor/QFT paths |
| `src/qec/mwpm.rs` | 94/243 | the heuristic decoder the docs already warn about |
| `src/hardware.rs` | 113/234 | provider scaffolds behind feature flags |
| `src/repl/state.rs` | 1179/2207 | 2207 lines of dispatch |
| `src/mbqc.rs` | 281/513 | measurement-based QC |
| `src/qec/surface.rs` | 160/277 | surface-code layout |

### Highest coverage

`src/bin/main.rs` 252/265 (95.1%), `qec/gf2.rs` 86/86, `subscore.rs` 464/469
(98.9%), `qaoa_transfer.rs` 293/299 (98.0%), `qec/codes.rs` 131/134,
`noisy_optimizer.rs` 145/149, `collisional.rs` 31/32, `mps_qaoa.rs` 213/223,
`thermal.rs` 134/142, `algorithms_templates.rs` 192/204, `qec/mod.rs` 44/47,
`parser.rs` 386/417, `vqt.rs` 85/92, `dissipative.rs` 352/383,
`qec/noise.rs` 142/157, `numeric_ir.rs` 157/183 (85.8%), `serialize.rs` 144/170.

### Reading the number

Before this pass `src/bin/main.rs` (0/259) and `src/numeric_ir.rs` (3/183) were
the two worst files, 440 lines with essentially nothing behind them. Both are
now above 85%, and both were covered by writing the tests the modules' own docs
asked for. `algorithms.rs` at 29.3% is the next target.

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

The run takes roughly 20 minutes because every test binary is rebuilt with
instrumentation.
