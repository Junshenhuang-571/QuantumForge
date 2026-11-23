# Release Checklist

Use this checklist before creating a release tag. The publish workflow builds
Python wheels from `v*` tags, so a tag can trigger distribution. The crate is
published to crates.io with `cargo publish`; docs.rs and crates.io render from
the **published tarball**, not from the repository, so documentation-only
changes still need a release to become visible.

## Prepare

- [ ] Confirm the release scope and compatibility impact (breaking API changes
  bump the minor version under the pre-1.0 policy in `API_STABILITY.md`).
- [ ] Update `CHANGELOG.md`: move entries from `[Unreleased]` into a dated
  `[<version>]` heading and restart the empty `[Unreleased]` template.
- [ ] Synchronize the version in `Cargo.toml`, `pyproject.toml`,
  `API_STABILITY.md`, and the changelog heading.
- [ ] Confirm `README.md`, `Cargo.toml`, `pyproject.toml`, and `LICENSE` agree
  on the project's licensing model.
- [ ] Make sure nothing generated is inside the package: `dist/`, `target/`,
  wheel-test extraction dirs. `Cargo.toml` carries an `exclude` list — keep it
  current. (A stray 1.3 MB wheel once pushed the tarball from 314 KiB to
  1.5 MiB and caused crates.io upload timeouts, HTTP 408.)

## Verify

Run all of these on the **stable** toolchain; CI uses stable, and stable clippy
has lints nightly lacks (and vice versa), so verify with the same one CI uses:

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets -- -D warnings`
- [ ] `cargo test --workspace --lib --tests` (327 tests at 0.2.9)
- [ ] `cargo test --doc` — the crate docs are example-driven; doctests must pass
- [ ] `cargo doc --workspace --no-deps`
- [ ] Feature matrix: `cargo clippy --no-default-features --features <f> --lib
      -- -D warnings` for `llvm`, `python`, `rayon`, `ibm`, `ionq`, `rigetti`
      (the CI `feature-compile` jobs run these; `cuda`/`gpu` need a GPU host)
- [ ] `cargo package` and check the reported file count/size is sane
      (197 files, ~549 KiB compressed from the current working tree; recheck
      after release-only files or large fixtures change)
- [ ] `cargo publish --dry-run` — verifies the tarball compiles from a clean
      package directory. Note: it refuses on a dirty working tree; commit first.

### Clippy portability notes

- Windows `cargo package` currently warns that the `quantumforge` CLI and
  library produce the same `.pdb` filename. Packaging succeeds today, but Cargo
  warns this may become a hard error; resolve or verify this before a future
  toolchain upgrade.
- Avoid `#[allow(clippy::<lint>)]` for lints that may not exist on the
  toolchain CI resolves (an unknown lint fails under `-D warnings`). Prefer
  writing code that needs no allow.
- Prefer `x.is_multiple_of(2)` over `x % 2 == 0` (nightly clippy's
  `manual_is_multiple_of`), and inline format args (`"{x}"` over `"{}", x`).

## Publish

- [ ] Commit everything and push `main` (cargo publish requires a clean tree).
- [ ] `cargo publish` — the registry upload can hit transient network errors;
      retry. Success prints `Published quantumforge <version> at registry
      'crates-io'`.
- [ ] Verify the release is live: `curl -s https://crates.io/api/v1/crates/quantumforge/<version>`
- [ ] Create and push the `v<version>` tag. This triggers the wheel workflow.
- [ ] Confirm the wheel workflow's build jobs succeeded (linux-cpu,
      linux-llvm, macos-cpu, macos-arm64, windows-cpu).
- [ ] Wait for docs.rs to build the new version (a few minutes; `/crate/...`
      redirects until the build finishes).
- [ ] Publish release notes with compatibility and installation guidance, and
      record any feature or platform limitations.

## Wheel workflow notes (`.github/workflows/publish.yml`)

Lessons encoded in the workflow — do not "simplify" these away:

- **maturin is pinned (`maturin-version: 1.12.6`).** The latest maturin probes
  `cffi` bindings on this crate layout and crashes inside GitHub's hosted
  Python tool caches.
- **Every wheel build passes `--features python`** (plus extras). The crate's
  `default` feature set does not include `python`, so a `default` wheel ships
  no pyo3 module and maturin falls back to cffi detection.
- **Wheel matrix values containing commas break `actions/cache` keys** — the
  cache key uses `matrix.wheel`, not `matrix.feature`.
- **manylinux is `2_28`** (in both `pyproject.toml` and the action input). The
  older 2014 image ships a cffi that fails with `multiple declarations of
  function dlopen` on current Python tooling.
- **No linux `cuda` wheel**: the `cuda` feature links `libcuda` at build time,
  which GPU-less runners cannot do. GPU users build from source with the
  dynamic-loading `gpu` feature.
- **The PyPI publish job uses OIDC trusted publishing** (`id-token: write`).
  It stays red until a trusted publisher is configured on pypi.org for this
  repository and workflow — that step is account-side, not in the repo.
