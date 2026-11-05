# Release Checklist

Use this checklist before creating a release tag. The publish workflow builds
Python wheels from `v*` tags, so a tag can trigger distribution.

## Prepare

- [ ] Confirm the release scope and compatibility impact.
- [ ] Update `CHANGELOG.md` with user-visible changes, fixes, and migration
  notes.
- [ ] Synchronize the package version in `Cargo.toml`, `pyproject.toml`, and
  the changelog heading.
- [ ] Confirm `README.md`, `Cargo.toml`, `pyproject.toml`, and `LICENSE` agree
  on the project's licensing model.
- [ ] Review optional feature and platform support claims against what was
  actually built and tested.

## Verify

- [ ] Run `cargo fmt --all -- --check`.
- [ ] Run `cargo clippy --all-targets -- -D warnings`.
- [ ] Run `cargo test --workspace --lib --tests`.
- [ ] Run `cargo doc --workspace --no-deps`.
- [ ] Build and test applicable optional features on hosts that meet their
  external toolchain requirements.
- [ ] Build the Python package and inspect its wheel metadata when publishing
  Python distributions.

## Publish

- [ ] Create a `v<version>` tag matching the package versions.
- [ ] Confirm the tag workflow completed successfully and produced the intended
  artifacts.
- [ ] Publish release notes with compatibility and installation guidance.
- [ ] Record any feature or platform limitations in the release notes.
