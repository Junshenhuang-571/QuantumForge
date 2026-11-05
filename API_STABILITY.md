# API Stability Policy

QuantumForge is currently published as version `0.2.4`. The Rust crate is
pre-1.0, so its public API is not yet declared stable. This policy describes
how changes are communicated and how stability will be introduced; it does not
promise that every current public item will remain unchanged.

## Before version 1.0

- Treat documented public Rust APIs as supported, but evolving.
- Keep patch releases backward-compatible whenever practical.
- A minor release may make a breaking API change. Call it out in the changelog
  and provide migration guidance for changes that affect normal usage.
- Deprecate public items before removal when a compatible transition is
  practical. State the replacement and intended removal window in rustdoc and
  the changelog.
- Avoid exposing implementation details as public API unless users need them.
- Feature-gated APIs (`cuda`, `gpu`, `llvm`, `python`, and provider features)
  are supported only under their corresponding feature and platform
  requirements; document those requirements explicitly.

## Stable API boundaries

The public surface includes public modules, types, functions, traits, enum
variants, fields, and feature names exposed by the crate. A `pub` declaration
is not, by itself, a promise of long-term stability. Each release should
document material changes to these surfaces and keep examples compiling.

Once the project reaches `1.0.0`, the intended policy is conventional SemVer:
patch releases contain compatible fixes, minor releases add compatible
functionality, and major releases may break the public API. Any change to that
commitment should be announced before the version transition.

## Change checklist

For a public API change:

1. Identify affected items, callers, and enabled feature combinations.
2. Prefer additive changes and preserve existing behavior when possible.
3. If breaking, explain why, document migration steps, and update examples.
4. Update `CHANGELOG.md` and relevant rustdoc or user documentation.
5. Run formatting, lint, test, and documentation checks from
   [CONTRIBUTING.md](CONTRIBUTING.md).

## Version and distribution consistency

Before publishing, verify that the Rust crate version, Python package version,
changelog heading, release tag, and generated package metadata refer to the
same release. The project uses Apache-2.0; keep `Cargo.toml`, `pyproject.toml`,
`README.md`, and `LICENSE` consistent with that licensing model.
