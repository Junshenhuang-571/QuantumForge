# QuantumForge Roadmap

This is a directional roadmap, not a promise of dates or a fixed release
schedule. Work should be prioritized through issues and reviewed against
correctness, usability, and maintainability.

## Near term

- **Onboarding:** keep the getting-started guide and runnable examples aligned
  with the current API; add focused tutorials for the main workflows.
- **Correctness confidence:** expand numerical and property-based tests around
  simulation semantics, algorithm invariants, and backend parity.
- **Release discipline:** synchronize crate and Python metadata, maintain
  changelog entries, and use the release checklist in `docs/releasing.md`.
- **API clarity:** document public module stability and migration guidance in
  accordance with `API_STABILITY.md`.
- **Performance evidence:** publish reproducible benchmark methods and
  hardware/feature context rather than unsupported speed claims.

## Longer term

- Improve scaling and memory guidance for state-vector, density-matrix, and
  tensor-network workloads.
- Strengthen backend conformance and test coverage across supported platforms.
- Improve Python packaging and examples so Python users can follow the same
  first-run path as Rust users.
- Grow community documentation and contributor workflows based on feedback
  from actual users.

## How roadmap items move forward

Open an issue describing the user problem, expected outcome, and relevant
workflow. Proposals should be scoped so they can be reviewed and validated
independently. The roadmap should be revised as priorities or evidence change.
