# Contributing to QuantumForge

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/Junshenhuang-571/QuantumForge.git
cd QuantumForge
cargo build
cargo test
```

## Guidelines

- All code must pass `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test`
- Add tests for new functionality
- Update documentation for public API changes
- Keep commits focused and atomic

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `cargo test` and `cargo clippy -- -D warnings`
5. Submit a PR with a clear description

## Code of Conduct

Be respectful and constructive. We follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

## Hardware-Specific Testing

- GPU features: `cargo test --features gpu` (NVIDIA) or `--features cuda`
- macOS: Metal support is planned; contributions welcome
- AMD/Intel GPUs: ROCm/HIP and oneAPI contributions welcome

CI runs on Linux, Windows, and macOS. GPU tests run locally only (no GPU in CI).
