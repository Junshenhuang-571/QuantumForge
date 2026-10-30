# Contributing

## Prerequisites

- Rust 1.70+ (stable)
- For GPU: NVIDIA CUDA toolkit or compatible

## Building

```bash
cargo build              # CPU-only
cargo build --features gpu  # NVIDIA GPU
cargo build --features cuda # Native CUDA driver
```

## Testing

```bash
cargo test              # 202 tests (default)
cargo test --features gpu  # GPU tests (requires hardware)
cargo clippy -- -D warnings  # Lint check
cargo fmt -- --check    # Format check
```

## Hardware-Specific Testing

- GPU features: `cargo test --features gpu` (NVIDIA) or `--features cuda`
- macOS: contributions welcome for Metal support
- AMD/Intel GPUs: ROCm/HIP and oneAPI contributions welcome

CI runs on Linux, Windows, and macOS. GPU tests run locally only.

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `cargo test` and `cargo clippy -- -D warnings`
5. Submit a PR with a clear description
