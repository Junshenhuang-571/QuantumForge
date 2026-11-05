//! Criterion benchmarks for QuantumForge's hot paths: state-vector gate
//! application, density-matrix operations, and the exact-rational analytic
//! core. These make the performance envelope durable instead of anecdotal.
//!
//! Run with: `cargo bench --bench core`

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};
use quantumforge::optimizer::{OptimizeConfig, optimize_circuit};
use quantumforge::{Circuit, DensityMatrix, Gate, SmallRng, StateVector};

fn bench_state_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_vector");
    for &n in &[8usize, 12, 16] {
        // Single-qubit gate applied across every wire (the O(2^n) inner loop).
        group.bench_with_input(BenchmarkId::new("apply_all_h", n), &n, |b, &n| {
            let mut sv = StateVector::zero(n).unwrap();
            b.iter(|| {
                for q in 0..n {
                    sv.apply(black_box(Gate::H), q).unwrap();
                }
            });
        });
        // CNOT chain (entangling).
        group.bench_with_input(BenchmarkId::new("cnot_chain", n), &n, |b, &n| {
            let mut sv = StateVector::zero(n).unwrap();
            b.iter(|| {
                for q in 0..n - 1 {
                    sv.apply_cnot(black_box(q), black_box(q + 1)).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_density_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_matrix");
    for &n in &[4usize, 6, 8] {
        group.bench_with_input(BenchmarkId::new("cnot", n), &n, |b, &n| {
            let mut dm = DensityMatrix::from_state(&StateVector::zero(n).unwrap());
            b.iter(|| {
                dm.apply_cnot(black_box(0), black_box(1)).unwrap();
            });
        });
        group.bench_with_input(BenchmarkId::new("from_state", n), &n, |b, &n| {
            let sv = StateVector::zero(n).unwrap();
            b.iter(|| DensityMatrix::from_state(black_box(&sv)));
        });
    }
    group.finish();
}

fn bench_measurement(c: &mut Criterion) {
    c.bench_function("measure_12q", |b| {
        let mut sv = StateVector::zero(12).unwrap();
        sv.apply(Gate::H, 0).unwrap();
        let mut rng = SmallRng::seeded(42);
        b.iter(|| {
            let mut sv = sv.clone();
            sv.measure(black_box(0), &mut rng).unwrap();
        });
    });
}

fn bench_exact_rational(c: &mut Criterion) {
    c.bench_function("rational_add_chain", |b| {
        b.iter(|| {
            let mut acc = Rational::new(1, 3).unwrap();
            for k in 1..50 {
                // Reduced from 200 to avoid i128 overflow
                let r = Rational::new(k, k + 7).unwrap();
                acc = acc.checked_add(black_box(r)).unwrap();
            }
            acc
        });
    });
    c.bench_function("rational_mul_reduce", |b| {
        b.iter(|| {
            let mut acc = Rational::new(6, 35).unwrap();
            for k in 1..50 {
                // Reduced from 200 to avoid i128 overflow
                let r = Rational::new(k + 1, k + 3).unwrap();
                acc = acc.checked_mul(black_box(r)).unwrap();
            }
            acc
        });
    });
    c.bench_function("exact_complex_add_chain", |b| {
        b.iter(|| {
            let mut acc = ExactComplex::real(Rational::new(2, 5).unwrap());
            for k in 1..50 {
                // Reduced from 200 to avoid i128 overflow
                let r = ExactComplex::real(Rational::new(k, k + 11).unwrap());
                acc = acc.checked_add(black_box(r)).unwrap();
            }
            acc
        });
    });
}

fn bench_pauli_sum(c: &mut Criterion) {
    // Build a Heisenberg-like sum over a 4-site chain, then add two copies.
    c.bench_function("pauli_sum_build_add", |b| {
        b.iter(|| {
            let mut sum = PauliSum::zero(4);
            for i in 0..3 {
                for &p in &[Pauli::X, Pauli::Y, Pauli::Z] {
                    let mut factors = vec![Pauli::I; 4];
                    factors[i] = p;
                    factors[i + 1] = p;
                    let word = PauliWord::new(factors);
                    let coeff = ExactComplex::real(Rational::new(1, 2).unwrap());
                    sum.add_term(coeff, word).unwrap();
                }
            }
            sum.checked_add(black_box(&sum)).unwrap()
        });
    });
}

fn bench_circuit_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_optimization");
    for &pairs in &[128usize, 512, 2048] {
        let mut source = Circuit::new();
        for _ in 0..pairs {
            source.push(Gate::H, 0);
            source.push(Gate::H, 0);
        }
        group.bench_with_input(
            BenchmarkId::new("cancel_inverse_pairs", pairs),
            &source,
            |b, source| {
                b.iter_batched(
                    || source.clone(),
                    |mut circuit| {
                        let result =
                            optimize_circuit(&mut circuit, &OptimizeConfig::conservative());
                        black_box((circuit, result));
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    let pairs = 512;
    let mut redundant = Circuit::new();
    for _ in 0..pairs {
        redundant.push(Gate::H, 0);
        redundant.push(Gate::H, 0);
    }
    let (optimized, _) = redundant.optimized();
    let initial = StateVector::zero(12).unwrap();
    group.bench_function("run_redundant_512_pairs", |b| {
        b.iter_batched(
            || initial.clone(),
            |state| black_box(redundant.run(state).unwrap()),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("run_preoptimized_512_pairs", |b| {
        b.iter_batched(
            || initial.clone(),
            |state| black_box(optimized.run(state).unwrap()),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("optimize_then_run_512_pairs", |b| {
        b.iter_batched(
            || (redundant.clone(), initial.clone()),
            |(circuit, state)| black_box(circuit.run_optimized(state).unwrap()),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_state_vector,
    bench_density_matrix,
    bench_measurement,
    bench_exact_rational,
    bench_pauli_sum,
    bench_circuit_optimization,
);
criterion_main!(benches);
