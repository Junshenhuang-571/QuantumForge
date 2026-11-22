//! Quantum noise as statistical mechanics: a probe of the existing surfaces.
//!
//! ```bash
//! cargo run --release --example noise_thermo_probe
//! ```
//!
//! Three measurements, all on public API:
//!
//! A. Entropy production under the three built-in channels. The H-theorem
//!    (monotone entropy increase) is a statement about *unital* channels; the
//!    amplitude-damping channel is non-unital and is a refrigerator, so its
//!    system entropy is not monotone.
//!
//! B. Collisional thermalization. A partial-swap unitary against a bath held
//!    in a Gibbs state is a thermalizing channel; its fixed point is the bath
//!    state and its population rate is `cos^2(phi)` per collision. This is the
//!    statistical-mechanics use of a noise channel: noise as a thermostat.
//!
//! C. What the built-in amplitude-damping channel is missing: it is the `T=0`
//!    limit (fixed point `|0><0|`). A finite-temperature channel needs an
//!    excitation branch with `gamma_up / gamma_down = exp(-beta * Delta)`.

use quantumforge::quantum::{HilbertSpace, Operator};
use quantumforge::{Complex64, DensityMatrix, Gate, NoiseChannel, StateVector};

/// Von Neumann entropy of a single-qubit density matrix, from its eigenvalues.
fn von_neumann_1qubit(rho: &DensityMatrix) -> f64 {
    let e = rho.entries();
    let det = (e[0] * e[3] - e[1] * e[2]).re;
    let disc = (1.0 - 4.0 * det).max(0.0);
    let lambda_plus = 0.5 * (1.0 + disc.sqrt());
    let lambda_minus = 0.5 * (1.0 - disc.sqrt());
    let mut entropy = 0.0;
    for lambda in [lambda_plus, lambda_minus] {
        if lambda > 1e-15 {
            entropy -= lambda * lambda.ln();
        }
    }
    entropy
}

fn plus_state() -> StateVector {
    let mut state = StateVector::zero(1).expect("one qubit");
    state.apply(Gate::H, 0).expect("hadamard");
    state
}

fn one_state() -> StateVector {
    let mut state = StateVector::zero(1).expect("one qubit");
    state.apply(Gate::X, 0).expect("pauli x");
    state
}

/// The partial-swap unitary `V = cos(phi) I - i sin(phi) S` on two qubits.
fn partial_swap(phi: f64) -> Operator {
    let space = HilbertSpace::qubits(2).expect("two-qubit space");
    let c = phi.cos();
    let s = phi.sin();
    let mut entries = vec![Complex64::ZERO; 16];
    for index in 0..4usize {
        entries[index * 4 + index] = Complex64::new(c, 0.0);
    }
    for index in 0..4usize {
        let swapped = ((index & 1) << 1) | ((index >> 1) & 1);
        entries[swapped * 4 + index] = entries[swapped * 4 + index] + Complex64::new(0.0, -s);
    }
    Operator::new(space.clone(), space, entries).expect("partial swap operator")
}

/// Joint state `rho (system, qubit 0) tensor tau (bath, qubit 1)`.
fn joint(rho: &DensityMatrix, tau: &DensityMatrix) -> DensityMatrix {
    let r = rho.entries();
    let t = tau.entries();
    let mut entries = vec![Complex64::ZERO; 16];
    for i0 in 0..2 {
        for i1 in 0..2 {
            for j0 in 0..2 {
                for j1 in 0..2 {
                    let row = i0 + 2 * i1;
                    let col = j0 + 2 * j1;
                    entries[row * 4 + col] = r[i0 * 2 + j0] * t[i1 * 2 + j1];
                }
            }
        }
    }
    DensityMatrix::from_entries(entries, 2)
}

/// One collision: `rho -> Tr_bath[ V (rho tensor tau) V^dagger ]`.
fn collide(rho: &DensityMatrix, bath: &DensityMatrix, unitary: &Operator) -> DensityMatrix {
    let mut state = joint(rho, bath);
    state
        .apply_unitary(unitary, 1e-10)
        .expect("unitary collision");
    state.partial_trace(&[0]).expect("trace out bath")
}

/// Bath Gibbs state `exp(-beta * Delta * Z / 2) / Z`, so `p1/p0 = exp(-beta*Delta)`.
fn gibbs_bath(beta_delta: f64) -> DensityMatrix {
    let weight = (-beta_delta).exp();
    let partition = 1.0 + weight;
    DensityMatrix::from_entries(
        vec![
            Complex64::new(1.0 / partition, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(weight / partition, 0.0),
        ],
        1,
    )
}

fn main() {
    println!("=== A. entropy production under the built-in channels ===");
    println!("start |+> (pure, S = 0), 12 applications\n");
    let channels = [
        (
            "depolarizing p=0.10",
            NoiseChannel::Depolarizing { probability: 0.10 },
        ),
        (
            "amplitude damping g=0.10",
            NoiseChannel::AmplitudeDamping { gamma: 0.10 },
        ),
        (
            "phase damping g=0.10",
            NoiseChannel::PhaseDamping { gamma: 0.10 },
        ),
    ];
    for (name, channel) in channels {
        let mut rho = DensityMatrix::from_state(&plus_state());
        println!("{name}");
        println!("  {:>3}  {:>10}  {:>10}  {:>10}", "t", "purity", "p1", "S");
        for step in 0..=12 {
            let p1 = rho.probability_one(0).expect("p1");
            println!(
                "  {:>3}  {:>10.6}  {:>10.6}  {:>10.6}",
                step,
                rho.purity(),
                p1,
                von_neumann_1qubit(&rho)
            );
            rho.apply_channel(channel, 0).expect("channel");
        }
        println!();
    }

    println!("=== B. collisional thermalization: partial swap against a Gibbs bath ===");
    let beta_delta = 1.0f64; // p1/p0 = e^{-1}, so p1 = 0.26894142
    let phi = 0.35f64; // population rate cos^2(phi) = 0.8875626
    let bath = gibbs_bath(beta_delta);
    let unitary = partial_swap(phi);
    let target = bath.probability_one(0).expect("bath p1");
    let rate = phi.cos().powi(2);
    println!("bath beta*Delta = {beta_delta}, target p1 = {target:.9}, cos^2(phi) = {rate:.9}\n");
    for (label, initial) in [("|1>", one_state()), ("|+>", plus_state())] {
        let mut rho = DensityMatrix::from_state(&initial);
        let p1_zero = rho.probability_one(0).expect("p1(0)");
        println!("start {label}");
        println!(
            "  {:>3}  {:>12}  {:>12}  {:>12}  {:>12}",
            "t", "p1", "predicted", "|rho01|", "S"
        );
        for step in 0..=12 {
            let p1 = rho.probability_one(0).expect("p1");
            let coherence = rho.entries()[1].norm_sqr().sqrt();
            let predicted = target + (p1_zero - target) * rate.powi(step);
            println!(
                "  {:>3}  {:>12.9}  {:>12.9}  {:>12.9}  {:>12.9}",
                step,
                p1,
                predicted,
                coherence,
                von_neumann_1qubit(&rho)
            );
            rho = collide(&rho, &bath, &unitary);
        }
        let fitted =
            ((rho.probability_one(0).expect("p1") - target) / (p1_zero - target)).powf(1.0 / 13.0);
        println!("  fitted per-step population factor after 13 collisions: {fitted:.9}\n");
    }

    println!("=== C. the built-in amplitude-damping channel is the T=0 limit ===");
    let mut rho = DensityMatrix::from_state(&one_state());
    println!("start |1>, amplitude damping gamma = 0.10 (predicted p1 = 0.9^t)");
    for step in 0..=8 {
        println!(
            "  t={}  p1 = {:.9}  predicted = {:.9}",
            step,
            rho.probability_one(0).expect("p1"),
            0.9f64.powi(step)
        );
        rho.apply_channel(NoiseChannel::AmplitudeDamping { gamma: 0.10 }, 0)
            .expect("damping");
    }
    println!("\nfixed point is |0><0| (T=0 Gibbs). A finite-temperature channel needs");
    println!("an excitation branch with gamma_up/gamma_down = exp(-beta*Delta);");
    println!("part D uses the built-in ThermalDamping branch added for this work.");

    println!("\n=== D. the built-in ThermalDamping channel ===");
    let gamma_down = 0.10;
    let beta_delta = 1.0;
    let channel = NoiseChannel::ThermalDamping {
        gamma_down,
        beta_delta,
    };
    let gamma_up = gamma_down * (-beta_delta).exp();
    let target = (-beta_delta).exp() / (1.0 + (-beta_delta).exp());
    let rate = 1.0 - gamma_up - gamma_down;
    println!("gamma_up = {gamma_up:.9}, target p1 = {target:.9}, rate = {rate:.9}");
    let mut rho = DensityMatrix::from_state(&one_state());
    println!(
        "  {:>3}  {:>12}  {:>12}  {:>12}",
        "t", "p1", "predicted", "S"
    );
    for step in 0..=12 {
        let p1 = rho.probability_one(0).unwrap();
        let predicted = target + (1.0 - target) * rate.powi(step);
        println!(
            "  {:>3}  {:>12.9}  {:>12.9}  {:>12.9}",
            step,
            p1,
            predicted,
            rho.von_neumann_entropy().unwrap()
        );
        rho.apply_channel(channel, 0).unwrap();
    }
}
