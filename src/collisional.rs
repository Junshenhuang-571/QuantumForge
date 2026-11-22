//! Collisional (repeated-interaction) thermalization.
//!
//! A system collides with a fresh bath copy each step through the partial-swap
//! unitary `V = cos(phi) I - i sin(phi) S`. Tracing out the bath gives the
//! exactly solvable channel
//!
//! ```text
//! E(rho) = cos^2(phi) rho + sin^2(phi) tau + i cos(phi) sin(phi) [rho, tau],
//! ```
//!
//! where `tau` is the bath's reduced state. For diagonal states the cross term
//! vanishes and the channel is the convex mixture `cos^2(phi) rho +
//! sin^2(phi) tau`, so the fixed point is `tau` and the population converges at
//! `cos^2(phi)` per collision. Unlike the unital depolarizing channel this is a
//! genuine thermalizing map: it has a non-trivial fixed point and an detailed
//! -balance interpretation when `tau` is a Gibbs state.

use crate::{Complex64, DensityMatrix, QuantumError};

/// The population convergence rate per collision, `cos^2(phi)`.
pub fn collision_rate(phi: f64) -> f64 {
    phi.cos().powi(2)
}

/// One collision against a bath in `tau`.
pub fn collisional_channel(
    rho: &DensityMatrix,
    bath: &DensityMatrix,
    phi: f64,
) -> Result<DensityMatrix, QuantumError> {
    let dimension = rho.dimension();
    if bath.dimension() != dimension || bath.qubits() != rho.qubits() {
        return Err(QuantumError::InvalidSubsystemSelection);
    }
    let cosine = phi.cos();
    let sine = phi.sin();
    let c2 = cosine * cosine;
    let s2 = sine * sine;
    let cross = Complex64::new(0.0, cosine * sine);
    let system = rho.entries();
    let equilibrium = bath.entries();
    let mut out = vec![Complex64::ZERO; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            // [rho, tau]_{row,col} = sum_k rho_{row,k} tau_{k,col} - tau_{row,k} rho_{k,col}
            let mut commutator = Complex64::ZERO;
            for inner in 0..dimension {
                commutator = commutator
                    + system[row * dimension + inner] * equilibrium[inner * dimension + col]
                    - equilibrium[row * dimension + inner] * system[inner * dimension + col];
            }
            out[row * dimension + col] = system[row * dimension + col] * c2
                + equilibrium[row * dimension + col] * s2
                + commutator * cross;
        }
    }
    Ok(DensityMatrix::from_entries(out, rho.qubits()))
}

/// One collision in place.
pub fn collisional_step(
    rho: &mut DensityMatrix,
    bath: &DensityMatrix,
    phi: f64,
) -> Result<(), QuantumError> {
    *rho = collisional_channel(rho, bath, phi)?;
    Ok(())
}

/// Repeated collisions against a fresh bath copy each step.
pub fn thermalize_collisions(
    rho: &mut DensityMatrix,
    bath: &DensityMatrix,
    phi: f64,
    steps: usize,
) -> Result<(), QuantumError> {
    for _ in 0..steps {
        collisional_step(rho, bath, phi)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Complex64, StateVector};

    fn diagonal(prob_one: f64) -> DensityMatrix {
        let mut amplitudes = vec![Complex64::ZERO; 2];
        amplitudes[0] = Complex64::new((1.0 - prob_one).sqrt(), 0.0);
        amplitudes[1] = Complex64::new(prob_one.sqrt(), 0.0);
        let mut rho = DensityMatrix::from_state(&StateVector::from_amplitudes(amplitudes).unwrap());
        // Force an exact diagonal state: zero the coherence.
        let entries = rho.entries().to_vec();
        let mut diagonal = vec![Complex64::ZERO; 4];
        diagonal[0] = entries[0];
        diagonal[3] = entries[3];
        rho = DensityMatrix::from_entries(diagonal, 1);
        rho
    }

    #[test]
    fn fixed_point_is_the_bath_and_rate_is_cos_squared() {
        let bath = diagonal(0.2689414213699951);
        let mut rho = diagonal(0.9);
        let phi = 0.35;
        let rate = collision_rate(phi);
        let expected = bath.probability_one(0).unwrap();
        let initial = rho.probability_one(0).unwrap();
        for step in 1..=13 {
            collisional_step(&mut rho, &bath, phi).unwrap();
            let predicted = expected + (initial - expected) * rate.powi(step);
            assert!(
                (rho.probability_one(0).unwrap() - predicted).abs() < 1e-12,
                "step {step}: {} vs {predicted}",
                rho.probability_one(0).unwrap()
            );
        }
    }

    #[test]
    fn a_fixed_point_state_is_unchanged() {
        let bath = diagonal(0.2689414213699951);
        let mut rho = bath.clone();
        collisional_step(&mut rho, &bath, 0.7).unwrap();
        let difference: f64 = rho
            .entries()
            .iter()
            .zip(bath.entries().iter())
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum();
        assert!(difference < 1e-24);
    }
}
