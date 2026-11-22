//! Noise simulation and threshold estimation for stabilizer codes.
//!
//! This module injects Pauli noise on the data qubits, optionally flips the
//! syndrome bits (measurement error), decodes with an exact minimum-weight
//! decoder, and checks whether the net Pauli is a nontrivial logical operator.
//! Sweeping the physical error rate over a family of codes gives a
//! logical-error curve per code size; [`threshold_crossing`] locates the
//! threshold where the curves intersect.
//!
//! The model is phenomenological (data errors plus noisy syndrome readout),
//! which is the standard testbed for a decoder. A full circuit-level
//! simulation would additionally noise every gate in the syndrome circuit,
//! which the current `QecCode` builders do not expose.
//!
//! # Decoder
//!
//! [`DecodingTable`] precomputes, for every syndrome, the minimum-weight Pauli
//! error that produces it: a breadth-first search over the `2^s` syndrome
//! states (`s` = number of stabilizers), where each step applies one
//! single-qubit Pauli. This is exact minimum-weight decoding, the right
//! decoder for depolarizing noise; [`Decoder::BitFlip`] restricts the
//! transitions to X errors, the right decoder for bit-flip noise.

use super::{PauliString, QecCode};
use crate::SmallRng;
use std::collections::VecDeque;

/// Pauli labels used across the crate: `0 = I`, `1 = X`, `2 = Y`, `3 = Z`.
const PAULI_COUNT: usize = 4;

/// Multiplication table for single-qubit Pauli labels, ignoring phase.
///
/// `PRODUCT[a][b]` is the label of the product `a * b` (the phase is irrelevant
/// for commutation and syndrome checks).
const PRODUCT: [[u8; PAULI_COUNT]; PAULI_COUNT] =
    [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]];

/// Which single-qubit Paulis the decoder is allowed to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoder {
    /// X, Y, and Z errors, each of weight one: minimum-weight decoding for
    /// depolarizing noise.
    Depolarizing,
    /// X errors only: minimum-weight decoding for bit-flip noise.
    BitFlip,
}

/// Single-qubit Pauli noise: the probability of an X, Y, or Z error.
///
/// The probabilities must sum to at most one; the remaining probability is the
/// identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PauliNoise {
    /// Probability of an X error.
    pub p_x: f64,
    /// Probability of a Y error.
    pub p_y: f64,
    /// Probability of a Z error.
    pub p_z: f64,
}

impl PauliNoise {
    /// Depolarizing noise with total error probability `p`: `p/3` on each of
    /// X, Y, and Z.
    pub fn depolarizing(p: f64) -> Self {
        Self {
            p_x: p / 3.0,
            p_y: p / 3.0,
            p_z: p / 3.0,
        }
    }

    /// Bit-flip noise: probability `p` of an X error and nothing else.
    pub fn bit_flip(p: f64) -> Self {
        Self {
            p_x: p,
            p_y: 0.0,
            p_z: 0.0,
        }
    }

    /// Total error probability `p_x + p_y + p_z`.
    pub fn total(&self) -> f64 {
        self.p_x + self.p_y + self.p_z
    }

    /// Probability of a specific Pauli label.
    pub fn probability(&self, pauli: u8) -> f64 {
        match pauli {
            0 => (1.0 - self.total()).max(0.0),
            1 => self.p_x,
            2 => self.p_y,
            3 => self.p_z,
            _ => 0.0,
        }
    }

    /// Sample one Pauli label.
    pub fn sample(&self, rng: &mut SmallRng) -> u8 {
        let sample = rng.next_f64();
        let mut cumulative = 0.0;
        for pauli in 1..PAULI_COUNT as u8 {
            cumulative += self.probability(pauli);
            if sample < cumulative {
                return pauli;
            }
        }
        0
    }
}

/// A noise model: data-qubit Pauli noise plus a syndrome-bit flip probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseModel {
    /// Noise on the data qubits before syndrome extraction.
    pub data: PauliNoise,
    /// Probability that a syndrome bit is flipped (readout error).
    pub measurement: f64,
}

/// A logical-error-rate curve: one physical error rate per entry.
#[derive(Clone, Debug)]
pub struct ThresholdCampaign {
    /// Physical error rates, in increasing order.
    pub rates: Vec<f64>,
    /// Logical error rate at each rate.
    pub logical_error_rates: Vec<f64>,
    /// Monte Carlo trials per rate.
    pub trials: usize,
}

/// Syndrome of `error`: bit `i` is set when `error` anticommutes with
/// stabilizer `i`.
pub fn syndrome_of(error: &PauliString, code: &QecCode) -> u32 {
    let mut syndrome = 0u32;
    for (index, stabilizer) in code.stabilizers.iter().enumerate() {
        if !error.commutes(stabilizer) {
            syndrome |= 1 << index;
        }
    }
    syndrome
}

/// True when `net` is a nontrivial logical operator: it anticommutes with at
/// least one logical operator. A stabilizer commutes with every logical, so it
/// is not counted as an error.
pub fn is_logical_error(net: &PauliString, code: &QecCode) -> bool {
    code.logical_x
        .iter()
        .chain(code.logical_z.iter())
        .any(|logical| !net.commutes(logical))
}

/// Multiply two Pauli strings qubit-wise, ignoring phase.
pub fn multiply_pauli_strings(first: &PauliString, second: &PauliString) -> PauliString {
    let length = first.paulis.len().min(second.paulis.len());
    let mut product = PauliString::new(first.paulis.len());
    for qubit in 0..length {
        product.paulis[qubit] =
            PRODUCT[first.paulis[qubit] as usize][second.paulis[qubit] as usize];
    }
    product
}

/// Exact minimum-weight decoder for a code, precomputed for every syndrome.
pub struct DecodingTable {
    /// Number of data qubits.
    qubits: usize,
    /// Number of syndrome states (`2^stabilizers`).
    states: usize,
    /// For each syndrome state, the transition `(qubit, pauli, previous_state)`
    /// on a shortest path from the zero syndrome.
    parent: Vec<Option<(usize, u8, usize)>>,
}

impl DecodingTable {
    /// Build the table for `code` using the allowed Paulis of `decoder`.
    ///
    /// The number of syndrome states is `2^stabilizers`; building the table is
    /// `O(2^s * 3n)`. For the codes in this crate (`s <= 12`) that is small.
    pub fn new(code: &QecCode, decoder: Decoder) -> Self {
        let qubits = code.n_physical;
        let states = 1usize << code.stabilizers.len();
        let mut transitions: Vec<(usize, u8, usize)> = Vec::new();
        for qubit in 0..qubits {
            for pauli in 1..PAULI_COUNT as u8 {
                if decoder == Decoder::BitFlip && pauli != 1 {
                    continue;
                }
                let mut error = PauliString::new(qubits);
                error.set(qubit, pauli);
                let syndrome = syndrome_of(&error, code);
                if syndrome != 0 {
                    transitions.push((qubit, pauli, syndrome as usize));
                }
            }
        }
        let mut parent: Vec<Option<(usize, u8, usize)>> = vec![None; states];
        let mut seen = vec![false; states];
        seen[0] = true;
        let mut queue = VecDeque::new();
        queue.push_back(0usize);
        while let Some(state) = queue.pop_front() {
            for &(qubit, pauli, syndrome) in &transitions {
                let next = state ^ syndrome;
                if !seen[next] {
                    seen[next] = true;
                    parent[next] = Some((qubit, pauli, state));
                    queue.push_back(next);
                }
            }
        }
        Self {
            qubits,
            states,
            parent,
        }
    }

    /// Decode a syndrome into the minimum-weight Pauli error that produces it.
    ///
    /// Returns the identity when the syndrome is unreachable (no combination of
    /// the allowed Paulis produces it).
    pub fn decode(&self, syndrome: u32) -> PauliString {
        let mut net = PauliString::new(self.qubits);
        let mut state = syndrome as usize;
        if state >= self.states {
            return net;
        }
        while state != 0 {
            let Some((qubit, pauli, previous)) = self.parent[state] else {
                // Unreachable syndrome: no combination of the allowed Paulis
                // produces it. Returning the identity is a decoding failure,
                // which the caller counts as it would any other.
                return PauliString::new(self.qubits);
            };
            net.paulis[qubit] = PRODUCT[net.paulis[qubit] as usize][pauli as usize];
            state = previous;
        }
        net
    }
}

/// Monte Carlo logical error rate of `code` under `model`.
///
/// Each trial injects independent data-qubit Pauli noise, flips each syndrome
/// bit with probability `model.measurement`, decodes with `table`, and counts a
/// failure when the net Pauli (injected error times decoder recovery) is a
/// nontrivial logical operator.
pub fn logical_error_rate(
    code: &QecCode,
    model: &NoiseModel,
    table: &DecodingTable,
    trials: usize,
    seed: u64,
) -> f64 {
    if trials == 0 {
        return 0.0;
    }
    let mut rng = SmallRng::seeded(seed);
    let mut failures = 0usize;
    for _ in 0..trials {
        let mut error = PauliString::new(code.n_physical);
        for qubit in 0..code.n_physical {
            error.set(qubit, model.data.sample(&mut rng));
        }
        let mut syndrome = syndrome_of(&error, code);
        for bit in 0..code.stabilizers.len() {
            if rng.next_f64() < model.measurement {
                syndrome ^= 1 << bit;
            }
        }
        let recovery = table.decode(syndrome);
        let net = multiply_pauli_strings(&error, &recovery);
        if is_logical_error(&net, code) {
            failures += 1;
        }
    }
    failures as f64 / trials as f64
}

/// Sweep depolarizing data noise with a fixed syndrome-readout error rate,
/// using [`Decoder::Depolarizing`].
pub fn depolarizing_sweep_with_measurement(
    code: &QecCode,
    rates: &[f64],
    measurement: f64,
    trials: usize,
    seed: u64,
) -> ThresholdCampaign {
    let table = DecodingTable::new(code, Decoder::Depolarizing);
    let logical_error_rates = rates
        .iter()
        .enumerate()
        .map(|(index, &rate)| {
            let model = NoiseModel {
                data: PauliNoise::depolarizing(rate),
                measurement,
            };
            logical_error_rate(
                code,
                &model,
                &table,
                trials,
                seed.wrapping_add(index as u64 * 1_000_003),
            )
        })
        .collect();
    ThresholdCampaign {
        rates: rates.to_vec(),
        logical_error_rates,
        trials,
    }
}

/// Sweep depolarizing data noise with the measurement rate equal to the data
/// rate, the standard circuit-level-like model.
pub fn depolarizing_sweep(
    code: &QecCode,
    rates: &[f64],
    trials: usize,
    seed: u64,
) -> ThresholdCampaign {
    let table = DecodingTable::new(code, Decoder::Depolarizing);
    let logical_error_rates = rates
        .iter()
        .enumerate()
        .map(|(index, &rate)| {
            let model = NoiseModel {
                data: PauliNoise::depolarizing(rate),
                measurement: rate,
            };
            logical_error_rate(
                code,
                &model,
                &table,
                trials,
                seed.wrapping_add(index as u64 * 1_000_003),
            )
        })
        .collect();
    ThresholdCampaign {
        rates: rates.to_vec(),
        logical_error_rates,
        trials,
    }
}

/// Sweep bit-flip data noise with perfect syndrome readout, using
/// [`Decoder::BitFlip`].
pub fn bitflip_sweep(code: &QecCode, rates: &[f64], trials: usize, seed: u64) -> ThresholdCampaign {
    let table = DecodingTable::new(code, Decoder::BitFlip);
    let logical_error_rates = rates
        .iter()
        .enumerate()
        .map(|(index, &rate)| {
            let model = NoiseModel {
                data: PauliNoise::bit_flip(rate),
                measurement: 0.0,
            };
            logical_error_rate(
                code,
                &model,
                &table,
                trials,
                seed.wrapping_add(index as u64 * 1_000_003),
            )
        })
        .collect();
    ThresholdCampaign {
        rates: rates.to_vec(),
        logical_error_rates,
        trials,
    }
}

/// Estimate the threshold from campaigns of increasing code size.
///
/// For every adjacent pair of campaigns, each sign change of
/// `rate_a(p) - rate_b(p)` gives a linearly interpolated crossing; the returned
/// value is the mean of those crossings. Returns `None` when there are fewer
/// than two campaigns or no crossing was found.
pub fn threshold_crossing(campaigns: &[ThresholdCampaign]) -> Option<f64> {
    if campaigns.len() < 2 {
        return None;
    }
    let mut crossings = Vec::new();
    for pair in campaigns.windows(2) {
        let first = &pair[0];
        let second = &pair[1];
        if first.rates.len() < 2 || first.rates.len() != second.rates.len() {
            return None;
        }
        for index in 0..first.rates.len() - 1 {
            let difference_a = first.logical_error_rates[index] - second.logical_error_rates[index];
            let difference_b =
                first.logical_error_rates[index + 1] - second.logical_error_rates[index + 1];
            if difference_a == 0.0 {
                crossings.push(first.rates[index]);
            } else if difference_a * difference_b < 0.0 {
                let span = first.rates[index + 1] - first.rates[index];
                crossings
                    .push(first.rates[index] + span * difference_a / (difference_a - difference_b));
            }
        }
    }
    if crossings.is_empty() {
        None
    } else {
        Some(crossings.iter().sum::<f64>() / crossings.len() as f64)
    }
}
