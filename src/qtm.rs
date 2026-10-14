//! Quantum Turing Machine (QTM) — the Bernstein–Vazirani unitary formulation.
//!
//! A classical Turing machine is `(Q, Σ, Γ, δ, q₀, q_accept, q_reject)` with a
//! transition `δ(q, a) = (q', b, D)`. Its quantum extension keeps the same
//! structure but requires the transition function to be **reversible**, compiled
//! here into a unitary `U_δ` acting on `|tape, head, internal-state>`.
//!
//! This module gives you two things on top of the gate-model kernel:
//! 1. a declarative QTM (tape, head, transition table), and
//! 2. a way to run it as a *quantum* circuit: the transition unitary is
//!    synthesised over the kernel's `StateVector`, so a single step is `U_δ`,
//!    and after `t` steps you measure the tape/head/state register.
//!
//! Because the tape is a finite register here, the machine is simulated in the
//! standard "bounded-tape" model: `n` tape cells, each holding a symbol from a
//! small alphabet `Γ`, an `m`-cell head register, and an `h`-qubit internal
//! state. The transition table must be a bijection on reachable configurations
//! for `U_δ` to be unitary (we verify this and report non-reversible tables).

use crate::{Complex64, SmallRng, StateVector, QuantumError};
use std::collections::BTreeMap;

/// One tape symbol / alphabet entry. Kept as a small `u8` alphabet index.
pub type Symbol = u8;

/// Head movement direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
    Stay,
}

/// A single classical transition: in internal state `state`, reading `read`,
/// write `write`, move `dir`, and go to internal state `next`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    pub next_state: usize,
    pub write: Symbol,
    pub direction: Direction,
}

/// A declarative quantum Turing machine over a finite tape.
///
/// `transitions[state][read] -> Transition`. The transition table must be a
/// bijection (each reachable `(state, read)` maps to a distinct
/// `(next_state, write, dir)`); otherwise the unitary synthesis fails with
/// [`QtmError::NotReversible`].
#[derive(Clone, Debug)]
pub struct QuantumTuringMachine {
    /// Number of internal (control) states; must be a power of two (state register).
    pub states: usize,
    /// Alphabet size `|Γ|`; each symbol is `0..alphabet`.
    pub alphabet: usize,
    /// Number of tape cells (a power of two, so the tape is one register).
    pub tape_cells: usize,
    /// Head position register width (covers `tape_cells` positions).
    pub head_cells: usize,
    /// Transition table indexed `[state][read]`.
    pub transitions: Vec<Vec<Option<Transition>>>,
    /// Initial internal state.
    pub initial_state: usize,
}

/// Errors specific to building/running a QTM.
#[derive(Clone, Debug, PartialEq)]
pub enum QtmError {
    NotPowerOfTwo(usize),
    EmptyTransition(usize, Symbol),
    NotReversible,
    StateOutOfRange(usize),
    SymbolOutOfRange(Symbol, usize),
    TapeTooLarge(usize),
}

impl std::fmt::Display for QtmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QtmError::NotPowerOfTwo(n) => write!(f, "{n} is not a power of two"),
            QtmError::EmptyTransition(s, r) => {
                write!(f, "no transition for state {s} reading symbol {r}")
            }
            QtmError::NotReversible => {
                write!(f, "transition table is not a bijection; cannot form a unitary")
            }
            QtmError::StateOutOfRange(s) => write!(f, "state {s} out of range"),
            QtmError::SymbolOutOfRange(r, a) => write!(f, "symbol {r} not in alphabet 0..{a}"),
            QtmError::TapeTooLarge(n) => write!(f, "tape of {n} cells exceeds capacity"),
        }
    }
}
impl std::error::Error for QtmError {}

/// Configuration of a QTM at one instant: full tape, head index, internal state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Configuration {
    pub tape: Vec<Symbol>,
    pub head: usize,
    pub state: usize,
}

impl QuantumTuringMachine {
    /// Build and validate a QTM. `states` and `tape_cells` must be powers of two.
    pub fn new(
        states: usize,
        alphabet: usize,
        tape_cells: usize,
        head_cells: usize,
        transitions: Vec<Vec<Option<Transition>>>,
        initial_state: usize,
    ) -> Result<Self, QtmError> {
        if !states.is_power_of_two() || !tape_cells.is_power_of_two() || !head_cells.is_power_of_two()
        {
            return Err(QtmError::NotPowerOfTwo(if !states.is_power_of_two() {
                states
            } else if !tape_cells.is_power_of_two() {
                tape_cells
            } else {
                head_cells
            }));
        }
        if initial_state >= states {
            return Err(QtmError::StateOutOfRange(initial_state));
        }
        if transitions.len() != states {
            return Err(QtmError::StateOutOfRange(transitions.len()));
        }
        for row in &transitions {
            if row.len() != alphabet {
                return Err(QtmError::SymbolOutOfRange(row.len() as Symbol, alphabet));
            }
        }
        Ok(Self {
            states,
            alphabet,
            tape_cells,
            head_cells,
            transitions,
            initial_state,
        })
    }

    /// Encode a configuration into a computational-basis index. Layout is
    /// strictly little-endian and the exact inverse of [`Self::decode`]:
    /// `[tape: cells*symbol_bits][head: head_bits][state: state_bits]`.
    /// Public so callers can build basis states to feed [`Self::step_quantum`].
    pub fn encode(&self, config: &Configuration) -> Result<usize, QtmError> {
        if config.state >= self.states {
            return Err(QtmError::StateOutOfRange(config.state));
        }
        if config.head >= self.tape_cells {
            return Err(QtmError::StateOutOfRange(config.head));
        }
        let symbol_bits = self.alphabet.trailing_zeros() as usize;
        let tape_bits = self.tape_cells * symbol_bits;
        let head_bits = self.head_cells.trailing_zeros() as usize;

        let mut index = 0usize;
        for (cell, &sym) in config.tape.iter().enumerate() {
            if sym as usize >= self.alphabet {
                return Err(QtmError::SymbolOutOfRange(sym, self.alphabet));
            }
            index |= (sym as usize) << (cell * symbol_bits);
        }
        index |= config.head << tape_bits;
        index |= config.state << (tape_bits + head_bits);
        Ok(index)
    }

    /// Number of qubits needed to hold one configuration.
    pub fn qubits(&self) -> usize {
        let symbol_bits = self.alphabet.trailing_zeros() as usize;
        let tape_bits = self.tape_cells * symbol_bits;
        let head_bits = self.head_cells.trailing_zeros() as usize;
        let state_bits = self.states.trailing_zeros() as usize;
        tape_bits + head_bits + state_bits
    }

    /// The classical successor of a configuration, or `None` if the machine has
    /// no transition for the current `(state, read)`.
    pub fn step_classical(&self, config: &Configuration) -> Option<Configuration> {
        let read = *config.tape.get(config.head)?;
        let t = self.transitions.get(config.state)?.get(read as usize)?;
        let t = t.as_ref()?;
        let mut next = config.clone();
        next.tape[config.head] = t.write;
        match t.direction {
            Direction::Left => {
                next.head = config.head.checked_sub(1)?;
            }
            Direction::Right => {
                next.head = config.head + 1;
                if next.head >= self.tape_cells {
                    return None;
                }
            }
            Direction::Stay => {}
        }
        next.state = t.next_state;
        Some(next)
    }

    /// Run the machine classically for up to `max_steps`, returning the final
    /// configuration (or the halting one when no transition applies).
    pub fn run_classical(&self, initial: &Configuration, max_steps: usize) -> Configuration {
        let mut config = initial.clone();
        for _ in 0..max_steps {
            match self.step_classical(&config) {
                Some(next) => config = next,
                None => break,
            }
        }
        config
    }

    /// Build the transition permutation as a map `basis_index -> basis_index`,
    /// or reject the table if it is not a bijection (which would make `U_δ`
    /// non-unitary). A configuration with no applicable transition — either an
    /// explicit `None` entry or a head move off the tape — is a **fixed point**.
    /// The map is a valid permutation iff every target index is claimed exactly
    /// once across all `dimension` source indices.
    fn transition_permutation(&self) -> Result<BTreeMap<usize, usize>, QtmError> {
        let dimension = 1usize << self.qubits();
        let mut map = BTreeMap::new();
        let mut claimed = vec![false; dimension];
        for index in 0..dimension {
            let config = self.decode(index);
            let target = match self.step_classical(&config) {
                Some(next) => self.encode(&next)?,
                None => index, // halt / off-tape: fixed point
            };
            if claimed[target] {
                return Err(QtmError::NotReversible);
            }
            claimed[target] = true;
            map.insert(index, target);
        }
        Ok(map)
    }

    /// Decode a basis index into a configuration (exact inverse of [`Self::encode`]).
    /// Layout, little-endian: `[tape: cells*symbol_bits][head: head_bits][state: state_bits]`.
    pub fn decode(&self, index: usize) -> Configuration {
        let symbol_bits = self.alphabet.trailing_zeros() as usize;
        let symbol_mask = self.alphabet - 1;
        let head_bits = self.head_cells.trailing_zeros() as usize;
        let tape_bits = self.tape_cells * symbol_bits;
        let state_bits = self.states.trailing_zeros() as usize;

        let tape = (0..self.tape_cells)
            .map(|cell| ((index >> (cell * symbol_bits)) & symbol_mask) as Symbol)
            .collect();
        let head = (index >> tape_bits) & ((1 << head_bits) - 1);
        let state = (index >> (tape_bits + head_bits)) & ((1 << state_bits) - 1);
        Configuration { tape, head, state }
    }

    /// Synthesise the one-step transition unitary `U_δ` as a permutation and
    /// apply it to a state vector. This is the quantum step: it acts on a
    /// superposition of configurations coherently.
    pub fn step_quantum(&self, state: &StateVector) -> Result<StateVector, QtmError> {
        let permutation = self.transition_permutation()?;
        let amplitudes = state.amplitudes().to_vec();
        if amplitudes.len() != 1usize << self.qubits() {
            return Err(QtmError::NotPowerOfTwo(state.qubits()));
        }
        // Apply permutation: new[target] = old[index].
        let mut next = vec![Complex64::ZERO; amplitudes.len()];
        for (&index, &target) in &permutation {
            next[target] = amplitudes[index];
        }
        Ok(StateVector::from_amplitudes(next).map_err(|_| QtmError::NotPowerOfTwo(0))?)
    }

    /// Prepare the uniform superposition over all configurations sharing a
    /// given tape and head but every internal state — a common QTM initial
    /// condition for search oracles.
    pub fn prepare_superposition(
        &self,
        tape: &[Symbol],
        head: usize,
    ) -> Result<StateVector, QtmError> {
        let qubits = self.qubits();
        let dimension = 1usize << qubits;
        let mut amplitudes = vec![Complex64::ZERO; dimension];
        let amp = Complex64::new(1.0 / (self.states as f64).sqrt(), 0.0);
        for state in 0..self.states {
            let config = Configuration {
                tape: tape.to_vec(),
                head,
                state,
            };
            let index = self.encode(&config)?;
            amplitudes[index] = amp;
        }
        Ok(StateVector::from_amplitudes(amplitudes).map_err(|_| QtmError::NotPowerOfTwo(0))?)
    }

    /// Measure the internal-state register after `steps` quantum steps, seeded.
    /// Returns the sampled state index and the (collapsed) post-measurement state.
    pub fn run_quantum_measured(
        &self,
        initial: &StateVector,
        steps: usize,
        seed: u64,
    ) -> Result<(usize, StateVector), QuantumError> {
        let mut state = initial.clone();
        for _ in 0..steps {
            state = self
                .step_quantum(&state)
                .map_err(|_| QuantumError::OperationRequiresShotExecution)?;
        }
        let mut rng = SmallRng::seeded(seed);
        // Measure the top `state_bits` qubits (the internal state register).
        let state_bits = self.states.trailing_zeros() as usize;
        let total_qubits = self.qubits();
        let mut measured_state = 0usize;
        for offset in 0..state_bits {
            let qubit = total_qubits - 1 - offset;
            let bit = state.measure(qubit, &mut rng)?;
            measured_state |= (bit as usize) << offset;
        }
        Ok((measured_state, state))
    }
}
