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
//! state. Head moves **wrap** modulo the tape, so a total transition table whose
//! writes are injective forms a unitary directly ([`QuantumTuringMachine::step_quantum`]).
//! For machines whose tape writes are *not* injective (an information-erasing
//! write), [`QuantumTuringMachine::step_quantum_ancilla`] runs on an
//! ancilla-augmented register that preserves the erased symbol — a reversible
//! involution (apply once to advance, twice to return), which is the practical
//! form of Bennett's trick for a single step, not an unbounded history tape.
//!
//! Use [`QuantumTuringMachine::is_buildable`] to check the state register fits
//! and [`QuantumTuringMachine::reversibility_gap`] to diagnose a table that
//! is not a bijection.

use crate::{Complex64, QuantumError, SmallRng, StateVector};
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
    /// The internal-state register is too small to encode `states` injectively.
    StateRegisterTooSmall {
        states: usize,
        bits: usize,
    },
    /// Ancilla tape too small for the uncompute register (need ≥ tape_cells).
    AncillaTooSmall {
        need: usize,
        got: usize,
    },
}

impl std::fmt::Display for QtmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QtmError::NotPowerOfTwo(n) => write!(f, "{n} is not a power of two"),
            QtmError::EmptyTransition(s, r) => {
                write!(f, "no transition for state {s} reading symbol {r}")
            }
            QtmError::NotReversible => {
                write!(
                    f,
                    "transition table is not a bijection; cannot form a unitary"
                )
            }
            QtmError::StateOutOfRange(s) => write!(f, "state {s} out of range"),
            QtmError::SymbolOutOfRange(r, a) => write!(f, "symbol {r} not in alphabet 0..{a}"),
            QtmError::TapeTooLarge(n) => write!(f, "tape of {n} cells exceeds capacity"),
            QtmError::StateRegisterTooSmall { states, bits } => write!(
                f,
                "state register of {bits} bits cannot encode {states} states injectively"
            ),
            QtmError::AncillaTooSmall { need, got } => write!(
                f,
                "ancilla tape of {got} cells is too small; need at least {need}"
            ),
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
        if !states.is_power_of_two()
            || !tape_cells.is_power_of_two()
            || !head_cells.is_power_of_two()
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
    ///
    /// Head moves **wrap** modulo the tape length (`Right` off the end returns
    /// to cell 0, `Left` off the start to the last cell). Wrapping keeps every
    /// head move a bijection, which is what lets a total transition table form
    /// a unitary without extra machinery. Use [`Self::permutation_with_ancilla`]
    /// for machines whose tape writes are not injective.
    pub fn step_classical(&self, config: &Configuration) -> Option<Configuration> {
        let read = *config.tape.get(config.head)?;
        let t = self.transitions.get(config.state)?.get(read as usize)?;
        let t = t.as_ref()?;
        let mut next = config.clone();
        next.tape[config.head] = t.write;
        next.head = match t.direction {
            Direction::Left => (config.head + self.tape_cells - 1) % self.tape_cells,
            Direction::Right => (config.head + 1) % self.tape_cells,
            Direction::Stay => config.head,
        };
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
    /// non-unitary). A configuration with no applicable transition (an explicit
    /// `None` entry) is a **fixed point**. The map is a valid permutation iff
    /// every target index is claimed exactly once across all source indices.
    fn transition_permutation(&self) -> Result<BTreeMap<usize, usize>, QtmError> {
        let dimension = 1usize << self.qubits();
        let mut map = BTreeMap::new();
        let mut claimed = vec![false; dimension];
        for index in 0..dimension {
            let config = self.decode(index);
            let target = match self.step_classical(&config) {
                Some(next) => self.encode(&next)?,
                None => index, // no transition: fixed point (halt)
            };
            if claimed[target] {
                return Err(QtmError::NotReversible);
            }
            claimed[target] = true;
            map.insert(index, target);
        }
        Ok(map)
    }

    /// Whether this machine can be synthesised as a unitary on the plain
    /// configuration register. Requires `states ≤ 2^head_bits` (the internal
    /// state must fit injectively in the state register). Machines that also
    /// need ancilla tape are checked via [`Self::permutation_with_ancilla`].
    pub fn is_buildable(&self) -> Result<(), QtmError> {
        let head_bits = self.head_cells.trailing_zeros() as usize;
        if self.states > (1usize << head_bits) {
            return Err(QtmError::StateRegisterTooSmall {
                states: self.states,
                bits: head_bits,
            });
        }
        Ok(())
    }

    /// Number of configurations whose transition collides with another's (a
    /// bijection would have zero). A total, injective transition table gives 0;
    /// a partial table contributes one collision per gap against the fixed
    /// points it lands on. Useful for diagnosing *why* a table is not unitary.
    pub fn reversibility_gap(&self) -> usize {
        let dimension = 1usize << self.qubits();
        let mut claimed = vec![false; dimension];
        let mut gaps = 0usize;
        for index in 0..dimension {
            let config = self.decode(index);
            let target = match self.step_classical(&config) {
                Some(next) => match self.encode(&next) {
                    Ok(t) => t,
                    Err(_) => index,
                },
                None => index,
            };
            if claimed[target] {
                gaps += 1;
            } else {
                claimed[target] = true;
            }
        }
        gaps
    }

    /// Bennett-style ancilla augmentation: build the one-step transition map on
    /// an enlarged register that appends `extra_cells` ancilla cells. Each step
    /// saves the symbol read at the head into ancilla slot `head` (storing
    /// `read + 1`, so `0` means "empty") *before* overwriting it, which is what
    /// makes an information-erasing write injective.
    ///
    /// The returned map is a **whole-space involution**: each clean-ancilla
    /// source swaps with its ancilla-dirty target (`src ↔ tgt`), everything
    /// else is fixed. So applying `U_δ` once advances a clean state into the
    /// ancilla record; applying it twice returns to the start (the inverse is
    /// the same map). This is the reversible-step primitive that
    /// [`Self::step_quantum_ancilla`] applies coherently — note it is a
    /// *swap*, not a multi-step history tape, so it is "reversible ancilla
    /// stepping" rather than full unbounded Bennett uncomputation.
    ///
    /// Requires `extra_cells == tape_cells` (one ancilla slot per head
    /// position). Rejects a tape large enough to exhaust memory.
    pub fn permutation_with_ancilla(
        &self,
        extra_cells: usize,
    ) -> Result<(BTreeMap<usize, usize>, usize), QtmError> {
        let symbol_bits = self.alphabet.trailing_zeros() as usize;
        // The ancilla register holds one slot per work-tape head position, so
        // the enlarged tape is exactly `tape_cells (work) + extra_cells (ancilla)`
        // with `extra_cells == tape_cells`; the head ranges only over the work
        // region and wraps modulo `tape_cells`.
        if extra_cells != self.tape_cells {
            return Err(QtmError::AncillaTooSmall {
                need: self.tape_cells,
                got: extra_cells,
            });
        }
        let work = self.tape_cells;
        // Each ancilla slot stores a symbol `0..=alphabet` (the read value plus
        // an "empty = 0" sentinel), so it needs one more bit than a tape cell.
        let ancilla_bits = symbol_bits + 1;
        let head_bits = work.trailing_zeros() as usize;
        // Layout (little-endian): [work tape: work*symbol_bits][ancilla:
        // work*ancilla_bits][head: head_bits][state: state_bits].
        let work_bits = work * symbol_bits;
        let ancilla_region_bits = work * ancilla_bits;
        let ancilla_off = work_bits;
        let head_off = ancilla_off + ancilla_region_bits;
        let state_off = head_off + head_bits;
        let state_bits = self.states.trailing_zeros() as usize;
        let total_qubits = state_off + state_bits;
        if total_qubits > 24 {
            return Err(QtmError::TapeTooLarge(total_qubits));
        }
        let dimension = 1usize << total_qubits;
        let ancilla_mask = (1usize << ancilla_bits) - 1;

        let encode = |tape: &[Symbol], anc: &[usize], head: usize, state: usize| -> usize {
            let mut index = 0usize;
            for (cell, &sym) in tape[..work].iter().enumerate() {
                index |= (sym as usize) << (cell * symbol_bits);
            }
            for (slot, &a) in anc.iter().enumerate() {
                index |= a << (ancilla_off + slot * ancilla_bits);
            }
            index |= head << head_off;
            index |= state << state_off;
            index
        };
        let decode = |mut index: usize| -> (Vec<Symbol>, Vec<usize>, usize, usize) {
            let tape = (0..work)
                .map(|cell| ((index >> (cell * symbol_bits)) & (self.alphabet - 1)) as Symbol)
                .collect();
            index >>= work_bits;
            let anc = (0..work)
                .map(|slot| (index >> (slot * ancilla_bits)) & ancilla_mask)
                .collect();
            index >>= ancilla_region_bits;
            let head = index & ((1 << head_bits) - 1);
            index >>= head_bits;
            let state = index & (self.states - 1);
            (tape, anc, head, state)
        };

        // Pass 1: every CLEAN-ancilla config fires its transition into a target.
        // Collect src -> tgt; reject two sources sharing a target or a target
        // that lands back on the physical (clean) subspace.
        let mut is_source = vec![false; dimension];
        for (index, slot) in is_source.iter_mut().enumerate() {
            let (_, anc, _, _) = decode(index);
            if anc.iter().all(|&a| a == 0) {
                *slot = true;
            }
        }
        let mut forward: BTreeMap<usize, usize> = BTreeMap::new();
        for index in 0..dimension {
            if !is_source[index] {
                continue; // only clean-ancilla configs are physical sources
            }
            let (tape, mut anc, head, state) = decode(index);
            let read = tape[head] as usize;
            let target = match self.transitions[state][read] {
                Some(t) => {
                    // Save `read + 1` into the ancilla slot for this head so the
                    // slot is always non-zero (0 = empty); the erased read value
                    // is what makes an information-erasing write injective.
                    anc[head] = read + 1;
                    let mut next = tape.clone();
                    next[head] = t.write;
                    let next_head = match t.direction {
                        Direction::Left => (head + work - 1) % work,
                        Direction::Right => (head + 1) % work,
                        Direction::Stay => head,
                    };
                    encode(&next, &anc, next_head, t.next_state)
                }
                None => index, // no transition: fixed point
            };
            if target != index {
                if is_source[target] || forward.values().any(|&v| v == target) {
                    return Err(QtmError::NotReversible);
                }
                forward.insert(index, target);
            }
        }
        // Pass 2: compose the full-space permutation as transpositions — each
        // moved source swaps with its dirty target (forward step == its own
        // inverse, so applying twice uncomputes). Untouched indices are fixed.
        let mut map = BTreeMap::new();
        for (&src, &tgt) in &forward {
            map.insert(src, tgt);
            map.insert(tgt, src);
        }
        for index in 0..dimension {
            map.entry(index).or_insert(index);
        }
        Ok((map, total_qubits))
    }

    /// Run one quantum step on the **ancilla-augmented** register, applying the
    /// involution from [`Self::permutation_with_ancilla`] coherently to a state
    /// vector over the enlarged space. This is what lets an information-erasing
    /// transition table (rejected by [`Self::step_quantum`] on the plain
    /// register) run as a unitary: the erased read symbol is preserved in the
    /// ancilla, so the map is a bijection on the full space.
    ///
    /// `state` must be a `StateVector` over `2^total_qubits` amplitudes, where
    /// `total_qubits` is the value returned alongside the permutation. Build
    /// the initial state with [`Self::prepare_superposition_ancilla`].
    pub fn step_quantum_ancilla(
        &self,
        state: &StateVector,
        extra_cells: usize,
    ) -> Result<StateVector, QtmError> {
        let (permutation, total_qubits) = self.permutation_with_ancilla(extra_cells)?;
        let amplitudes = state.amplitudes().to_vec();
        if amplitudes.len() != 1usize << total_qubits {
            return Err(QtmError::NotPowerOfTwo(state.qubits()));
        }
        // Apply the permutation: new[target] = old[index].
        let mut next = vec![Complex64::ZERO; amplitudes.len()];
        for (&index, &target) in &permutation {
            next[target] = amplitudes[index];
        }
        StateVector::from_amplitudes(next).map_err(|_| QtmError::NotPowerOfTwo(0))
    }

    /// Prepare the uniform superposition over internal states for a given tape
    /// and head, on the **ancilla-augmented** register (ancilla starts empty).
    /// Feed the result to [`Self::step_quantum_ancilla`].
    pub fn prepare_superposition_ancilla(
        &self,
        tape: &[Symbol],
        head: usize,
        extra_cells: usize,
    ) -> Result<StateVector, QtmError> {
        let (_, total_qubits) = self.permutation_with_ancilla(extra_cells)?;
        let symbol_bits = self.alphabet.trailing_zeros() as usize;
        let work = self.tape_cells;
        let ancilla_bits = symbol_bits + 1;
        let work_bits = work * symbol_bits;
        let ancilla_region_bits = work * ancilla_bits;
        let ancilla_off = work_bits;
        let head_off = ancilla_off + ancilla_region_bits;
        let state_off = head_off + work.trailing_zeros() as usize;
        let dimension = 1usize << total_qubits;
        let mut amplitudes = vec![Complex64::ZERO; dimension];
        let amp = Complex64::new(1.0 / (self.states as f64).sqrt(), 0.0);
        for state in 0..self.states {
            let mut index = 0usize;
            for (cell, &sym) in tape.iter().enumerate() {
                index |= (sym as usize) << (cell * symbol_bits);
            }
            // ancilla region left zero (empty)
            index |= head << head_off;
            index |= state << state_off;
            amplitudes[index] = amp;
        }
        StateVector::from_amplitudes(amplitudes).map_err(|_| QtmError::NotPowerOfTwo(0))
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
        StateVector::from_amplitudes(next).map_err(|_| QtmError::NotPowerOfTwo(0))
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
        StateVector::from_amplitudes(amplitudes).map_err(|_| QtmError::NotPowerOfTwo(0))
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
