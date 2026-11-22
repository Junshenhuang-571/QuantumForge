//! Stabilizer tableau simulator (Gottesman-Knill).
//!
//! Simulates Clifford circuits (H, S, X, Y, Z, CNOT, measurement) in
//! O(n²) time per gate, avoiding the exponential 2^n cost of state vectors.
//! This is the standard Aaronson-Gottesman algorithm.

use crate::SmallRng;

/// A stabilizer tableau for n qubits.
/// Stores n stabilizer rows and n destabilizer rows, each a Pauli string
/// with a phase (±1 or ±i).
pub struct StabilizerTableau {
    n: usize,
    // Each row: [X bits (n), Z bits (n), phase]
    // Stabilizers: rows 0..n
    // Destabilizers: rows n..2n
    x: Vec<Vec<bool>>,
    z: Vec<Vec<bool>>,
    phase: Vec<u8>, // 0=+1, 1=+i, 2=-1, 3=-i
}

impl StabilizerTableau {
    /// Number of qubits the tableau acts on.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Create a tableau for `n` qubits with stabilizer `i` equal to `Z_i` and destabilizer `i` equal to `X_i`.
    pub fn new(n: usize) -> Self {
        let mut x = vec![vec![false; n]; 2 * n];
        let mut z = vec![vec![false; n]; 2 * n];
        let phase = vec![0u8; 2 * n];
        // Initialize: stabilizer i = Z_i, destabilizer i = X_i
        for i in 0..n {
            z[i][i] = true; // stabilizer Z_i
            x[n + i][i] = true; // destabilizer X_i
        }
        Self { n, x, z, phase }
    }

    /// Apply Hadamard on qubit q.
    pub fn h(&mut self, q: usize) {
        for i in 0..2 * self.n {
            // Swap X and Z bits
            std::mem::swap(&mut self.x[i][q], &mut self.z[i][q]);
            // H Y H = -Y, so a row that is Y on this qubit picks up a minus.
            // (After the swap, both bits set means both were set before it.)
            if self.x[i][q] && self.z[i][q] {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
        }
    }

    /// Apply S (phase gate) on qubit q.
    pub fn s(&mut self, q: usize) {
        for i in 0..2 * self.n {
            if self.x[i][q] {
                // S X S† = Y and S Y S† = -X; both pick up a factor of i.
                self.z[i][q] ^= true;
                self.phase[i] = (self.phase[i] + 1) % 4;
            }
        }
    }

    /// Apply CNOT with control c, target t.
    ///
    /// Conjugation is `X_c -> X_c X_t`, `X_t -> X_t`, `Z_c -> Z_c`,
    /// `Z_t -> Z_c Z_t`, which in the `(x|z)` representation flips `x_t` by
    /// `x_c` and `z_c` by `z_t`. With the `i^phase X^x Z^z` convention this map
    /// carries no phase correction.
    pub fn cnot(&mut self, c: usize, t: usize) {
        for i in 0..2 * self.n {
            if self.x[i][c] {
                self.x[i][t] ^= true;
            }
            if self.z[i][t] {
                self.z[i][c] ^= true;
            }
        }
    }

    /// Apply X on qubit q.
    pub fn x(&mut self, q: usize) {
        for i in 0..2 * self.n {
            if self.z[i][q] {
                self.phase[i] = (self.phase[i] + 2) % 4; // multiply by -1
            }
        }
    }

    /// Apply Z on qubit q.
    pub fn z(&mut self, q: usize) {
        for i in 0..2 * self.n {
            if self.x[i][q] {
                self.phase[i] = (self.phase[i] + 2) % 4;
            }
        }
    }

    /// Measure qubit q in Z basis. Returns `(outcome, probability)`.
    ///
    /// For Clifford states the probability is 0, 1, or 1/2. When the outcome is
    /// random this returns a fixed `false`; use [`Self::measure_seeded`] for a
    /// reproducible draw or [`Self::measure_with_rng`] to supply a generator.
    pub fn measure(&mut self, q: usize) -> (bool, f64) {
        self.measure_inner(q, false)
    }

    /// Measure qubit `q`, drawing a random outcome when the stabilizer group
    /// does not determine one.
    ///
    /// The draw happens only when it matters, so a deterministic qubit does not
    /// consume generator state.
    pub fn measure_with_rng(&mut self, q: usize, rng: &mut SmallRng) -> (bool, f64) {
        self.measure_inner(q, rng.next_f64() < 0.5)
    }

    /// Measure qubit `q` with an outcome drawn from `seed`.
    ///
    /// Deterministic for a given `(state, q, seed)` triple, so a run can be
    /// reproduced exactly.
    pub fn measure_seeded(&mut self, q: usize, seed: u64) -> (bool, f64) {
        let mut rng = SmallRng::seeded(seed);
        self.measure_with_rng(q, &mut rng)
    }

    /// Shared measurement body. `random_outcome` is the result to report on the
    /// random branch; on the deterministic branch it is ignored.
    fn measure_inner(&mut self, q: usize, random_outcome: bool) -> (bool, f64) {
        // Find a *stabilizer* row r (rows 0..n) with X_r[q] = 1, meaning it
        // anticommutes with Z_q. Only stabilizers decide whether the outcome is
        // random; a destabilizer anticommuting is expected and says nothing.
        // Searching all 2n rows instead would pick a destabilizer and then
        // index r + n out of bounds.
        let mut r = None;
        for (i, row) in self.x.iter().take(self.n).enumerate() {
            if row[q] {
                r = Some(i);
                break;
            }
        }

        if let Some(r) = r {
            let outcome = random_outcome;
            // Update tableau: new stabilizer is ±Z_q (depending on outcome)
            // Destabilizer becomes old stabilizer r
            for i in 0..2 * self.n {
                if i != r && i != r + self.n && self.x[i][q] {
                    // Multiply row i by row r
                    self.row_mul(i, r);
                }
            }
            // The destabilizer r+n must receive the OLD stabilizer r, so save
            // it before row r is replaced by ±Z_q.
            for j in 0..self.n {
                self.x[r + self.n][j] = self.x[r][j];
                self.z[r + self.n][j] = self.z[r][j];
            }
            self.phase[r + self.n] = self.phase[r];
            // Set stabilizer r to ±Z_q
            for j in 0..self.n {
                self.x[r][j] = false;
                self.z[r][j] = j == q;
            }
            self.phase[r] = if outcome { 2 } else { 0 };
            (outcome, 0.5)
        } else {
            // Deterministic. `Z_q` commutes with every stabilizer, so by
            // maximality it is in the stabilizer group up to sign. Find the
            // combination by projecting onto the destabilizer dual basis:
            // if `v = (0 | e_q)` then the coefficient of stabilizer `j` in
            // `v = sum c_j s_j` is `c_j = <v, d_j> = x[n+j][q]`. Summing the
            // phases of whichever rows have `Z` on `q` is *not* equivalent and
            // gives the wrong answer whenever those rows carry X support.
            let mut product_x = vec![false; self.n];
            let mut product_z = vec![false; self.n];
            let mut product_phase = 0u8;
            for j in 0..self.n {
                if !self.x[self.n + j][q] {
                    continue;
                }
                // product *= stabilizer row j, with the same phase rule as
                // `row_mul`.
                let comm = (0..self.n)
                    .filter(|&k| (product_x[k] && self.z[j][k]) || (product_z[k] && self.x[j][k]))
                    .count() as u8;
                let mut phase = product_phase + self.phase[j];
                if comm % 2 == 1 {
                    phase += 2;
                }
                for k in 0..self.n {
                    product_x[k] ^= self.x[j][k];
                    product_z[k] ^= self.z[j][k];
                }
                product_phase = phase % 4;
            }
            let expected_z: Vec<bool> = (0..self.n).map(|k| k == q).collect();
            debug_assert_eq!(product_x, vec![false; self.n], "product should carry no X");
            debug_assert_eq!(product_z, expected_z, "product should be Z_q");
            // The product is `i^phase Z_q`: phase 0 is +Z_q, phase 2 is -Z_q.
            let outcome = product_phase == 2 || product_phase == 3;
            (outcome, 1.0)
        }
    }

    fn row_mul(&mut self, dst: usize, src: usize) {
        for j in 0..self.n {
            self.x[dst][j] ^= self.x[src][j];
            self.z[dst][j] ^= self.z[src][j];
        }
        // Phase: multiply phases, accounting for commutation
        let mut phase = self.phase[dst] + self.phase[src];
        // Commutation factor: (-1)^{<x_dst, z_src> + <z_dst, x_src>}
        let mut comm = 0i32;
        for j in 0..self.n {
            if self.x[dst][j] && self.z[src][j] {
                comm += 1;
            }
            if self.z[dst][j] && self.x[src][j] {
                comm += 1;
            }
        }
        if comm % 2 == 1 {
            phase = (phase + 2) % 4;
        }
        self.phase[dst] = phase % 4;
    }

    /// Check if two tableaux are equivalent (same stabilizer group).
    ///
    /// Two stabilizer states are the same state exactly when their stabilizer
    /// groups coincide. The groups are compared by reducing each stabilizer
    /// block to reduced row echelon form over GF(2) — tracking phases through
    /// the row multiplications — and comparing the canonical rows. RREF is
    /// unique for a row space, and each canonical row's phase is determined by
    /// the group element it represents, so equal groups give equal output.
    pub fn equivalent(&self, other: &Self) -> bool {
        if self.n != other.n {
            return false;
        }
        self.canonical_stabilizers() == other.canonical_stabilizers()
    }

    /// The stabilizer rows in reduced row echelon form, with phases.
    ///
    /// Column order is `(x_0..x_{n-1}, z_0..z_{n-1})`. Row multiplication
    /// uses the same phase rule as [`Self::row_mul`].
    fn canonical_stabilizers(&self) -> Vec<(Vec<bool>, Vec<bool>, u8)> {
        let n = self.n;
        let mut rows: Vec<(Vec<bool>, Vec<bool>, u8)> = (0..n)
            .map(|i| (self.x[i].clone(), self.z[i].clone(), self.phase[i]))
            .collect();

        let bit = |row: &(Vec<bool>, Vec<bool>, u8), col: usize| -> bool {
            if col < n { row.0[col] } else { row.1[col - n] }
        };

        let mut pivot = 0;
        for col in 0..2 * n {
            if pivot >= n {
                break;
            }
            let found = (pivot..n).find(|&r| bit(&rows[r], col));
            let Some(r) = found else {
                continue;
            };
            rows.swap(pivot, r);
            for other in 0..n {
                if other == pivot || !bit(&rows[other], col) {
                    continue;
                }
                let (px, pz, pphase) = rows[pivot].clone();
                let (sx, sz, sphase) = rows[other].clone();
                let comm = (0..n)
                    .filter(|&j| (sx[j] && pz[j]) || (sz[j] && px[j]))
                    .count() as u8;
                let mut phase = sphase + pphase;
                if comm % 2 == 1 {
                    phase += 2;
                }
                for j in 0..n {
                    rows[other].0[j] = sx[j] ^ px[j];
                    rows[other].1[j] = sz[j] ^ pz[j];
                }
                rows[other].2 = phase % 4;
            }
            pivot += 1;
        }
        rows
    }
}
