//! GF(2) linear algebra: row reduction, nullspace, membership, and
//! logical-operator extraction for CSS stabilizer codes.
//!
//! Row vectors are bitmasks over `n` qubits; requires `n <= 128`.

fn gf2_rref(rows: &mut [u128], n: usize) -> Vec<usize> {
    let mut pivots = Vec::new();
    let mut r = 0usize;
    for c in 0..n {
        let mut piv = None;
        for (i, row) in rows.iter().enumerate() {
            if i >= r && (row >> c) & 1 == 1 {
                piv = Some(i);
                break;
            }
        }
        if let Some(p) = piv {
            rows.swap(r, p);
            for i in 0..rows.len() {
                if i != r && (rows[i] >> c) & 1 == 1 {
                    rows[i] ^= rows[r];
                }
            }
            pivots.push(c);
            r += 1;
        }
    }
    pivots
}

fn gf2_in_span(v: u128, basis: &[u128], n: usize) -> bool {
    let mut b = basis.to_vec();
    let _ = gf2_rref(&mut b, n);
    let mut t = v;
    for row in &b {
        let c = (0..n).find(|&c| (row >> c) & 1 == 1).unwrap_or(n);
        if c < n && (t >> c) & 1 == 1 {
            t ^= row;
        }
    }
    t == 0
}

/// Basis of {v : M v = 0} for binary matrix M given by rows.
fn gf2_nullspace(rows: &[u128], n: usize) -> Vec<u128> {
    let mut m = rows.to_vec();
    let pivots = gf2_rref(&mut m, n);
    let pivot_set: std::collections::HashSet<usize> = pivots.into_iter().collect();
    let mut basis = Vec::new();
    for free_col in (0..n).filter(|c| !pivot_set.contains(c)) {
        let mut v = 1u128 << free_col;
        for row in &m {
            let c = (0..n).find(|&c| (row >> c) & 1 == 1).unwrap_or(n);
            if c < n && c != free_col && (row >> free_col) & 1 == 1 {
                v |= 1 << c;
            }
        }
        basis.push(v);
    }
    basis
}

/// Extract paired logical operators (X_logicals, Z_logicals) as qubit-index
/// vectors for a CSS code given by its X-stabilizer and Z-stabilizer supports.
/// Requires n <= 128.
pub(crate) fn extract_logicals(
    n: usize,
    xrows: &[Vec<usize>],
    zrows: &[Vec<usize>],
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let to_mask = |s: &[usize]| -> u128 {
        let mut p = 0u128;
        for &q in s {
            p |= 1u128 << q;
        }
        p
    };
    let x_masks: Vec<u128> = xrows.iter().map(|s| to_mask(s)).collect();
    let z_masks: Vec<u128> = zrows.iter().map(|s| to_mask(s)).collect();
    // logical Z candidates: z-vectors orthogonal to all X-rows
    let zl = gf2_nullspace(&x_masks, n);
    // logical X candidates: x-vectors orthogonal to all Z-rows
    let xl = gf2_nullspace(&z_masks, n);
    // reduce out the stabilizer spans
    let zl_pure: Vec<u128> = zl
        .into_iter()
        .filter(|v| !gf2_in_span(*v, &z_masks, n))
        .collect();
    let xl_pure: Vec<u128> = xl
        .into_iter()
        .filter(|v| !gf2_in_span(*v, &x_masks, n))
        .collect();
    // symplectic Gram-Schmidt into standard pairing form
    let mut z_vecs = zl_pure;
    let mut x_vecs = xl_pure;
    let mut pairs: Vec<(u128, u128)> = Vec::new();
    while let Some(pos) =
        (0..z_vecs.len()).find(|&i| x_vecs.iter().any(|x| (z_vecs[i] & x).count_ones() % 2 == 1))
    {
        let z0 = z_vecs[pos];
        let x0 = *x_vecs
            .iter()
            .find(|x| (z0 & *x).count_ones() % 2 == 1)
            .unwrap();
        // orthogonalize the remaining z's against x0, and remaining x's against z0
        z_vecs.retain(|z| *z != z0);
        x_vecs.retain(|x| *x != x0);
        z_vecs = z_vecs
            .into_iter()
            .map(|z| {
                if (z & x0).count_ones() % 2 == 1 {
                    z ^ z0
                } else {
                    z
                }
            })
            .collect();
        x_vecs = x_vecs
            .into_iter()
            .map(|x| {
                if (x & z0).count_ones() % 2 == 1 {
                    x ^ x0
                } else {
                    x
                }
            })
            .collect();
        pairs.push((z0, x0));
    }
    let _ = &mut x_vecs;
    pairs
        .into_iter()
        .map(|(z, x)| {
            let zs: Vec<usize> = (0..n).filter(|&q| (z >> q) & 1 == 1).collect();
            let xs: Vec<usize> = (0..n).filter(|&q| (x >> q) & 1 == 1).collect();
            (xs, zs)
        })
        .collect()
}
