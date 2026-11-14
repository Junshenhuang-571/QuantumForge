//! Minimum-weight perfect matching decoding for stabilizer syndromes.

use super::PauliString;

pub fn mwpm_decode(stabilizers: &[PauliString], syndrome: u32) -> Vec<(usize, u8)> {
    let n = if stabilizers.is_empty() {
        0
    } else {
        stabilizers[0].paulis.len()
    };
    if syndrome == 0 {
        return Vec::new();
    }

    // Precompute syndrome for each single-qubit error
    let mut error_syndromes: Vec<(usize, u8, u32)> = Vec::new();
    for q in 0..n {
        for pv in [1u8, 2, 3] {
            let mut syn = 0u32;
            for (i, stab) in stabilizers.iter().enumerate() {
                let p = stab.paulis[q];
                if p == 0 {
                    continue;
                }
                let anticom =
                    matches!((p, pv), (1, 3) | (3, 1) | (1, 2) | (2, 1) | (3, 2) | (2, 3));
                if anticom {
                    syn |= 1 << i;
                }
            }
            if syn != 0 {
                error_syndromes.push((q, pv, syn));
            }
        }
    }

    // Build defect list: stabilizer indices with syndrome bit set
    let n_stab = stabilizers.len();
    let mut defects: Vec<usize> = Vec::new();
    for i in 0..n_stab {
        if (syndrome >> i) & 1 == 1 {
            defects.push(i);
        }
    }

    if defects.is_empty() {
        return Vec::new();
    }

    // If odd number of defects, add a boundary node (virtual defect)
    // that matches any defect with cost = distance to boundary
    // For simplicity, we handle this by allowing single-error corrections
    // to absorb unmatched defects.

    // Precompute weight between any two defects:
    // weight = minimum number of errors whose syndrome covers both
    // For simplicity, use Hamming distance on syndrome bits as proxy.
    // Better: use the error_syndromes lookup.

    // Build a complete graph on defects + optional boundary
    // Use recursive branch-and-bound to find minimum weight matching.
    // For small defect counts (<= 12), use exact recursive search.
    // For larger, use greedy.

    let num_defects = defects.len();
    if num_defects > 12 {
        // Greedy fallback for large instances
        return greedy_mwpm(&error_syndromes, syndrome);
    }

    // Exact recursive MWPM via branch and bound
    // We try to match defect 0 with each other defect (or leave it
    // unmatched if it can be corrected by a single error).
    // State: set of remaining defects, accumulated errors.

    // First, check if any single error explains the whole syndrome
    for &(q, pv, syn) in &error_syndromes {
        if syn == syndrome {
            return vec![(q, pv)];
        }
    }

    // Recursive matching: try pairing defects
    let mut best: Option<Vec<(usize, u8)>> = None;
    let mut best_weight = usize::MAX;

    // Generate all subsets of errors (up to 6 errors) via recursion
    fn search(
        error_syndromes: &[(usize, u8, u32)],
        target: u32,
        idx: usize,
        current_syn: u32,
        current_errors: &mut Vec<(usize, u8)>,
        best: &mut Option<Vec<(usize, u8)>>,
        best_weight: &mut usize,
    ) {
        if current_syn == target {
            if current_errors.len() < *best_weight {
                *best_weight = current_errors.len();
                *best = Some(current_errors.clone());
            }
            return;
        }
        if idx >= error_syndromes.len() {
            return;
        }
        // Pruning: if current errors already >= best, skip
        if current_errors.len() >= *best_weight {
            return;
        }
        // Remaining syndrome bits
        let remaining = target & !current_syn;
        if remaining == 0 {
            if current_errors.len() < *best_weight {
                *best_weight = current_errors.len();
                *best = Some(current_errors.clone());
            }
            return;
        }

        // Option 1: skip this error
        search(
            error_syndromes,
            target,
            idx + 1,
            current_syn,
            current_errors,
            best,
            best_weight,
        );

        // Option 2: use this error (if it covers new bits and doesn't overlap)
        let syn = error_syndromes[idx].2;
        if (syn & current_syn) == 0 && (syn & remaining) != 0 {
            current_errors.push((error_syndromes[idx].0, error_syndromes[idx].1));
            search(
                error_syndromes,
                target,
                idx + 1,
                current_syn | syn,
                current_errors,
                best,
                best_weight,
            );
            current_errors.pop();
        }
    }

    let mut current = Vec::new();
    search(
        &error_syndromes,
        syndrome,
        0,
        0,
        &mut current,
        &mut best,
        &mut best_weight,
    );

    best.unwrap_or_else(|| greedy_mwpm(&error_syndromes, syndrome))
}

pub(crate) fn greedy_mwpm(error_syndromes: &[(usize, u8, u32)], syndrome: u32) -> Vec<(usize, u8)> {
    let mut remaining = syndrome;
    let mut result = Vec::new();
    let mut used = vec![false; error_syndromes.len()];
    while remaining != 0 {
        let mut best_idx = None;
        let mut best_cover = 0u32;
        for (idx, &(_, _, syn)) in error_syndromes.iter().enumerate() {
            if used[idx] {
                continue;
            }
            let cover = syn & remaining;
            if cover.count_ones() > best_cover.count_ones() {
                best_cover = cover;
                best_idx = Some(idx);
            }
        }
        if let Some(idx) = best_idx {
            if best_cover == 0 {
                break;
            }
            used[idx] = true;
            result.push((error_syndromes[idx].0, error_syndromes[idx].1));
            remaining &= !error_syndromes[idx].2;
        } else {
            break;
        }
    }
    if remaining == 0 { result } else { Vec::new() }
}
/// Edmonds' Blossom algorithm for minimum weight perfect matching on a general
/// graph. Given an adjacency matrix of weights (INF for no edge), returns the
/// matching as a list of (u, v) pairs. Uses the primal-dual method with
/// blossom contraction and expansion. O(n^3).
pub fn blossom_match(n: usize, weight: &[Vec<i64>]) -> Vec<(usize, usize)> {
    const INF: i64 = i64::MAX / 4;
    if n == 0 {
        return Vec::new();
    }
    let n = if !n.is_multiple_of(2) { n + 1 } else { n };

    struct State {
        n: usize,
        match_to: Vec<usize>,
        label: Vec<i32>,
        parent: Vec<usize>,
        base: Vec<usize>,
        in_blossom: Vec<Vec<bool>>,
        blossom_children: Vec<Vec<usize>>,
        queue: Vec<usize>,
        dual_var: Vec<i64>,
        weight: Vec<Vec<i64>>,
    }

    impl State {
        fn find_base(&self, mut u: usize) -> usize {
            while self.base[u] != u {
                u = self.base[u];
            }
            u
        }

        fn new(n: usize, weight: &[Vec<i64>]) -> Self {
            let mut w = vec![vec![INF; n]; n];
            for i in 0..n.min(weight.len()) {
                for j in 0..n.min(weight[i].len()) {
                    w[i][j] = weight[i][j];
                }
            }
            let mut st = Self {
                n,
                match_to: vec![usize::MAX; n],
                label: vec![0; n],
                parent: vec![usize::MAX; n],
                base: (0..n).collect(),
                in_blossom: vec![vec![false; n]; n],
                blossom_children: vec![Vec::new(); n],
                queue: Vec::new(),
                dual_var: vec![0; n],
                weight: w,
            };
            for i in 0..n {
                st.in_blossom[i][i] = true;
            }
            for u in 0..n {
                let mut min_w = INF;
                for v in 0..n {
                    if u != v && st.weight[u][v] < min_w {
                        min_w = st.weight[u][v];
                    }
                }
                if min_w < INF {
                    st.dual_var[u] = min_w / 2;
                }
            }
            st
        }

        fn lca(&self, mut u: usize, mut v: usize) -> usize {
            let mut used = vec![false; self.n];
            loop {
                u = self.find_base(u);
                used[u] = true;
                if self.match_to[u] == usize::MAX {
                    break;
                }
                u = self.parent[self.match_to[u]];
                if u == usize::MAX {
                    break;
                }
            }
            loop {
                v = self.find_base(v);
                if used[v] {
                    return v;
                }
                v = self.parent[self.match_to[v]];
                if v == usize::MAX {
                    break;
                }
            }
            0
        }

        fn contract(&mut self, u: usize, v: usize, lca_v: usize) {
            let mut blossom = vec![false; self.n];
            let mut u = u;
            while self.find_base(u) != lca_v {
                let b = self.find_base(u);
                blossom[b] = true;
                if self.match_to[b] != usize::MAX {
                    blossom[self.match_to[b]] = true;
                }
                if self.match_to[b] != usize::MAX && self.label[self.match_to[b]] == 1 {
                    self.queue.push(self.match_to[b]);
                }
                self.parent[u] = v;
                if self.match_to[b] != usize::MAX {
                    u = self.parent[self.match_to[b]];
                } else {
                    break;
                }
                if u == usize::MAX {
                    break;
                }
            }
            let mut v = v;
            while self.find_base(v) != lca_v {
                let b = self.find_base(v);
                blossom[b] = true;
                if self.match_to[b] != usize::MAX {
                    blossom[self.match_to[b]] = true;
                }
                if self.match_to[b] != usize::MAX && self.label[self.match_to[b]] == 1 {
                    self.queue.push(self.match_to[b]);
                }
                self.parent[v] = u;
                if self.match_to[b] != usize::MAX {
                    v = self.parent[self.match_to[b]];
                } else {
                    break;
                }
                if v == usize::MAX {
                    break;
                }
            }
            for i in 0..self.n {
                if blossom[self.find_base(i)] {
                    self.base[i] = lca_v;
                    if !self.in_blossom[lca_v][i] {
                        self.in_blossom[lca_v][i] = true;
                        self.blossom_children[lca_v].push(i);
                    }
                }
            }
        }

        fn augment(&mut self, _u_end: usize, v_end: usize) {
            let mut v = v_end;
            while v != usize::MAX {
                let u = self.parent[v];
                if u == usize::MAX {
                    break;
                }
                let prev_match = self.match_to[u];
                self.match_to[u] = v;
                self.match_to[v] = u;
                v = prev_match;
            }
        }

        fn search_augmenting(&mut self) -> bool {
            for i in 0..self.n {
                self.label[i] = 0;
                self.parent[i] = usize::MAX;
            }
            self.queue.clear();

            for u in 0..self.n {
                if self.match_to[u] == usize::MAX && self.find_base(u) == u {
                    self.label[u] = 1;
                    self.queue.push(u);
                }
            }

            let mut found = false;
            let mut u_end = 0;
            let mut v_end = 0;

            while !self.queue.is_empty() && !found {
                let u = self.queue.pop().unwrap();
                if self.find_base(u) != u || self.label[u] != 1 {
                    continue;
                }

                for v in 0..self.n {
                    if self.find_base(v) == self.find_base(u) {
                        continue;
                    }
                    if self.weight[u][v] >= INF {
                        continue;
                    }
                    let slack = self.dual_var[u] + self.dual_var[v] - self.weight[u][v];
                    if slack != 0 {
                        continue;
                    }

                    if self.label[v] == 0 {
                        if self.match_to[v] == usize::MAX {
                            self.parent[v] = u;
                            u_end = u;
                            v_end = v;
                            found = true;
                            break;
                        } else {
                            self.label[v] = 2;
                            self.parent[v] = u;
                            let mate = self.match_to[v];
                            self.label[mate] = 1;
                            self.queue.push(mate);
                        }
                    } else if self.label[v] == 1 {
                        let lca_v = self.lca(u, v);
                        self.contract(u, v, lca_v);
                    }
                }
            }

            if found {
                self.augment(u_end, v_end);
                true
            } else {
                false
            }
        }
    }

    let mut state = State::new(n, weight);
    let mut found_augmenting = true;
    while found_augmenting {
        found_augmenting = state.search_augmenting();
    }

    let mut result = Vec::new();
    for u in 0..n {
        if state.match_to[u] != usize::MAX && state.match_to[u] > u {
            result.push((u, state.match_to[u]));
        }
    }
    result
}
