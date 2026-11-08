use quantumforge::Complex64;
use quantumforge::tensor_network::{DMRG, MPS};

/// TFIM Hamiltonian gates at critical point h=1.
/// even_h = odd_h = -ZZ (bond term), single-site = -X (transverse field)
fn tfim_gates() -> ([[Complex64; 4]; 4], [[Complex64; 4]; 4]) {
    // ZZ gate: -Z⊗Z
    let zz = [
        [
            Complex64::new(-1.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
        ],
        [
            Complex64::ZERO,
            Complex64::new(1.0, 0.0),
            Complex64::ZERO,
            Complex64::ZERO,
        ],
        [
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(1.0, 0.0),
            Complex64::ZERO,
        ],
        [
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::ZERO,
            Complex64::new(-1.0, 0.0),
        ],
    ];
    (zz, zz)
}

fn tfim_single_site() -> ([[Complex64; 2]; 2], [[Complex64; 2]; 2]) {
    // Single-site: -X
    let x = [
        [Complex64::ZERO, Complex64::new(-1.0, 0.0)],
        [Complex64::new(-1.0, 0.0), Complex64::ZERO],
    ];
    (x, x)
}

#[test]
fn critical_point_energy_convergence() {
    // At critical point h=1, energy per site → -4/π ≈ -1.2732
    let thermodynamic_limit = -4.0 / std::f64::consts::PI;
    for n in [6, 8, 10] {
        let mps = MPS::zero_state(n, 32);
        let mut dmrg = DMRG::new(mps, 32, 1e-10, 30);
        let (zz, _) = tfim_gates();
        let (sx, _) = tfim_single_site();
        dmrg.set_single_site_gates(vec![sx; 3], vec![sx; 3]);
        let energy = dmrg.run_mps_only(&zz, &zz).unwrap();
        let per_site = energy / n as f64;
        println!(
            "n={}: E={:.6}, E/site={:.6}, thermodynamic={:.6}",
            n, energy, per_site, thermodynamic_limit
        );
        // Energy should be above thermodynamic limit (finite-size correction)
        assert!(
            per_site > thermodynamic_limit - 0.5,
            "n={}: E/site too low",
            n
        );
        assert!(per_site < 0.0, "n={}: E/site should be negative", n);
    }
}

#[test]
fn bond_dimension_scaling_analysis() {
    // Scaling analysis: fit energy error vs χ to extract convergence exponent.
    // At critical point, error ~ χ^(-κ) where κ ≈ 4/3 for TFIM.
    let thermodynamic_limit = -4.0 / std::f64::consts::PI;
    for n in [6, 8, 10] {
        let mut energies: Vec<(usize, f64)> = Vec::new();
        for chi in [4, 8, 16, 32] {
            let mps = MPS::zero_state(n, chi);
            let mut dmrg = DMRG::new(mps, chi, 1e-12, 30);
            let (zz, _) = tfim_gates();
            let (sx, _) = tfim_single_site();
            let n_even = n.div_ceil(2);
            let n_odd = n / 2;
            dmrg.set_single_site_gates(vec![sx; n_even], vec![sx; n_odd]);
            let energy = dmrg.run_mps_only(&zz, &zz).unwrap();
            energies.push((chi, energy));
        }
        // Compute energy error per site
        let errors: Vec<(f64, f64)> = energies
            .iter()
            .map(|(chi, e)| (*chi as f64, (e / n as f64 - thermodynamic_limit).abs()))
            .collect();
        // Fit log(error) vs log(χ) to extract slope κ
        let n_pts = errors.len() as f64;
        let sum_x: f64 = errors.iter().map(|(x, _)| x.ln()).sum();
        let sum_y: f64 = errors.iter().map(|(_, y)| y.ln()).sum();
        let sum_xx: f64 = errors.iter().map(|(x, _)| x.ln().powi(2)).sum();
        let sum_xy: f64 = errors.iter().map(|(x, y)| x.ln() * y.ln()).sum();
        let kappa = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_xx - sum_x * sum_x);
        println!("n={}: χ scaling: {:?}, fitted κ={:.3}", n, errors, kappa);
        // κ documents convergence rate (may be ~0 for exact diag, positive for MPS)
        println!("n={}: fitted convergence exponent κ={:.3}", n, kappa);
        // Best energy should be close to thermodynamic limit
        let best = energies.last().unwrap().1 / n as f64;
        assert!(
            (best - thermodynamic_limit).abs() < 0.5,
            "n={}: best E/site={:.6}, thermodynamic={:.6}",
            n,
            best,
            thermodynamic_limit
        );
    }
}
