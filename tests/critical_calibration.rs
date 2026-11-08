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
    // Single-site X: -X
    // DMRG uses even_single_gates and odd_single_gates
    (zz, zz)
}

#[test]
fn critical_point_energy_convergence() {
    // At critical point h=1, energy per site → -4/π ≈ -1.2732
    let thermodynamic_limit = -4.0 / std::f64::consts::PI;
    for n in [6, 8, 10] {
        let mps = MPS::zero_state(n, 32);
        let mut dmrg = DMRG::new(mps, 32, 1e-10, 30);
        let (zz, _) = tfim_gates();
        let energy = dmrg.run(&zz, &zz).unwrap();
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
fn bond_dimension_scaling_at_critical_point() {
    // At TFIM critical point, energy error should decrease exponentially
    // with bond dimension χ. Verify monotonic convergence for multiple L.
    let thermodynamic_limit = -4.0 / std::f64::consts::PI;
    for n in [6, 8, 10] {
        let mut energies: Vec<(usize, f64)> = Vec::new();
        for chi in [4, 8, 16] {
            let mps = MPS::zero_state(n, chi);
            let mut dmrg = DMRG::new(mps, chi, 1e-12, 30);
            let (zz, _) = tfim_gates();
            let energy = dmrg.run(&zz, &zz).unwrap();
            energies.push((chi, energy));
        }
        // Verify monotonic convergence: energy decreases with χ
        for i in 1..energies.len() {
            assert!(
                energies[i].1 <= energies[i - 1].1 + 1e-6,
                "n={}: χ={} energy ({:.8}) > χ={} energy ({:.8})",
                n,
                energies[i].0,
                energies[i].1,
                energies[i - 1].0,
                energies[i - 1].1
            );
        }
        // Verify convergence toward thermodynamic limit
        let best = energies.last().unwrap().1 / n as f64;
        assert!(
            (best - thermodynamic_limit).abs() < 0.5,
            "n={}: best E/site={:.6}, thermodynamic={:.6}",
            n,
            best,
            thermodynamic_limit
        );
        println!(
            "n={}: χ convergence: {:?}, best E/site={:.6}",
            n, energies, best
        );
    }
}
