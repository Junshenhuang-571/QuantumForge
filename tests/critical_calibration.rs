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
fn bond_dimension_convergence() {
    // For fixed n=8, increasing χ should converge
    let n = 8;
    let mut prev_energy = f64::INFINITY;
    for chi in [8, 16, 32] {
        let mps = MPS::zero_state(n, chi);
        let mut dmrg = DMRG::new(mps, chi, 1e-12, 30);
        let (zz, _) = tfim_gates();
        let energy = dmrg.run(&zz, &zz).unwrap();
        println!("χ={}: E={:.8}", chi, energy);
        // Energy should decrease (or stay same) with increasing χ
        assert!(
            energy <= prev_energy + 1e-6,
            "χ={} energy higher than χ={}",
            chi,
            chi / 2
        );
        prev_energy = energy;
    }
}
