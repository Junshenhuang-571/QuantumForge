use quantumforge::Complex64;
use quantumforge::tensor_network::{DMRG, MPS};
use std::time::Instant;

fn zz_gate() -> [[Complex64; 4]; 4] {
    [
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
    ]
}

fn main() {
    let zz = zz_gate();
    println!("n,time_ms,energy,energy_per_site");
    for &n in &[6usize, 8, 10, 12] {
        let mps = MPS::zero_state(n, 32);
        let mut dmrg = DMRG::new(mps, 32, 1e-10, 20);
        let t0 = Instant::now();
        let energy = dmrg.run(&zz, &zz).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        println!("{},{:.2},{:.6},{:.6}", n, dt, energy, energy / n as f64);
    }
}
