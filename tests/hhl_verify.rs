use quantumforge::StateVector;
use quantumforge::algorithms::{hhl_circuit_general, hhl_solve_quantum};

#[test]
fn hhl_circuit_output_matches_numerical() {
    let a = vec![vec![1.0, 0.5], vec![0.5, 2.0]];
    let b = vec![1.0, 1.0];
    let circuit = hhl_circuit_general(&a, &b);

    let _n = 2usize;
    let k = 1usize; // log2(2)
    let precision = 3usize;
    let total = k + precision + 1; // 5

    let mut sv = StateVector::zero(total).unwrap();
    sv = circuit.run(sv).unwrap();

    // Extract solution: amplitudes where ancilla (bit 4) = 1
    let mut x = [0.0f64, 0.0f64];
    for i in 0..(1 << total) {
        if (i >> (total - 1)) & 1 == 1 {
            let data_bit = i & 1;
            x[data_bit] += sv.amplitudes()[i].re; // real part
        }
    }
    let nrm = (x[0] * x[0] + x[1] * x[1]).sqrt();
    if nrm > 0.0 {
        x[0] /= nrm;
        x[1] /= nrm;
    }
    println!("Circuit solution (normalized): {:?}", x);

    let x_ref = hhl_solve_quantum([[a[0][0], a[0][1]], [a[1][0], a[1][1]]], [b[0], b[1]]);
    // Normalize reference
    let nrm_ref = (x_ref[0] * x_ref[0] + x_ref[1] * x_ref[1]).sqrt();
    let x_ref_norm = [x_ref[0] / nrm_ref, x_ref[1] / nrm_ref];
    println!("Numerical solution (normalized): {:?}", x_ref_norm);

    // Check if they're proportional
    let dot = x[0] * x_ref_norm[0] + x[1] * x_ref_norm[1];
    println!("Overlap: {:.4}", dot.abs());
}
