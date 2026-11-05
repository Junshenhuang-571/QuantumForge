use quantumforge::{Circuit, Gate, StateVector};
use std::time::Instant;

fn main() {
    println!("Circuit, Source Corpus, Qubits, Gates, Time (ms), Norm");

    // QFT (found in MQT Bench, QKNOB, HamLib)
    for &n in &[4usize, 8] {
        let mut c = Circuit::new();
        for i in 0..n {
            c.push(Gate::H, i);
            for j in (i + 1)..n {
                let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                c.push(Gate::Rz(angle), j);
            }
        }
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(n).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!(
            "QFT, MQT/QKNOB/HamLib, {}, {}, {:.3}, {:.10}",
            n, gates, dt, norm
        );
    }

    // GHZ (found in MQT Bench, HamLib)
    for &n in &[4usize, 8] {
        let mut c = Circuit::new();
        c.push(Gate::H, 0);
        for i in 0..(n - 1) {
            c.cnot(i, i + 1);
        }
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(n).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!("GHZ, MQT/HamLib, {}, {}, {:.3}, {:.10}", n, gates, dt, norm);
    }

    // Grover 4q (found in MQT Bench)
    {
        let mut c = Circuit::new();
        for i in 0..4 {
            c.push(Gate::H, i);
        }
        c.push(Gate::Z, 3);
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(4).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!("Grover, MQT Bench, 4, {}, {:.3}, {:.10}", gates, dt, norm);
    }

    // BV 4q (found in MQT Bench, QKNOB)
    {
        let mut c = Circuit::new();
        for i in 0..4 {
            c.push(Gate::H, i);
        }
        // Oracle: f(x) = s.x (s=1010)
        c.push(Gate::Z, 0);
        c.push(Gate::Z, 2);
        for i in 0..4 {
            c.push(Gate::H, i);
        }
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(4).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!(
            "Bernstein-Vazirani, MQT/QKNOB, 4, {}, {:.3}, {:.10}",
            gates, dt, norm
        );
    }
}
