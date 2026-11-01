use quantumforge::{Circuit, Gate, StateVector};
use std::time::Instant;

fn main() {
    println!("Circuit, Qubits, Gates, Time (ms), Norm");
    
    // QFT 4q
    {
        let mut c = Circuit::new();
        for i in 0..4 {
            c.push(Gate::H, i);
            for j in (i + 1)..4 {
                let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                c.push(Gate::Rz(angle), j);
            }
        }
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(4).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!("QFT, 4, {}, {:.3}, {:.10}", gates, dt, norm);
    }
    
    // QFT 8q
    {
        let mut c = Circuit::new();
        for i in 0..8 {
            c.push(Gate::H, i);
            for j in (i + 1)..8 {
                let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                c.push(Gate::Rz(angle), j);
            }
        }
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(8).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!("QFT, 8, {}, {:.3}, {:.10}", gates, dt, norm);
    }
    
    // GHZ 4q
    {
        let mut c = Circuit::new();
        c.push(Gate::H, 0);
        for i in 0..3 { c.cnot(i, i + 1); }
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(4).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!("GHZ, 4, {}, {:.3}, {:.10}", gates, dt, norm);
    }
    
    // GHZ 8q
    {
        let mut c = Circuit::new();
        c.push(Gate::H, 0);
        for i in 0..7 { c.cnot(i, i + 1); }
        let gates = c.operations.len();
        let t0 = Instant::now();
        let sv = StateVector::zero(8).unwrap();
        let sv = c.run(sv).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        let norm: f64 = sv.amplitudes().iter().map(|a| a.norm_sqr()).sum();
        println!("GHZ, 8, {}, {:.3}, {:.10}", gates, dt, norm);
    }
}
