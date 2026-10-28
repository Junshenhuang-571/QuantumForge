#![cfg(feature = "cuda")]
use quantumforge::cuda::CudaBackend;
use quantumforge::{Circuit, Gate, StateVector};
use std::time::Instant;

fn main() {
    let n_qubits = 20;
    let batch_size = 32;
    let depth = 20;

    // Build a deep circuit
    let mut circuit = Circuit::new();
    for _ in 0..depth {
        for q in 0..n_qubits {
            circuit.push(Gate::Ry(0.5), q);
        }
        for q in 0..n_qubits - 1 {
            circuit.cnot(q, q + 1);
        }
    }

    // Generate batch of initial states (all |0>)
    let states: Vec<StateVector> = (0..batch_size)
        .map(|_| StateVector::zero(n_qubits).unwrap())
        .collect();

    // CPU: sequential
    let start = Instant::now();
    let mut cpu_results = Vec::with_capacity(batch_size);
    for s in &states {
        let mut sv = s.clone();
        for op in circuit.operations() {
            match *op {
                quantumforge::CircuitOp::SingleQubit { gate, target } => {
                    sv.apply(gate, target).unwrap();
                }
                quantumforge::CircuitOp::Cnot { control, target } => {
                    sv.apply_cnot(control, target).unwrap();
                }
                _ => {}
            }
        }
        cpu_results.push(sv);
    }
    let cpu_time = start.elapsed();

    // GPU: batched
    let backend = CudaBackend::new(0).expect("cuda");
    let start = Instant::now();
    let gpu_results = backend
        .run_circuit_batch(&states, &circuit)
        .expect("gpu batch");
    let gpu_time = start.elapsed();

    // Verify correctness
    let mut max_err = 0.0f64;
    for (c, g) in cpu_results.iter().zip(gpu_results.iter()) {
        for (a, b) in c.amplitudes().iter().zip(g.amplitudes().iter()) {
            max_err = max_err.max((*a - *b).abs());
        }
    }

    println!(
        "Batch circuit benchmark: {} qubits, depth {}, batch size {}",
        n_qubits, depth, batch_size
    );
    println!("CPU (sequential): {:?}", cpu_time);
    println!("GPU (batched):    {:?}", gpu_time);
    println!(
        "Speedup: {:.1}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );
    println!("Max error: {:.3e}", max_err);
}
