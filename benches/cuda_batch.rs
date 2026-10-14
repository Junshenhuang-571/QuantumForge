use quantumforge::{Circuit, Gate, StateVector};
use std::time::{Duration, Instant};

const QUBITS: usize = 8;
const BATCH_SIZE: usize = 1024;
const ITERATIONS: usize = 5;

fn workload() -> (Vec<StateVector>, Circuit) {
    let mut states = Vec::with_capacity(BATCH_SIZE);
    for basis in 0..BATCH_SIZE {
        let mut state = StateVector::zero(QUBITS).expect("valid qubit count");
        for qubit in 0..QUBITS {
            if (basis >> (qubit % 4)) & 1 != 0 {
                state.apply(Gate::X, qubit).expect("valid qubit index");
            }
        }
        states.push(state);
    }

    let mut circuit = Circuit::new();
    for layer in 0..20 {
        for qubit in 0..QUBITS {
            circuit.push(Gate::Ry(0.013 * (layer + qubit + 1) as f64), qubit);
        }
        for qubit in 0..QUBITS - 1 {
            circuit.cnot(qubit, qubit + 1);
        }
        circuit.push(Gate::Rz(0.017 * (layer + 1) as f64), layer % QUBITS);
    }
    (states, circuit)
}

fn elapsed_average(mut run: impl FnMut() -> Result<(), Box<dyn std::error::Error>>) -> Duration {
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        run().expect("benchmark workload should succeed");
    }
    started.elapsed() / ITERATIONS as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (states, circuit) = workload();
    let cpu_reference: Vec<_> = states
        .iter()
        .map(|state| circuit.run(state.clone()))
        .collect::<Result<_, _>>()?;
    let backend = quantumforge::cuda::CudaBackend::new(0)?;

    let upload = elapsed_average(|| {
        drop(backend.upload_batch(&states)?);
        Ok(())
    });

    let mut resident = backend.upload_batch(&states)?;
    let gpu_results = backend.run_circuit_batch(&states, &circuit)?;
    for (actual, expected) in gpu_results.iter().zip(&cpu_reference) {
        for (actual, expected) in actual.amplitudes().iter().zip(expected.amplitudes()) {
            if (actual.re - expected.re).abs() > 1e-10 || (actual.im - expected.im).abs() > 1e-10 {
                return Err("CUDA batch output differs from the CPU reference".into());
            }
        }
    }

    let resident_kernels = elapsed_average(|| {
        resident.run_circuit(&circuit)?;
        resident.synchronize()?;
        Ok(())
    });
    let download = elapsed_average(|| {
        drop(resident.download()?);
        Ok(())
    });
    let cpu = elapsed_average(|| {
        for state in &states {
            drop(circuit.run(state.clone())?);
        }
        Ok(())
    });
    let end_to_end = elapsed_average(|| {
        drop(backend.run_circuit_batch(&states, &circuit)?);
        Ok(())
    });

    println!(
        "CUDA batch benchmark: {BATCH_SIZE} states x {QUBITS} qubits, {} circuit operations, average over {ITERATIONS} iterations",
        circuit.operations().len()
    );
    println!(
        "CPU batch circuit:          {:.3} ms",
        cpu.as_secs_f64() * 1e3
    );
    println!(
        "GPU allocate + H2D + free:   {:.3} ms",
        upload.as_secs_f64() * 1e3
    );
    println!(
        "GPU resident gate kernels:   {:.3} ms",
        resident_kernels.as_secs_f64() * 1e3
    );
    println!(
        "GPU download + host alloc:   {:.3} ms",
        download.as_secs_f64() * 1e3
    );
    println!(
        "GPU end-to-end batch:        {:.3} ms",
        end_to_end.as_secs_f64() * 1e3
    );
    Ok(())
}
