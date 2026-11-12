//! Command parsing and dispatch for the REPL.

use crate::state::Session;
use quantumforge::analytic::{Assumptions, ExactComplex, Rational, ScalarExpr, Simplifier};
use quantumforge::parser::{parse_circuit, parse_scalar_expr};
use quantumforge::qec::{
    QecCode, encode_repetition_bitflip, repetition_code_bitflip, repetition_code_phaseflip,
};
use quantumforge::tensor_network::{DMRG, MPS, TEBD};
use quantumforge::{
    CircuitOp, Complex64, DensityMatrix, Gate, ParameterizedCircuit, QmlTrainingConfig,
    RotationAxis, SingleQubitObservable, StateVector, VariationalQnn,
};
use std::collections::HashMap;

pub fn dispatch(session: &mut Session, line: &str) -> Result<String, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    let (cmd, rest) = match trimmed.split_once(' ') {
        Some((c, r)) => (c, r.trim()),
        None => (trimmed, ""),
    };

    match cmd {
        "help" => Ok(HELP.to_string()),
        "quit" | "exit" => Err("QUIT".to_string()),
        "qubits" => cmd_qubits(session, rest),
        "circuit" => Ok(format!("circuit = {}", session.circuit)),
        "draw" => Ok(session.draw()),
        "layers" => Ok(session.layers()),
        "census" => Ok(session.census()),
        "add" => cmd_add(session, rest),
        "clear" => cmd_clear(session),
        "run" => cmd_run(session),
        "noisy" => cmd_noisy(session),
        "state" => Ok(session.state_str()),
        "probs" => Ok(session.probs_str()),
        "ptrace" => cmd_ptrace(session, rest),
        "mix" => cmd_mix(session),
        "channel" => cmd_channel(session, rest),
        "purity" => Ok(session.purity_str()),
        "expect" => cmd_expect(session, rest),
        "bloch" => cmd_bloch(session, rest),
        "gradient" => cmd_gradient(session, rest),
        "vqe" => cmd_vqe(session, rest),
        "train" => cmd_train(session, rest),
        "shots" => cmd_shots(session, rest),
        "shots-noise" => cmd_shots_noise(session, rest),
        "qnn" => cmd_qnn(session, rest),
        "qtm" => cmd_qtm(session, rest),
        "mbqc" => cmd_mbqc(session, rest),
        "anneal" => cmd_anneal(session, rest),
        "gas" => cmd_gas(session, rest),
        "cuda" => cmd_cuda(session, rest),
        "gpu" => Ok(cmd_gpu()),
        "tebd" => cmd_tebd(session, rest),
        "dmrg" => cmd_dmrg(session, rest),
        _ => Err(format!("unknown command: {}", cmd)),
    }
}

// Include all the command implementations...
// (This would contain all the cmd_* functions)

const HELP: &str = "\
QuantumForge REPL — one session, live state. Type 'help' for this, 'quit' to exit.

  qubits N             set system size N (resets circuit + state)
  circuit              print the current circuit source
  draw                 ASCII circuit diagram (qubit wires + gates)
  layers               greedy gate scheduling: parallel layers + circuit depth
  census               gate counts by arity (1-qubit vs 2-qubit)
  add <op>             append an op: H[0] Rx[0.5,1] CNOT[0,1] CZ[0,1] SWAP[0,1]
                         Measure[0,0] IfBit[0,1,X,1] AmplitudeDamping[0,0.2]
                         PhaseDamping[0,0.1] Depolarizing[0,0.1]
  clear                empty the circuit (state stays |0..0>)
  run                  apply the circuit, update the live state
  noisy                run the circuit through a density matrix (live mixed state)
  state                show live amplitudes (or density diagonal if noisy)
  probs                show |amp|^2 (or density populations if noisy)
  ptrace Q1,Q2,...     partial trace of the density matrix to a subsystem
  mix                  replace the state with the maximally-mixed state (I/2^n)
  channel dep|ad|pd Q P   apply a noise channel to the density matrix (needs noisy/mix)
  purity               show trace + purity of the density matrix
  expect X|Y|Z Q       expectation value of X/Y/Z on qubit Q
  bloch Q              print Bloch vector (<X>, <Y>, <Z>) for qubit Q
  gradient X|Y|Z Q     d<O>/dtheta for each rotation (adjoint vs parameter-shift)
  vqe <ham>            Variational Quantum Eigensolver: minimize <ham> over the
                         circuit's rotation params; ham = h2 | heh+ | tfim |
                         heisenberg | ising | xxz | hubbard  (exact ground energy shown too)
  train <task>         QML training on a variational circuit (Adam):
                         regression  — fit cos(x+0.7) with an angle-encoded Ry
                         classify    — binary classifier on 2 Gaussian blobs
                         regression DATA [nfeat=K lr=E epochs=N]  — your data
                         classify DATA [nfeat=K lr=E epochs=N]    — your data
                         DATA = flat brace list {f1,..,fK,target, ...}, each
                         K+1 numbers = one row (K features + 1 target), e.g.
                         train regression {0,1, 1,2, 2,3} nfeat=1
                         train classify {-1,0, 1,1} nfeat=1   (labels 0/1)
  shots N SEED         run the circuit as seeded shots, show counts
  shots-noise N SEED   same, but allow noise-channel Kraus trajectories
  qnn [DATA]           1-parameter VariationalQnn (Ry-encode + trainable Ry):
                         bare = fit cos(x+0.7) demo (Adam)
                         qnn {x0,y0, x1,y1, ...} = fit your (x,y) pairs
                         config: target=K (demo) lr=E steps=N opt=adam|gd
  qtm [machine=counter|scan|erase] [cells=N]
                       Quantum Turing Machine: reversible transition unitary
                         U_delta on a superposition of configurations, measured;
                         `erase` shows a non-injective table fixed by ancilla
  mbqc [edges=a-b,..] [qubits=N] [pattern=SPEC]
                       Measurement-Based QC: cluster state on your graph +
                         adaptive measurement pattern (explicit angles/deps or
                         auto) with by-product corrections
  anneal [NAME] [T=.. tol=.. steps=..] [path=linear|fast|poly] [driver=tf|xz]
                        Adiabatic QC: interpolate driver -> problem Hamiltonian
                          NAME = ising|tfim|heisenberg|xxz|hubbard (default demo ZZ)
    gas [qubits=N] [edges=a-b:w,..] [shots=N] [seed=N] [max-outer=N]
                            Grover Adaptive Search for MaxCut:
                              builds oracle for threshold T, runs Grover with k=0,1,2...
                              updates T to best found value, repeats
                              edges: space-separated list like \"0-1:1.0 1-2:0.5\"
        cuda [device=N]         run the current circuit on CUDA GPU:
                                      uploads live state, executes circuit, downloads result
                                      requires `--features cuda` at compile time
        gpu                     report GPU availability and cuBLAS/cuSOLVER status
                                      requires `--features gpu` at compile time
                                      DMRG uses GPU automatically when compiled with --features gpu
          tebd [qubits=N] [dt=..] [tol=..] [max-bond=..] [steps=N]
                                    TEBD imaginary time evolution for TFIM ground state:
";