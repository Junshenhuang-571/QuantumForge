//! Help text for the QuantumForge REPL.

#[allow(clippy::all)]
/// Help text printed by the REPL's `help` command, listing every command
/// with its syntax and options.
pub const HELP: &str = "\
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
  qaoa-transfer [donor=8] [acceptor=N] [p=6] [dt=0.75] [lr=0.05] [iters=100]
                [lambda=1e-4] [full=0] [seed=42]
                        QAOA parameter transfer + targeted-single-layer
                        optimization (arXiv:2601.15760) on seeded 3-regular
                        graphs: optimize the donor, transfer the angles, pick a
                        target layer, optimize only its two angles.
                        acceptor defaults to the live qubit count (otherwise
                        donor+2); prints r_n (transferred), r_s (targeted) and,
                        with full=1, r_f (full re-optimization). lr=0.1 is the
                        paper's value but stalls with finite-difference
                        gradients; 0.05 is the default. Debug builds are tens of
                        times slower than release — lower p/iters there.
  qaoa-angles [which]     print stored angles: which = donor|transferred|optimized
  qaoa-layers            print the cut expectation of every targeted layer
  qaoa-layer K           optimize only layer K and report its ratio
    gas [qubits=N] [edges=a-b:w,..] [shots=N] [seed=N] [max-outer=N]
                            Grover Adaptive Search for MaxCut:
                              builds oracle for threshold T, runs Grover with k=0,1,2...
                              updates T to best found value, repeats
                              edges: space-separated list like \\\"0-1:1.0 1-2:0.5\\\"
        cuda [device=N]         run the current circuit on CUDA GPU:
                                      uploads live state, executes circuit, downloads result
                                      requires `--features cuda` at compile time
        gpu                     report GPU availability and cuBLAS/cuSOLVER status
                                      requires `--features gpu` at compile time
                                      DMRG uses GPU automatically when compiled with --features gpu
          tebd [qubits=N] [dt=..] [tol=..] [max-bond=..] [steps=N]
                                    TEBD imaginary time evolution for TFIM ground state:
                                      H = -Σ Z_i Z_{i+1} - Σ X_i (nearest-neighbor)
                                      outputs ground state energy, updates live MPS
          dmrg [qubits=N] [bond=..] [tol=..] [sweeps=N]
                                    DMRG variational ground state for TFIM:
                                      same Hamiltonian, sweeps until energy converges
                                      outputs energy, updates live MPS
          tebd-correl [site=..]    print <Z_i Z_{i+1}> correlation from live TEBD MPS
          tebd-correl-all          print all nearest-neighbor <Z_i Z_{i+1}> from TEBD
          tebd-sites               print single-site <Z>, <X> for all sites from TEBD
          dmrg-correl [site=..]    print <Z_i Z_{i+1}> correlation from live DMRG MPS
          qec-code [type=bitflip|phaseflip|shor|steane|surface] [n=..] [d=..]
                                    build QEC code, print stabilizers + logical ops
          qec-encode [type=bitflip|phaseflip] [n=..] [state=0|1]
                                    encode logical |0> or |1> into repetition code,
                                    updates live state to encoded physical qubits
          qec-syndrome             measure syndrome on live state using code's circuit
          qec-recover [syndrome=N] apply recovery circuit for given syndrome bits
          eval EXPR            numeric eval of a scalar expression, e.g. 2 + 3*Pi
  simplify EXPR        canonicalize a scalar expression
  diff EXPR            symbolic derivative d/dx of EXPR
  pauli EXPR           evaluate a Pauli sum like X, 2*Y, I+Z (1 qubit)
  assume X             declare a symbol real (affects Sqrt/abs simplification)
  help                 this text
  quit                 exit
";
