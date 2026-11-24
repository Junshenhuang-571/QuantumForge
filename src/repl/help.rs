//! Help text for the QuantumForge REPL.

#[allow(clippy::all)]
/// Help text printed by the REPL's `help` command, listing every command
/// with its syntax and options.
pub const HELP: &str = "\
QuantumForge REPL — one session, live state.
Type 'help' for this text, 'quit' to exit.

SESSION AND STATE
  qubits N                    set system size N (resets circuit and state)
  clear                       empty the circuit (state stays |0..0>)
  run                         apply the circuit, update the live state
  noisy                       run through a density matrix (live mixed state)
  state                       show live amplitudes (or density diagonal if noisy)
  probs                       show |amp|^2 (or density populations if noisy)
  mix                         replace the state with the maximally-mixed state I/2^n
  ptrace Q1,Q2,...            partial trace of the density matrix to a subsystem
  purity                      trace and purity of the density matrix
  entropy                     von Neumann entropy S = -Tr(rho ln rho)
  channel KIND Q P [BETA]     apply a noise channel to the density matrix (needs noisy/mix).
                              KIND = dep|ad|pd|th; th is thermal damping with excitation rate
                              exp(-BETA)*gamma_down.
  thermal [HAM] [BETA]        build the Gibbs state exp(-BETA H)/Z of the named Hamiltonian and
                              make it the live density matrix. HAM = tfim, BETA default 1.0.

CIRCUIT
  add <OP>                    append an op:
                              H[0] Rx[0.5,1] CNOT[0,1] CZ[0,1] SWAP[0,1] Measure[0,0]
                              IfBit[0,1,X,1] AmplitudeDamping[0,0.2] PhaseDamping[0,0.1]
                              Depolarizing[0,0.1]
  circuit                     print the current circuit source
  draw                        ASCII circuit diagram (qubit wires + gates)
  layers                      greedy gate scheduling: parallel layers and depth
  census                      gate counts by arity (1-qubit vs 2-qubit)
  shots N SEED                run the circuit as seeded shots, show counts
  shots-noise N SEED          same, but allow noise-channel Kraus trajectories

OBSERVABLES
  expect X|Y|Z Q              expectation value of X/Y/Z on qubit Q
  bloch Q                     Bloch vector (<X>, <Y>, <Z>) for qubit Q
  gradient X|Y|Z Q            d<O>/dtheta for each rotation. METHOD =
                              auto|shift|adjoint|backprop|fd;
                              without it the exact rules are cross-validated, with it one rule runs
                              and its evaluation cost is reported.

VARIATIONAL AND QML
  vqe <HAM>                   minimize <HAM> over the circuit's rotation params; the exact ground
                              energy is shown too. HAM = h2 | heh+ | tfim | heisenberg | ising |
                              xxz | hubbard.
  vqe-subscore                SubsCoRe (arXiv:2502.01704): VQE with adaptive observation cost
                              control. Optimizes one parameter axis at a time and spends shots
                              where they matter under a total shot budget.
                              [ham=tfim] [budget=20000] [shots=64] [strategy=center|bound]
                              [update=mean|thompson|guarded] [explore=1.0] [seed=42]
                              [entropy=off] [patience=3]
                              update picks the zero-temperature step (mean) or a finite-temperature
                              Thompson step; guarded refuses a kick the surrogate calls uphill.
                              entropy=T stops once an iteration's information gain stays below T
                              nats for `patience` iterations (off by default); the gain is
                              D_KL(posterior || prior) on each observed marginal.
  train <TASK>                QML training on a variational circuit (Adam). TASK =
                              regression | classify | regression DATA | classify DATA
                              DATA = flat brace list {f1,..,fK,target, ...}, K+1 numbers per row.
                              config: nfeat=K lr=E epochs=N
                              train regression {0,1, 1,2, 2,3} nfeat=1
                              train classify {-1,0, 1,1} nfeat=1 (labels 0/1)
  qnn [DATA]                  1-parameter VariationalQnn (Ry-encode + trainable Ry). Bare = fit
                              cos(x+0.7) demo; qnn {x0,y0, x1,y1, ...} fits your pairs.
                              config: target=K (demo) lr=E steps=N opt=adam|gd

ALGORITHMS
  qtm [MACHINE] [CELLS]       Quantum Turing Machine: reversible transition unitary U_delta on a
                              superposition of configurations, then measured. MACHINE =
                              counter|scan|erase; erase shows a non-injective table fixed by
                              ancilla.
  mbqc [SPEC]                 Measurement-Based QC: cluster state on your graph plus an adaptive
                              measurement pattern with by-product corrections.
                              [edges=a-b,..] [qubits=N] [pattern=SPEC]
  anneal [NAME]               Adiabatic QC: interpolate driver -> problem Hamiltonian.
                              NAME = ising|tfim|heisenberg|xxz|hubbard (default demo ZZ)
                              [T=.. tol=.. steps=..] [path=linear|fast|poly] [driver=tf|xz]
  qaoa-transfer               QAOA parameter transfer + targeted-layer optimization
                              (arXiv:2601.15760): optimize the donor, transfer the angles, then
                              jointly optimize the last `layers` layers (the paper uses 1; 3
                              recovers most of the gap to a full re-optimization).
                              [donor=8] [acceptor=N] [family=3r] [p=6] [dt=0.75] [lr=0.05]
                              [iters=100] [lambda=1e-4] [layers=3] [sweeps=2] [full=0] [seed=42]
                              family = 3r | er | ba, prefixed with w for weighted (w3r, wer);
                              m = ER edge probability (default 0.5) or BA edges per new vertex (2).
                              Prints r_n (transferred), r_s (targeted), the per-sweep refinement
                              and, with full=1, r_f (full re-optimization). Debug builds are tens
                              of times slower than release.
  qaoa-angles [WHICH]         print stored angles: WHICH = donor|transferred|optimized
  qaoa-layers                 scan every single layer, print its cut expectation
  qaoa-layer K                optimize only layer K and report its ratio
  gas [N] [EDGES]             Grover Adaptive Search for MaxCut: build an oracle for threshold T,
                              run Grover with k=0,1,2..., update T to the best found value, repeat.
                              EDGES = \"0-1:1.0 1-2:0.5\"; also [shots=N] [seed=N] [max-outer=N]
  tebd [N]                    TEBD imaginary-time evolution for the TFIM ground state
                              H = -sum Z_i Z_{i+1} - sum X_i. Outputs energy, updates live MPS.
                              [dt=..] [tol=..] [max-bond=..] [steps=N]
  dmrg [N]                    DMRG variational ground state for the same TFIM Hamiltonian; sweeps
                              until the energy converges, updates live MPS.
                              [bond=..] [tol=..] [sweeps=N]
  tebd-correl [SITE]          print <Z_i Z_{i+1}> from the live TEBD MPS
  tebd-correl-all             print all nearest-neighbor <Z_i Z_{i+1}> from TEBD
  tebd-sites                  print single-site <Z>, <X> for all sites from TEBD
  dmrg-correl [SITE]          print <Z_i Z_{i+1}> from the live DMRG MPS
  qec-code [TYPE]             build a QEC code, print stabilizers and logical ops.
                              TYPE = bitflip|phaseflip|shor|steane|surface; [n=..] [d=..]
  qec-encode [TYPE] [STATE]   encode logical |0> or |1> into the repetition code; updates the
                              live state to the encoded physical qubits. TYPE = bitflip|phaseflip
  qec-syndrome                measure the syndrome on the live state
  qec-recover [SYNDROME]      apply the recovery circuit for the given syndrome bits
  cuda [DEVICE]               run the current circuit on a CUDA GPU (uploads live state, executes,
                              downloads). Requires --features cuda at compile time.
  gpu                         report GPU availability and cuBLAS/cuSOLVER status. Requires
                              --features gpu; DMRG uses the GPU automatically when compiled with it.

SYMBOLIC MATH
  eval EXPR                   numeric evaluation, e.g. eval 2 + 3*Pi
  simplify EXPR               canonicalize a scalar expression
  diff EXPR                   symbolic derivative d/dx of EXPR
  pauli EXPR                  evaluate a Pauli sum like X, 2*Y, I+Z (1 qubit)
  assume X                    declare a symbol real (affects Sqrt/abs simplification)

GENERAL
  help                        print this text
  quit                        exit the REPL
";
