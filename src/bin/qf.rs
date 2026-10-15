//! Interactive QuantumForge REPL.
//!
//! One persistent session with a live circuit, live state, and live math.
//! Commands are dispatched on the first whitespace-delimited word; scalar
//! expressions and circuit bodies are passed verbatim to the existing parser.

use quantumforge::analytic::{Assumptions, ExactComplex, Rational, ScalarExpr, Simplifier};
use quantumforge::parser::{parse_circuit, parse_scalar_expr};
use quantumforge::{
    CircuitOp, Complex64, DensityMatrix, Gate, ParameterizedCircuit, QmlTrainingConfig,
    RotationAxis, SingleQubitObservable, StateVector, VariationalQnn,
};
use std::collections::HashMap;

struct Session {
    qubits: usize,
    circuit: String,
    state: StateVector,
    /// When Some, `state`/`probs`/`expect` read from this mixed state instead.
    density: Option<DensityMatrix>,
    assume: Assumptions,
    qnn: VariationalQnn,
    shots: usize,
    seed: u64,
}

const HELP: &str = "\
QuantumForge REPL — one session, live state. Type 'help' for this, 'quit' to exit.

  qubits N             set system size N (resets circuit + state)
  circuit              print the current circuit source
  draw                 ASCII circuit diagram (qubit wires + gates)
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
  eval EXPR            numeric eval of a scalar expression, e.g. 2 + 3*Pi
  simplify EXPR        canonicalize a scalar expression
  diff EXPR            symbolic derivative d/dx of EXPR
  pauli EXPR           evaluate a Pauli sum like X, 2*Y, I+Z (1 qubit)
  assume X             declare a symbol real (affects Sqrt/abs simplification)
  help                 this text
  quit                 exit
";

fn main() {
    let mut session = Session::new();
    println!("QuantumForge REPL — type 'help' for commands, 'quit' to exit.");
    let mut line = String::new();
    loop {
        line.clear();
        print!("qf> ");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if !session.dispatch(trimmed) {
                    break;
                }
            }
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        }
    }
    println!("bye.");
}

impl Session {
    fn new() -> Self {
        let qubits = 1;
        Self {
            qubits,
            circuit: String::new(),
            state: StateVector::zero(qubits).expect("initial state"),
            density: None,
            assume: Assumptions::default(),
            qnn: VariationalQnn,
            shots: 256,
            seed: 42,
        }
    }

    fn reset_state(&mut self) -> Result<(), String> {
        self.state = StateVector::zero(self.qubits).map_err(|e| e.to_string())?;
        self.density = None;
        Ok(())
    }

    fn full_circuit_source(&self) -> String {
        if self.circuit.is_empty() {
            "Circuit[]".to_owned()
        } else {
            format!("Circuit[{}]", self.circuit)
        }
    }

    /// Returns false to exit the loop.
    fn dispatch(&mut self, line: &str) -> bool {
        let mut parts = line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("");
        let rest = parts.next().unwrap_or("").trim();
        match cmd {
            "quit" | "exit" | "q" => return false,
            "help" | "?" => print!("{HELP}"),
            "qubits" => self.cmd_qubits(rest),
            "circuit" => println!("{}", self.full_circuit_source()),
            "draw" => self.cmd_draw(),
            "add" => self.cmd_add(rest),
            "clear" => {
                self.circuit.clear();
                let _ = self.reset_state();
                println!("cleared.");
            }
            "run" => self.cmd_run(),
            "noisy" => self.cmd_noisy(),
            "state" => self.cmd_state(),
            "probs" => self.cmd_probs(),
            "ptrace" => self.cmd_ptrace(rest),
            "mix" => self.cmd_mix(),
            "expect" => self.cmd_expect(rest),
            "expz" => self.cmd_expect(&format!("Z {rest}")),
            "bloch" => self.cmd_bloch(rest),
            "gradient" => self.cmd_gradient(rest),
            "vqe" => self.cmd_vqe(rest),
            "train" => self.cmd_train(rest),
            "shots" => self.cmd_shots(rest, false),
            "shots-noise" => self.cmd_shots(rest, true),
            "qnn" => self.cmd_qnn(rest),
            "qtm" => self.cmd_qtm(rest),
            "mbqc" => self.cmd_mbqc(rest),
            "anneal" => self.cmd_anneal(rest),
            "eval" => self.cmd_eval(rest),
            "simplify" => self.cmd_simplify(rest),
            "diff" => self.cmd_diff(rest),
            "pauli" => self.cmd_pauli(rest),
            "assume" => {
                self.assume.set_real(rest);
                println!("assumed {rest} is real.");
            }
            other => eprintln!("unknown command: {other} (type 'help')"),
        }
        true
    }

    fn cmd_qubits(&mut self, rest: &str) {
        match rest.parse::<usize>() {
            Ok(n) if (1..=20).contains(&n) => {
                self.qubits = n;
                self.circuit.clear();
                let _ = self.reset_state();
                println!("qubits = {n}");
            }
            Ok(n) => eprintln!("qubit count {n} out of range 1..=20"),
            Err(_) => eprintln!("usage: qubits N"),
        }
    }

    fn cmd_add(&mut self, rest: &str) {
        if rest.is_empty() {
            eprintln!("usage: add <op>   e.g. add H[0]");
            return;
        }
        // Validate eagerly (syntax AND semantics) so a bad op never mutates the
        // circuit. Syntax: parse_circuit. Semantics: dry-run the whole circuit
        // through the execution model that supports its ops. `run_shots_with_noise`
        // supports BOTH noise channels and measurement/classical-conditional, so
        // it's chosen whenever either is present; a plain state-vector circuit
        // (no channels, no measurement) validates via `run`. Any path exercises
        // the real QuantumError bounds (qubit index, duplicate pair, classical
        // bit) before commit.
        let separator = if self.circuit.is_empty() { "" } else { ", " };
        let trial = format!("Circuit[{}{}{}]", self.circuit, separator, rest);
        let valid = (|| {
            let circuit = parse_circuit(&trial).map_err(|e| e.to_string())?;
            let initial = StateVector::zero(self.qubits).map_err(|e| e.to_string())?;
            let has = |f: fn(&CircuitOp) -> bool| circuit.operations().iter().any(f);
            let needs_shots = has(|op| matches!(op, CircuitOp::Channel { .. }))
                || has(|op| {
                    matches!(
                        op,
                        CircuitOp::Measure { .. } | CircuitOp::ConditionalSingleQubit { .. }
                    )
                });
            if needs_shots {
                circuit
                    .run_shots_with_noise(&initial, 1, 0)
                    .map_err(|e| e.to_string())?;
            } else {
                circuit.run(initial).map_err(|e| e.to_string())?;
            }
            Ok::<(), String>(())
        })();
        match valid {
            Ok(()) => {
                self.circuit.push_str(separator);
                self.circuit.push_str(rest);
                println!("added. circuit = {}", self.full_circuit_source());
            }
            Err(msg) => eprintln!("invalid op: {msg}"),
        }
    }

    fn cmd_run(&mut self) {
        let circuit = match parse_circuit(&self.full_circuit_source()) {
            Ok(c) => c,
            Err(e) => return eprintln!("circuit parse error: {e}"),
        };
        let initial = match StateVector::zero(self.qubits) {
            Ok(s) => s,
            Err(e) => return eprintln!("state error: {e}"),
        };
        match circuit.run(initial) {
            Ok(state) => {
                self.state = state;
                self.density = None;
                self.cmd_state();
            }
            Err(e) => eprintln!("run error: {e}"),
        }
    }

    fn cmd_state(&self) {
        match &self.density {
            Some(d) => {
                println!(
                    "density state ({} qubits): trace={:.6} purity={:.6}",
                    d.qubits(),
                    d.trace().re,
                    d.purity()
                );
                let n = 1usize << self.qubits;
                for i in 0..n {
                    let p = d.entries()[i * n + i].re;
                    if p.abs() > 1e-9 {
                        let bits = format!("{:0>width$b}", i, width = self.qubits);
                        println!("  |{bits}>  {p:+.6}  (population)");
                    }
                }
            }
            None => {
                println!("state ({} qubits):", self.qubits);
                for (i, a) in self.state.amplitudes().iter().enumerate() {
                    if *a != Complex64::ZERO {
                        let bits = format!("{:0>width$b}", i, width = self.qubits);
                        println!("  |{bits}>  {:+7.4}{:+7.4}i", a.re, a.im);
                    }
                }
            }
        }
    }

    fn cmd_probs(&self) {
        match &self.density {
            Some(d) => {
                let n = 1usize << self.qubits;
                for i in 0..n {
                    let p = d.entries()[i * n + i].re;
                    if p.abs() > 1e-9 {
                        println!("  |{i:0width$b}>  {p:.6}", width = self.qubits);
                    }
                }
            }
            None => {
                for (i, p) in self.state.probabilities().iter().enumerate() {
                    if *p != 0.0 {
                        println!("  |{i:0width$b}>  {p:.6}", width = self.qubits);
                    }
                }
            }
        }
    }

    /// `ptrace KEEP` — partial trace of the live density matrix down to the
    /// subsystem `KEEP` (comma-separated qubit indices), tracing out the rest.
    /// Reduces the register to `|KEEP|` qubits. Requires a density-matrix state
    /// (use `noisy` or `mix` first). Example: `ptrace 0,2`.
    fn cmd_ptrace(&mut self, rest: &str) {
        let keep: Vec<usize> = rest
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        if keep.is_empty() {
            return eprintln!("usage: ptrace Q1,Q2,...   (qubits to keep; others traced out)");
        }
        let d = match &self.density {
            Some(d) => d,
            None => {
                return eprintln!("ptrace needs a density matrix — run `noisy` or `mix` first");
            }
        };
        for &q in &keep {
            if q >= d.qubits() {
                return eprintln!(
                    "ptrace error: qubit {q} out of range ({} qubits)",
                    d.qubits()
                );
            }
        }
        match d.partial_trace(&keep) {
            Ok(sub) => {
                let n = sub.qubits();
                self.density = Some(sub);
                self.qubits = n;
                println!("ptrace: kept qubits {keep:?} -> {n}-qubit reduced state");
                self.cmd_state();
            }
            Err(e) => eprintln!("ptrace error: {e}"),
        }
    }

    /// `mix` — replace the live state with the maximally-mixed state on the
    /// current qubit count (I/2^n). A convenience for exploring mixed-state
    /// dynamics without building a noisy circuit first.
    fn cmd_mix(&mut self) {
        match DensityMatrix::maximally_mixed(self.qubits) {
            Ok(d) => {
                self.density = Some(d);
                println!("mix: maximally-mixed state on {} qubits", self.qubits);
                self.cmd_state();
            }
            Err(e) => eprintln!("mix error: {e}"),
        }
    }

    /// `expect X|Y|Z Q` — expectation value on the live state or density matrix.
    fn cmd_expect(&self, rest: &str) {
        let mut parts = rest.split_whitespace();
        let obs = match parts.next() {
            Some("X") => SingleQubitObservable::X,
            Some("Y") => SingleQubitObservable::Y,
            Some("Z") => SingleQubitObservable::Z,
            _ => return eprintln!("usage: expect X|Y|Z Q"),
        };
        let q = match parts.next().and_then(|q| q.parse::<usize>().ok()) {
            Some(q) => q,
            None => return eprintln!("usage: expect X|Y|Z Q"),
        };
        if let Some(d) = &self.density {
            let space = match quantumforge::quantum::HilbertSpace::qubits(self.qubits) {
                Ok(s) => s,
                Err(e) => return eprintln!("error: {e}"),
            };
            let op = match obs {
                SingleQubitObservable::X => quantumforge::quantum::Operator::pauli_x(),
                SingleQubitObservable::Y => quantumforge::quantum::Operator::pauli_y(),
                SingleQubitObservable::Z => quantumforge::quantum::Operator::pauli_z(),
            };
            let observable = match quantumforge::quantum::Observable::new(op, 1e-12) {
                Ok(o) => o,
                Err(e) => return eprintln!("error: {e}"),
            };
            let _ = space;
            match observable.expectation_density(d) {
                Ok(v) => println!("<{obs:?}_{q}> = {:+.6}", v.re),
                Err(e) => eprintln!("error: {e}"),
            }
        } else {
            let value = match obs {
                SingleQubitObservable::X => self.state.expectation_x(q),
                SingleQubitObservable::Y => self.state.expectation_y(q),
                SingleQubitObservable::Z => self.state.expectation_z(q),
            };
            match value {
                Ok(v) => println!("<{obs:?}_{q}> = {v:+.6}"),
                Err(e) => eprintln!("error: {e}"),
            }
        }
    }

    fn cmd_bloch(&self, rest: &str) {
        let q = match rest.parse::<usize>() {
            Ok(q) => q,
            Err(_) => return eprintln!("usage: bloch Q"),
        };
        let (x, y, z) = if let Some(d) = &self.density {
            let op = |p: fn() -> quantumforge::quantum::Operator| {
                let o =
                    quantumforge::quantum::Observable::new(p(), 1e-12).expect("pauli is hermitian");
                o.expectation_density(d).map(|v| v.re).unwrap_or(f64::NAN)
            };
            (
                op(quantumforge::quantum::Operator::pauli_x),
                op(quantumforge::quantum::Operator::pauli_y),
                op(quantumforge::quantum::Operator::pauli_z),
            )
        } else {
            (
                self.state.expectation_x(q).unwrap_or(f64::NAN),
                self.state.expectation_y(q).unwrap_or(f64::NAN),
                self.state.expectation_z(q).unwrap_or(f64::NAN),
            )
        };
        println!("bloch q{q}: <X>={x:+.4} <Y>={y:+.4} <Z>={z:+.4}");
    }

    /// ASCII circuit diagram: one row per qubit wire, one column per time step.
    /// Two-qubit gates draw a vertical `│` connector across the wires they span.
    fn cmd_draw(&self) {
        let circuit = match parse_circuit(&self.full_circuit_source()) {
            Ok(c) => c,
            Err(e) => return eprintln!("circuit parse error: {e}"),
        };
        let ops = circuit.operations();
        if ops.is_empty() {
            println!("(empty circuit)");
            return;
        }
        // Fixed-width columns; gate labels sit in their CELL, the wire stays
        // continuous (dashes) everywhere else. Markers at the cell midpoint so
        // vertical connectors line up under them.
        let mid = CELL / 2;
        let width = PREFIX + ops.len() * CELL;
        let mut rows: Vec<String> = (0..self.qubits)
            .map(|q| {
                let mut r = format!("q{q} ");
                r.push_str(&"─".repeat(width - PREFIX));
                r
            })
            .collect();
        for (col, op) in ops.iter().enumerate() {
            let x = PREFIX + col * CELL + mid;
            match op {
                CircuitOp::SingleQubit { gate, target } => {
                    label(&mut rows[*target], col, &gate_label(gate));
                }
                CircuitOp::Channel { channel, target } => {
                    label(&mut rows[*target], col, &channel_label(channel));
                }
                CircuitOp::Cnot { control, target } => {
                    connect(&mut rows, *control, *target, x);
                    put_at(&mut rows[*control], x, "•");
                    put_at(&mut rows[*target], x, "⊕");
                }
                CircuitOp::Cz { first, second } => {
                    connect(&mut rows, *first, *second, x);
                    put_at(&mut rows[*first], x, "•");
                    put_at(&mut rows[*second], x, "•");
                }
                CircuitOp::Swap { first, second } => {
                    connect(&mut rows, *first, *second, x);
                    put_at(&mut rows[*first], x, "x");
                    put_at(&mut rows[*second], x, "x");
                }
                CircuitOp::Measure { target, .. } => {
                    put_at(&mut rows[*target], x, "M");
                }
                CircuitOp::ConditionalSingleQubit { gate, target, .. } => {
                    label(&mut rows[*target], col, &gate_label(gate));
                }
            }
        }
        println!("{}", rows.join("\n"));
        println!("legend: ─ wire  • control  ⊕ target(X)  │ span  x swap  M measure");
    }

    /// Build a ParameterizedCircuit from the live circuit, assigning one trainable
    /// parameter per rotation occurrence (in order). Fixed gates, entanglers, and
    /// non-differentiable ops (channels/measurement/control) are carried through
    /// as fixed. Returns the circuit and the initial parameter vector, or an error
    /// string if there are no rotations to optimize.
    fn build_parameterized(&self) -> Result<(ParameterizedCircuit, Vec<f64>), String> {
        let circuit = parse_circuit(&self.full_circuit_source()).map_err(|e| e.to_string())?;
        let rotation_count = circuit
            .operations()
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    CircuitOp::SingleQubit {
                        gate: Gate::Rx(_) | Gate::Ry(_) | Gate::Rz(_),
                        ..
                    }
                )
            })
            .count();
        if rotation_count == 0 {
            return Err("no rotation gates in circuit — nothing to optimize".to_owned());
        }
        let mut pc = ParameterizedCircuit::new(rotation_count);
        let mut params: Vec<f64> = Vec::with_capacity(rotation_count);
        for op in circuit.operations() {
            match op {
                CircuitOp::SingleQubit {
                    gate: Gate::Rx(t),
                    target,
                } => {
                    pc.rotation(RotationAxis::X, *target, params.len());
                    params.push(*t);
                }
                CircuitOp::SingleQubit {
                    gate: Gate::Ry(t),
                    target,
                } => {
                    pc.rotation(RotationAxis::Y, *target, params.len());
                    params.push(*t);
                }
                CircuitOp::SingleQubit {
                    gate: Gate::Rz(t),
                    target,
                } => {
                    pc.rotation(RotationAxis::Z, *target, params.len());
                    params.push(*t);
                }
                CircuitOp::SingleQubit { gate, target } => pc.push(*gate, *target),
                CircuitOp::Cnot { control, target } => pc.cnot(*control, *target),
                CircuitOp::Cz { first, second } => pc.cz(*first, *second),
                CircuitOp::Swap { first, second } => pc.swap(*first, *second),
                // Channels are not differentiable via adjoint; carry as fixed.
                CircuitOp::Channel { .. }
                | CircuitOp::Measure { .. }
                | CircuitOp::ConditionalSingleQubit { .. } => {}
            }
        }
        Ok((pc, params))
    }

    /// `gradient X|Y|Z Q` — build a parameterized circuit from the live circuit's
    /// rotations and print both the adjoint gradient and the parameter-shift
    /// gradient for cross-validation.
    fn cmd_gradient(&self, rest: &str) {
        let mut parts = rest.split_whitespace();
        let obs = match parts.next() {
            Some("X") => SingleQubitObservable::X,
            Some("Y") => SingleQubitObservable::Y,
            Some("Z") => SingleQubitObservable::Z,
            _ => return eprintln!("usage: gradient X|Y|Z Q"),
        };
        let q = match parts.next().and_then(|q| q.parse::<usize>().ok()) {
            Some(q) => q,
            None => return eprintln!("usage: gradient X|Y|Z Q"),
        };
        let (pc, params) = match self.build_parameterized() {
            Ok(v) => v,
            Err(e) => return eprintln!("{e}"),
        };
        let adjoint = match pc.adjoint_gradient(&params, self.qubits, obs, q) {
            Ok(g) => g,
            Err(e) => return eprintln!("adjoint error: {e}"),
        };
        let shift = match pc.parameter_shift_gradient_for(&params, self.qubits, obs, q) {
            Ok(g) => g,
            Err(e) => return eprintln!("parameter-shift error: {e}"),
        };
        println!("d<{obs:?}_{q}>/dtheta over {} rotations:", params.len());
        for (i, ((p, a), s)) in params.iter().zip(&adjoint).zip(&shift).enumerate() {
            let agree = (a - s).abs() < 1e-6;
            println!(
                "  θ[{i}]={p:+.4}  adjoint={a:+.6}  shift={s:+.6}  {}",
                if agree { "✓ agree" } else { "✗ MISMATCH" }
            );
        }
    }

    /// Variational Quantum Eigensolver: minimize <H> over the circuit's rotation
    /// parameters, then compare against the exact ground-state energy from a
    /// sparse Hermitian eigensolver. Uses gradient descent with the analytic
    /// adjoint gradient of <H> (sum of per-Pauli-term gradients).
    fn cmd_vqe(&self, rest: &str) {
        let name = rest.trim().to_ascii_lowercase();
        let hamiltonian = match named_hamiltonian(&name, self.qubits) {
            Some(h) => h,
            None => {
                return eprintln!(
                    "unknown or ill-sized hamiltonian '{rest}' for {} qubits; try: tfim, ising, heisenberg, xxz, hubbard",
                    self.qubits
                );
            }
        };
        let (pc, mut params) = match self.build_parameterized() {
            Ok(v) => v,
            Err(e) => return eprintln!("{e}"),
        };
        // Energy = <H>(theta).
        let energy = |p: &[f64]| -> f64 {
            let state = match pc.run(p, self.qubits) {
                Ok(s) => s,
                Err(_) => return f64::INFINITY,
            };
            let ket = match state.to_ket() {
                Ok(k) => k,
                Err(_) => return f64::INFINITY,
            };
            hamiltonian
                .expectation_ket(&ket)
                .map(|v| v.re)
                .unwrap_or(f64::INFINITY)
        };
        // Gradient of the total energy <H> by the parameter-shift rule applied per
        // parameter: d<H>/dθᵢ = 0.5·(<H>(θᵢ+π/2) - <H>(θᵢ-π/2)). This is exact for
        // any Pauli-word Hamiltonian (including multi-qubit ZZ/XX+YY+ZZ terms),
        // unlike a single-qubit-observable adjoint decomposition.
        let half_pi = std::f64::consts::FRAC_PI_2;
        let gradient = |p: &[f64]| -> Vec<f64> {
            let mut grad = vec![0.0; p.len()];
            for i in 0..p.len() {
                let mut plus = p.to_vec();
                plus[i] += half_pi;
                let mut minus = p.to_vec();
                minus[i] -= half_pi;
                grad[i] = 0.5 * (energy(&plus) - energy(&minus));
            }
            grad
        };

        // Gradient descent with a shrinking step; deterministic and dependency-free.
        let initial_energy = energy(&params);
        let mut step = 0.4;
        let mut best = initial_energy;
        for _ in 0..400 {
            let grad = gradient(&params);
            let candidate: Vec<f64> = params
                .iter()
                .zip(&grad)
                .map(|(p, g)| p - step * g)
                .collect();
            let value = energy(&candidate);
            if value < best - 1e-12 {
                best = value;
                params = candidate;
            } else {
                step *= 0.5; // shrink and retry
                if step < 1e-9 {
                    break;
                }
            }
        }

        // Exact ground-state energy for comparison.
        let exact = hamiltonian
            .to_sparse_operator()
            .ok()
            .and_then(|op| op.lowest_eigenpairs(1, 1e-10, 4096).ok())
            .map(|sys| sys.eigenvalues()[0]);

        println!(
            "vqe: hamiltonian '{name}' ({} qubits, {} params)",
            self.qubits,
            params.len()
        );
        println!("  initial energy = {initial_energy:.6}");
        println!("  vqe ground energy = {best:.6}");
        match exact {
            Some(e) => {
                let gap = (best - e).abs();
                println!("  exact ground energy = {e:.6}");
                println!(
                    "  |vqe - exact| = {gap:.2e}  {}",
                    if gap < 1e-3 {
                        "✓ converged"
                    } else {
                        "(ansatz/optimizer gap)"
                    }
                );
            }
            None => println!("  exact ground energy = (solver unavailable)"),
        }
        println!("  params = {params:?}");
    }

    /// QML training on a purpose-built variational circuit (angle encoding +
    /// trainable rotation + <Z> readout), optimized with Adam.
    ///
    /// - `train regression` fit the built-in cos(x+0.7) demo
    /// - `train classify` run the built-in separable-blobs demo
    /// - `train regression DATA ...` fit your own data
    /// - `train classify DATA ...` classify your own data
    ///
    /// DATA is a Wolfram-style list of {features..., target} rows, e.g.
    /// `train regression {{0,1},{1,2},{2,3},{3,4}}` or
    /// `train classify {{-1,0},{1,1}}`. Optional trailing config: lr=0.1 epochs=200
    fn cmd_train(&self, rest: &str) {
        let rest = rest.trim();
        // Split off the task word; the remainder is DATA + optional config.
        let (task, tail) = match rest.split_once(char::is_whitespace) {
            Some((t, r)) => (t.to_ascii_lowercase(), r.trim()),
            None => (rest.to_ascii_lowercase(), ""),
        };
        let (data_src, config) = split_config(tail);
        match task.as_str() {
            "regression" => {
                if data_src.is_empty() {
                    self.train_regression_demo();
                } else {
                    self.train_regression_data(data_src, &config);
                }
            }
            "classify" | "classification" => {
                if data_src.is_empty() {
                    self.train_classification_demo();
                } else {
                    self.train_classification_data(data_src, &config);
                }
            }
            other => eprintln!("unknown task '{other}'; try: regression, classify"),
        }
    }

    fn train_regression_demo(&self) {
        let target = 0.7_f64;
        let data: Vec<(Vec<f64>, f64)> = [-1.0, -0.5, 0.0, 0.5, 1.0]
            .into_iter()
            .map(|x| (vec![x], (x + target).cos()))
            .collect();
        run_regression(&data, "cos(x+0.7) demo", &TrainConfig::default());
    }

    fn train_classification_demo(&self) {
        let data: Vec<(Vec<f64>, f64)> = [
            (-1.5, 0.0),
            (-1.2, 0.0),
            (-0.9, 0.0),
            (-0.6, 0.0),
            (0.6, 1.0),
            (0.9, 1.0),
            (1.2, 1.0),
            (1.5, 1.0),
        ]
        .into_iter()
        .map(|(x, y)| (vec![x], y))
        .collect();
        run_classification(&data, "separable-blobs demo", &TrainConfig::default());
    }

    fn train_regression_data(&self, data_src: &str, config: &TrainConfig) {
        let data = match parse_dataset(data_src, config.nfeat) {
            Ok(d) => d,
            Err(e) => return eprintln!("data error: {e}"),
        };
        run_regression(&data, "your data", config);
    }

    fn train_classification_data(&self, data_src: &str, config: &TrainConfig) {
        let data = match parse_dataset(data_src, config.nfeat) {
            Ok(d) => d,
            Err(e) => return eprintln!("data error: {e}"),
        };
        // Labels must be exactly 0 or 1 for binary cross-entropy.
        if data.iter().any(|(_, y)| !matches!(*y, 0.0 | 1.0)) {
            return eprintln!("classify labels must be 0 or 1");
        }
        run_classification(&data, "your data", config);
    }

    fn cmd_shots(&mut self, rest: &str, allow_noise: bool) {
        let mut parts = rest.split_whitespace();
        let shots = match parts.next() {
            Some(n) => match n.parse::<usize>() {
                Ok(n) if n > 0 => n,
                _ => return eprintln!("usage: shots N SEED"),
            },
            None => return eprintln!("usage: shots N SEED"),
        };
        let seed = match parts.next() {
            Some(s) => match s.parse::<u64>() {
                Ok(s) => s,
                Err(_) => return eprintln!("SEED must be a nonnegative integer"),
            },
            None => return eprintln!("usage: shots N SEED"),
        };
        self.shots = shots;
        self.seed = seed;

        let circuit = match parse_circuit(&self.full_circuit_source()) {
            Ok(c) => c,
            Err(e) => return eprintln!("circuit parse error: {e}"),
        };
        let initial = match StateVector::zero(self.qubits) {
            Ok(s) => s,
            Err(e) => return eprintln!("state error: {e}"),
        };
        let result = if allow_noise {
            circuit.run_shots_with_noise(&initial, shots, seed)
        } else {
            circuit.run_shots(&initial, shots, seed)
        };
        match result {
            Ok(r) => {
                println!("shots: {} (seed {seed})", r.shots);
                for (bits, count) in r.counts {
                    let s: String = bits.iter().rev().map(|b| char::from(b'0' + *b)).collect();
                    println!("  |{s}>  {count}");
                }
            }
            Err(e) => eprintln!("shots error: {e}"),
        }
    }

    /// Run the circuit through a density matrix and keep it live, so damping /
    /// depolarizing states appear in `state`/`probs`/`expect`/`bloch` as mixed
    /// states. `run` returns to the pure state-vector view. Circuits mixing
    /// noise channels with measurement/classical-control can't evolve a single
    /// density matrix; those must use `shots-noise` instead.
    fn cmd_noisy(&mut self) {
        let circuit = match parse_circuit(&self.full_circuit_source()) {
            Ok(c) => c,
            Err(e) => return eprintln!("circuit parse error: {e}"),
        };
        if circuit.operations().iter().any(|op| {
            matches!(
                op,
                CircuitOp::Measure { .. } | CircuitOp::ConditionalSingleQubit { .. }
            )
        }) {
            return eprintln!(
                "this circuit mixes noise with measurement/classical-control; \
                 use `shots-noise N SEED` for Kraus-trajectory counts"
            );
        }
        let initial = match StateVector::zero(self.qubits) {
            Ok(s) => s,
            Err(e) => return eprintln!("state error: {e}"),
        };
        match circuit.run_density(DensityMatrix::from_state(&initial)) {
            Ok(d) => {
                println!(
                    "noisy density matrix built (trace={:.6}, purity={:.6}).",
                    d.trace().re,
                    d.purity()
                );
                self.density = Some(d);
                self.cmd_state();
            }
            Err(e) => eprintln!("density error: {e}"),
        }
    }

    /// `qnn` — the minimal 1-parameter VariationalQnn (Ry-encode x, trainable Ry,
    /// <Z> readout), exposed interactively:
    ///
    /// - `qnn` fit cos(x+0.7) demo (Adam)
    /// - `qnn DATA [opt=...]` fit your own (x,y) pairs
    ///
    /// DATA is a flat brace list {x0,y0, x1,y1, ...} (one pair per row).
    /// Config: target=K (demo only), lr=E, steps=N, opt=adam|gd.
    fn cmd_qnn(&mut self, rest: &str) {
        let (data_src, cfg) = split_qnn_config(rest.trim());
        let data: Vec<(f64, f64)> = if data_src.is_empty() {
            let target = cfg.target;
            [-1.0, -0.5, 0.0, 0.5, 1.0]
                .into_iter()
                .map(|x| (x, (x + target).cos()))
                .collect()
        } else {
            match parse_dataset(data_src, 1) {
                Ok(d) => d.into_iter().map(|(x, y)| (x[0], y)).collect(),
                Err(e) => return eprintln!("qnn data error: {e}"),
            }
        };
        run_qnn(&self.qnn, &data, &cfg);
    }

    fn cmd_eval(&self, rest: &str) {
        if rest.is_empty() {
            return eprintln!("usage: eval EXPR");
        }
        let expr = match parse_scalar_expr(rest) {
            Ok(e) => e,
            Err(e) => return eprintln!("parse error: {e}"),
        };
        let simplified = match Simplifier::new(self.assume.clone()).simplify(&expr) {
            Ok(s) => s,
            Err(e) => return eprintln!("simplify error: {e}"),
        };
        match simplified.evaluate(&HashMap::new()) {
            Ok(v) => {
                if v.im == 0.0 {
                    println!("{simplified} = {}", v.re);
                } else {
                    println!("{simplified} = {}{:+}i", v.re, v.im);
                }
            }
            Err(e) => eprintln!("eval error: {e}"),
        }
    }

    fn cmd_simplify(&self, rest: &str) {
        if rest.is_empty() {
            return eprintln!("usage: simplify EXPR");
        }
        let expr = match parse_scalar_expr(rest) {
            Ok(e) => e,
            Err(e) => return eprintln!("parse error: {e}"),
        };
        match Simplifier::new(self.assume.clone()).simplify(&expr) {
            Ok(s) => println!("{s}"),
            Err(e) => eprintln!("simplify error: {e}"),
        }
    }

    fn cmd_diff(&self, rest: &str) {
        if rest.is_empty() {
            return eprintln!("usage: diff EXPR");
        }
        let expr = match parse_scalar_expr(rest) {
            Ok(e) => e,
            Err(e) => return eprintln!("parse error: {e}"),
        };
        match expr.differentiate("x") {
            Ok(d) => println!("d/dx ({rest}) = {d}"),
            Err(e) => eprintln!("diff error: {e}"),
        }
    }

    fn cmd_pauli(&self, rest: &str) {
        if rest.is_empty() {
            return eprintln!("usage: pauli EXPR   e.g. X, 2*Y, I+Z");
        }
        let expr = match parse_scalar_expr(rest) {
            Ok(e) => e,
            Err(e) => return eprintln!("parse error: {e}"),
        };
        let terms = match collect_pauli_terms(&expr) {
            Ok(t) => t,
            Err(msg) => return eprintln!("{msg}"),
        };
        let mut sum = quantumforge::analytic::SymbolicPauliSum::zero(1);
        for (coeff, word) in terms {
            let word = quantumforge::analytic::PauliWord::new(vec![word]);
            let coeff = ScalarExpr::exact(coeff);
            if let Err(e) = sum.add_term(coeff, word) {
                return eprintln!("error: {e}");
            }
        }
        println!("{sum}");
    }

    /// `qtm` — Quantum Turing Machine (Bernstein–Vazirani unitary formulation).
    ///   qtm                       built-in counter machine, 4 cells
    ///   qtm machine=scan          "move right, flip state" scanner
    ///   qtm cells=8               counter machine on an 8-cell tape
    ///
    /// `cells=` must be a power of two (tape is one register). Head moves wrap
    /// modulo the tape, so both built-ins are reversible by construction (a
    /// total, injective transition table). The reversibility gap is reported;
    /// `machine=erase` shows a non-injective table rejected by `step_quantum`.
    fn cmd_qtm(&self, rest: &str) {
        use quantumforge::qtm::{Configuration, Direction, QuantumTuringMachine, Transition};

        let mut cells = 4usize;
        let mut machine = "counter".to_owned();
        for word in rest.split_whitespace() {
            if let Some(v) = word
                .strip_prefix("cells=")
                .and_then(|v| v.parse::<usize>().ok())
            {
                cells = v;
            } else if let Some(v) = word.strip_prefix("machine=") {
                machine = v.to_owned();
            }
        }
        if !cells.is_power_of_two() || cells == 0 {
            return eprintln!("qtm error: cells={cells} must be a power of two");
        }

        // `counter` and `scan` are reversible: they write the read symbol back
        // unchanged (injective), move right (wraps), and flip state. `erase`
        // writes 0 regardless of the read (non-injective -> needs the ancilla
        // uncomputation shown on rejection).
        let erase = machine == "erase";
        let transitions = if erase {
            vec![
                vec![
                    Some(Transition {
                        next_state: 1,
                        write: 0,
                        direction: Direction::Right,
                    }),
                    Some(Transition {
                        next_state: 1,
                        write: 0,
                        direction: Direction::Right,
                    }),
                ],
                vec![
                    Some(Transition {
                        next_state: 0,
                        write: 0,
                        direction: Direction::Right,
                    }),
                    Some(Transition {
                        next_state: 0,
                        write: 0,
                        direction: Direction::Right,
                    }),
                ],
            ]
        } else {
            // counter / scan: preserve the read symbol (write it back).
            vec![
                vec![
                    Some(Transition {
                        next_state: 1,
                        write: 0,
                        direction: Direction::Right,
                    }),
                    Some(Transition {
                        next_state: 1,
                        write: 1,
                        direction: Direction::Right,
                    }),
                ],
                vec![
                    Some(Transition {
                        next_state: 0,
                        write: 0,
                        direction: Direction::Right,
                    }),
                    Some(Transition {
                        next_state: 0,
                        write: 1,
                        direction: Direction::Right,
                    }),
                ],
            ]
        };
        let m = match QuantumTuringMachine::new(2, 2, cells, cells, transitions, 0) {
            Ok(m) => m,
            Err(e) => return eprintln!("qtm error: {e}"),
        };
        // The internal-state register must fit the machine's state count.
        if let Err(e) = m.is_buildable() {
            return eprintln!("qtm error: {e}");
        }
        let gap = m.reversibility_gap();
        let mut tape = vec![0u8; cells];
        if cells > 1 {
            tape[1] = 1;
        }
        let head = (cells / 2).min(cells - 1);
        println!(
            "qtm: {machine} machine, {} qubits, {cells} cells, tape {tape:?} head {head}, reversibility gap {gap}",
            m.qubits()
        );
        let start = Configuration {
            tape: tape.clone(),
            head,
            state: 0,
        };
        let classical = m.run_classical(&start, 1);
        println!(
            "  classical step: state {} -> {}, head {} -> {}",
            start.state, classical.state, start.head, classical.head
        );
        let sup = match m.prepare_superposition(&tape, head) {
            Ok(s) => s,
            Err(e) => return eprintln!("qtm error: {e}"),
        };
        let before = sup.norm_sqr();
        match m.step_quantum(&sup) {
            Ok(stepped) => println!(
                "  unitary step preserved norm: {} -> {}",
                before,
                stepped.norm_sqr()
            ),
            Err(e) => {
                // Non-injective table: the plain register can't hold a unitary.
                // Run it on the ancilla-augmented register instead, which
                // preserves the erased symbol so the step is a bijection.
                println!("  step_quantum rejected the plain register: {e}");
                match m.prepare_superposition_ancilla(&tape, head, cells) {
                    Ok(asup) => {
                        let abefore = asup.norm_sqr();
                        match m.step_quantum_ancilla(&asup, cells) {
                            Ok(astepped) => {
                                let (_, aq) = m.permutation_with_ancilla(cells).unwrap_or_default();
                                println!(
                                    "  ancilla step ({aq} qubits) preserved norm: {} -> {}",
                                    abefore,
                                    astepped.norm_sqr()
                                );
                                match m.step_quantum_ancilla(&astepped, cells) {
                                    Ok(back) => println!(
                                        "  ancilla involution: second step restores norm {} (reversible)",
                                        back.norm_sqr()
                                    ),
                                    Err(e3) => eprintln!("  ancilla step 2 error: {e3}"),
                                }
                            }
                            Err(e3) => eprintln!("  ancilla step error: {e3}"),
                        }
                    }
                    Err(e2) => eprintln!("  ancilla error: {e2}"),
                }
                return;
            }
        }
        match m.run_quantum_measured(&sup, 1, 7) {
            Ok((state, _)) => println!("  measured internal state after 1 step = {state}"),
            Err(e) => eprintln!("  measurement error: {e}"),
        }
    }

    /// `mbqc` — Measurement-Based Quantum Computing. Prepares a cluster state
    /// on a graph and runs an adaptive measurement pattern with by-product
    /// corrections.
    ///
    ///   mbqc                                         3-qubit linear path demo
    ///   mbqc edges=0-1,1-2,0-2 qubits=3              custom cluster graph
    ///   mbqc qubits=3 pattern="angles=0.5,0,pi basis=eq,eq,z qubits=1,2,0 deps=none,0,1 depsz=none,none,0 outputs=0,0,1"
    ///
    /// Graph edges are `a-b` pairs; `qubits=N` sets the vertex count. A
    /// `pattern=` spec supplies explicit measurement angles, bases, per-step
    /// qubit order (`qubits=`, any permutation), by-product dependencies, and
    /// output flags (see `quantumforge::mbqc::parse_pattern`); without it, each
    /// vertex is measured with equatorial/Z bases depending on its neighbours'
    /// earlier outcomes.
    fn cmd_mbqc(&self, rest: &str) {
        use quantumforge::mbqc::{
            ClusterState, MeasurementBasis, MeasurementPattern, MeasurementStep, parse_pattern,
        };

        // Parse: edges=a-b,c-d, qubits=N, pattern="...".
        let mut edges_spec: Option<String> = None;
        let mut qubits = 3usize;
        let mut pattern_spec: Option<String> = None;
        // Pull out a quoted pattern=... spec first (it contains spaces/commas).
        let rest_owned = rest.to_owned();
        let rest_clean: String = if let Some(idx) = rest_owned.find("pattern=\"") {
            let after = &rest_owned[idx + "pattern=\"".len()..];
            if let Some(end) = after.find('"') {
                pattern_spec = Some(after[..end].to_owned());
                let mut s = rest_owned[..idx].to_owned();
                s.push_str(&after[end + 1..]);
                s
            } else {
                return eprintln!("mbqc error: unterminated pattern=\"...\" quote");
            }
        } else {
            rest_owned.clone()
        };
        for word in rest_clean.split_whitespace() {
            if let Some(v) = word.strip_prefix("qubits=") {
                if let Ok(n) = v.parse::<usize>() {
                    qubits = n;
                }
            } else if let Some(v) = word.strip_prefix("edges=") {
                edges_spec = Some(v.to_owned());
            }
        }

        let cluster = match edges_spec {
            None => ClusterState::path(qubits),
            Some(spec) => {
                let mut edges = Vec::new();
                for pair in spec.split(',') {
                    let mut it = pair.split('-');
                    match (
                        it.next().and_then(|a| a.trim().parse::<usize>().ok()),
                        it.next().and_then(|b| b.trim().parse::<usize>().ok()),
                    ) {
                        (Some(a), Some(b)) if a < qubits && b < qubits && a != b => {
                            edges.push(if a < b { (a, b) } else { (b, a) });
                        }
                        _ => {
                            return eprintln!(
                                "mbqc error: bad edge '{pair}' (need a-b with a,b < {qubits})"
                            );
                        }
                    }
                }
                ClusterState::new(qubits, edges)
            }
        };
        println!(
            "mbqc: {qubits}-qubit cluster state, {} edges {:?}",
            cluster.edges.len(),
            cluster.edges
        );

        // Build the pattern: explicit spec, or the default neighbour-dependent one.
        let pattern = match &pattern_spec {
            Some(spec) => match parse_pattern(spec, qubits) {
                Ok(p) => p,
                Err(e) => return eprintln!("mbqc error: {e}"),
            },
            None => {
                let mut p = MeasurementPattern::new();
                for v in 0..qubits {
                    let deps = cluster.neighbours(v);
                    p.push(MeasurementStep {
                        qubit: v,
                        basis: if v % 2 == 0 {
                            MeasurementBasis::Equatorial(0.6)
                        } else {
                            MeasurementBasis::Z
                        },
                        depends_on_x: deps.clone(),
                        depends_on_z: deps,
                        output: v == 0,
                    });
                }
                p
            }
        };
        match pattern.run(&cluster, 7) {
            Ok(run) => {
                println!("  raw outcomes: {:?}", run.outcomes);
                let angles: Vec<String> = run.angles.iter().map(|a| format!("{a:.4}")).collect();
                println!("  corrected angles: {}", angles.join(", "));
                println!(
                    "  output bit = {}",
                    run.outputs.first().copied().unwrap_or(0)
                );
            }
            Err(e) => eprintln!("mbqc error: {e}"),
        }
    }

    /// `anneal` — Adiabatic Quantum Computing. Interpolates a driver Hamiltonian
    /// to a problem Hamiltonian and reports the final energy vs the exact ground.
    ///
    ///   anneal                    built-in demo: -(X0+X1) -> Z0Z1, T=20
    ///   anneal NAME               named schedule (ising, tfim, heisenberg, xxz, hubbard)
    ///   anneal NAME T=40 tol=1e-6 steps=100000
    ///   anneal ising path=fast T=30        fast-start annealing path
    ///   anneal ising path=poly T=30        polynomial (p=2) path
    ///
    /// Named schedules reuse the exact-rational spin Hamiltonians from `vqe` as
    /// the problem term, with a derived transverse-field driver. `path=` selects
    /// the annealing path: `linear` (default), `fast` (Roland–Cerf fast-start),
    /// or `poly` (quadratic).
    fn cmd_anneal(&self, rest: &str) {
        use quantumforge::aqc::{AnnealPath, run_adiabatic, uniform_superposition};

        // Parse: optional leading schedule name, then key=value config. A bare
        // leading number is accepted as total_time for backward compatibility
        // (`anneal 20` == `anneal T=20`).
        let mut name = "demo";
        let mut total_time = 20.0f64;
        let mut tol = 1e-6f64;
        let mut max_steps = 50000usize;
        let mut path = AnnealPath::Linear;
        let mut driver = "tf";
        for (i, word) in rest.split_whitespace().enumerate() {
            if let Some(v) = word.strip_prefix("T=").and_then(|v| v.parse::<f64>().ok()) {
                total_time = v;
            } else if let Some(v) = word
                .strip_prefix("tol=")
                .and_then(|v| v.parse::<f64>().ok())
            {
                tol = v;
            } else if let Some(v) = word
                .strip_prefix("steps=")
                .and_then(|v| v.parse::<usize>().ok())
            {
                max_steps = v;
            } else if let Some(v) = word.strip_prefix("path=") {
                path = match v {
                    "linear" => AnnealPath::Linear,
                    "fast" | "fast-start" => AnnealPath::FastStart(0.3),
                    "poly" | "polynomial" => AnnealPath::Polynomial(2.0),
                    other => {
                        return eprintln!(
                            "anneal error: unknown path '{other}' (linear|fast|poly)"
                        );
                    }
                };
            } else if let Some(v) = word.strip_prefix("driver=") {
                driver = match v {
                    "tf" | "transverse" => "tf",
                    "xz" | "mixed" => "xz",
                    other => {
                        return eprintln!(
                            "anneal error: unknown driver '{other}' (tf|transverse, xz|mixed)"
                        );
                    }
                };
            } else if i == 0 {
                // A bare leading number is total_time; otherwise it's a schedule name.
                match word.parse::<f64>() {
                    Ok(v) if v > 0.0 => total_time = v,
                    _ => name = word,
                }
            }
        }
        if total_time <= 0.0 {
            return eprintln!("anneal error: total_time must be positive");
        }

        // Build the schedule: demo (2-qubit ZZ) or a named spin chain.
        let (schedule, label) = if name == "demo" {
            match quantumforge::aqc::demo_schedule() {
                Ok(s) => (s, "-(X0+X1) -> Z0Z1".to_owned()),
                Err(e) => return eprintln!("anneal error: {e}"),
            }
        } else {
            let n = self.qubits;
            match build_named_schedule(name, n, driver) {
                Ok(pair) => pair,
                Err(e) => return eprintln!("anneal error: {e}"),
            }
        };
        let schedule = schedule.with_path(path);
        let initial = match uniform_superposition(schedule.qubits()) {
            Ok(s) => s,
            Err(e) => return eprintln!("anneal error: {e}"),
        };
        println!(
            "anneal: {label}, path={}, total_time={total_time}, tol={tol}, steps={max_steps}",
            schedule.path().label()
        );
        match run_adiabatic(&schedule, &initial, total_time, tol, max_steps) {
            Ok(r) => {
                println!("  initial ground energy = {:.6}", r.initial_energy);
                println!("  final energy under H1 = {:.6}", r.final_energy);
                println!("  exact ground energy    = {:.6}", r.exact_ground_energy);
                let gap = (r.final_energy - r.exact_ground_energy).abs();
                let verdict = if gap < 5e-3 {
                    "✓ adiabatic (tracked ground state)"
                } else {
                    "(leaked — try longer total_time / looser tol)"
                };
                println!("  |final - exact| = {gap:.2e}  {verdict}");
            }
            Err(e) => eprintln!("  anneal error: {e}"),
        }
    }
}

/// Build a named spin-chain adiabatic schedule: a driver Hamiltonian `H₀`
/// interpolated to the exact-rational problem Hamiltonian `H₁` named by `name`
/// on `qubits` sites. `driver` selects the driver: `tf` (transverse field
/// `-ΣX`, default) or `xz` (mixed basis `-Σ(X+Z)`). Returns (schedule, label).
fn build_named_schedule(
    name: &str,
    qubits: usize,
    driver: &str,
) -> Result<(quantumforge::aqc::AdiabaticSchedule, String), String> {
    use quantumforge::aqc::AdiabaticSchedule;
    use quantumforge::quantum::{HilbertSpace, Operator};

    // Driver Hamiltonian as a dense row-major matrix.
    let h0 = match driver {
        "tf" | "transverse" => quantumforge::aqc::transverse_field_driver(qubits),
        "xz" | "mixed" => quantumforge::aqc::mixed_basis_driver(qubits),
        other => {
            return Err(format!(
                "unknown driver '{other}' (try tf|transverse, xz|mixed)"
            ));
        }
    }
    .map_err(|e| e.to_string())?;
    let space = HilbertSpace::qubits(qubits).map_err(|e| e.to_string())?;
    let entries: Vec<quantumforge::Complex64> = h0
        .iter()
        .map(|&v| quantumforge::Complex64::new(v, 0.0))
        .collect();
    let driver_op = Operator::new(space.clone(), space, entries).map_err(|e| e.to_string())?;
    // Problem: reuse the exact-rational VQE Hamiltonian.
    let problem = named_hamiltonian(name, qubits).ok_or_else(|| {
        format!(
            "unknown or ill-sized schedule '{name}' for {qubits} qubits; \
             try: ising, tfim, heisenberg, xxz, hubbard (need >= 2 qubits)"
        )
    })?;
    let schedule =
        AdiabaticSchedule::from_hamiltonians(&driver_op, &problem).map_err(|e| e.to_string())?;
    let driver_label = match driver {
        "xz" | "mixed" => "-Σ(X+Z)",
        _ => "-ΣX",
    };
    Ok((schedule, format!("{driver_label} -> {name} ({qubits}q)")))
}

/// Interpret a parsed scalar expression as a sum of single-qubit Pauli words.
/// Recognizes I, X, Y, Z as symbols; anything else is an error.
fn collect_pauli_terms(
    expr: &ScalarExpr,
) -> Result<Vec<(ExactComplex, quantumforge::analytic::Pauli)>, String> {
    use quantumforge::analytic::{Pauli, ScalarExpr as S};
    fn word_of(s: &str) -> Option<Pauli> {
        match s {
            "I" | "Identity" => Some(Pauli::I),
            "X" => Some(Pauli::X),
            "Y" => Some(Pauli::Y),
            "Z" => Some(Pauli::Z),
            _ => None,
        }
    }
    match expr {
        S::Add(terms) => {
            let mut out = Vec::new();
            for t in terms {
                out.extend(collect_pauli_terms(t)?);
            }
            Ok(out)
        }
        S::Mul(factors) => {
            // Split the product into (optional exact coefficient) x (one Pauli symbol).
            let mut coeff = ExactComplex::real(Rational::new(1, 1).unwrap());
            let mut symbol: Option<Pauli> = None;
            for f in factors {
                match f {
                    S::Exact(c) => {
                        coeff = match coeff.checked_mul(*c) {
                            Ok(v) => v,
                            Err(_) => return Err("exact multiplication overflow".to_owned()),
                        }
                    }
                    S::Symbol(name) => match word_of(name) {
                        Some(p) => symbol = Some(p),
                        None => return Err(format!("unknown Pauli word: {name}")),
                    },
                    _ => return Err("pauli expression must be I, X, Y, Z or their sum".to_owned()),
                }
            }
            let symbol = symbol.ok_or_else(|| "pauli term has no Pauli symbol".to_owned())?;
            Ok(vec![(coeff, symbol)])
        }
        S::Symbol(name) => match word_of(name) {
            Some(p) => Ok(vec![(ExactComplex::real(Rational::new(1, 1).unwrap()), p)]),
            None => Err(format!("unknown Pauli word: {name}")),
        },
        S::Exact(c) => {
            // A bare number means the identity coefficient.
            let identity = word_of("I").unwrap();
            Ok(vec![(*c, identity)])
        }
        _ => Err("unsupported pauli term; use I, X, Y, Z, numeric coeffs, + and *".to_owned()),
    }
}

/// Fixed-width layout: every gate column is CELL chars wide, offset by the
/// `qN ` wire-label prefix so the qubit label is never overwritten.
const CELL: usize = 7;
/// Char width of the `qN ` wire label (e.g. `q0 `, `q10 `); rows start here.
const PREFIX: usize = 3;

/// Write a single-char `marker` at exact char index `x`, extending with spaces
/// if the row is too short. Used for gate markers + vertical connectors.
fn put_at(row: &mut String, x: usize, marker: &str) {
    let chars: Vec<char> = row.chars().collect();
    let mut out: String = chars[..x.min(chars.len())].iter().collect();
    while out.chars().count() < x {
        out.push(' ');
    }
    out.push_str(marker);
    if chars.len() > x + marker.chars().count() {
        out.extend(&chars[x + marker.chars().count()..]);
    }
    *row = out;
}

/// Center a gate `label` within its CELL-width column (offset past the `qN `
/// prefix), overwriting the wire dashes there. Labels longer than CELL are
/// truncated so the wire keeps its column rhythm.
fn label(row: &mut String, col: usize, text: &str) {
    let text: String = text.chars().take(CELL).collect();
    let pad = (CELL - text.chars().count()) / 2;
    let x = PREFIX + col * CELL + pad;
    put_at(row, x, &text);
}

/// Draw a vertical `│` at char index `x` on every wire strictly between qubits
/// `a` and `b` (exclusive), visually linking a two-qubit gate's control to its
/// target across the wires it spans.
fn connect(rows: &mut [String], a: usize, b: usize, x: usize) {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    for row in rows.iter_mut().skip(lo + 1).take(hi.saturating_sub(lo + 1)) {
        put_at(row, x, "│");
    }
}

fn gate_label(gate: &Gate) -> String {
    match gate {
        Gate::I => "I".to_owned(),
        Gate::X => "X".to_owned(),
        Gate::Y => "Y".to_owned(),
        Gate::Z => "Z".to_owned(),
        Gate::H => "H".to_owned(),
        Gate::S => "S".to_owned(),
        Gate::Sdg => "S†".to_owned(),
        Gate::T => "T".to_owned(),
        Gate::Tdg => "T†".to_owned(),
        Gate::Rx(t) => format!("Rx{t:.2}"),
        Gate::Ry(t) => format!("Ry{t:.2}"),
        Gate::Rz(t) => format!("Rz{t:.2}"),
    }
}

fn channel_label(channel: &quantumforge::NoiseChannel) -> String {
    match channel {
        quantumforge::NoiseChannel::Depolarizing { probability } => format!("Dep{probability:.2}"),
        quantumforge::NoiseChannel::AmplitudeDamping { gamma } => format!("AD{gamma:.2}"),
        quantumforge::NoiseChannel::PhaseDamping { gamma } => format!("PD{gamma:.2}"),
    }
}

/// Optional hyperparameters for `train` (lr=, epochs=). Defaults match the demos.
#[derive(Clone, Copy)]
struct TrainConfig {
    lr: f64,
    epochs: usize,
    nfeat: usize,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.1,
            epochs: 200,
            nfeat: 1,
        }
    }
}

/// Split a `train` tail into the DATA expression and any trailing `key=value`
/// config tokens (lr=, epochs=, nfeat=). DATA is everything up to the first
/// such token.
fn split_config(tail: &str) -> (&str, TrainConfig) {
    let mut config = TrainConfig::default();
    let mut data_end = tail.len();
    for (i, _) in tail.match_indices(char::is_whitespace) {
        let word = tail[i..].split_whitespace().next().unwrap_or("");
        if let Some(v) = word.strip_prefix("lr=").and_then(|v| v.parse::<f64>().ok()) {
            config.lr = v;
            data_end = data_end.min(i);
        } else if let Some(v) = word
            .strip_prefix("epochs=")
            .and_then(|v| v.parse::<usize>().ok())
        {
            config.epochs = v;
            data_end = data_end.min(i);
        } else if let Some(v) = word
            .strip_prefix("nfeat=")
            .and_then(|v| v.parse::<usize>().ok())
        {
            config.nfeat = v;
            data_end = data_end.min(i);
        }
    }
    (tail[..data_end].trim(), config)
}

/// Parse a flat, brace-delimited numeric dataset `{a, b, c, ...}` into rows of
/// `nfeat` features + 1 target each (so row length is nfeat+1). Numbers may be
/// decimals, negatives, or kernel expressions like `Pi/2` (evaluated via the
/// scalar parser). Every row must be complete.
fn parse_dataset(src: &str, nfeat: usize) -> Result<Vec<(Vec<f64>, f64)>, String> {
    let inner = src
        .trim()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| "DATA must be a brace list like {1,2,3,4}".to_owned())?;
    let mut values = Vec::new();
    for tok in inner.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let expr = parse_scalar_expr(tok).map_err(|e| format!("'{tok}': {e}"))?;
        let v = expr
            .evaluate(&HashMap::new())
            .map_err(|e| format!("'{tok}': {e}"))?;
        if v.im.abs() > 1e-12 {
            return Err(format!("'{tok}' is not real"));
        }
        values.push(v.re);
    }
    let row_len = nfeat + 1;
    if values.is_empty() {
        return Err("dataset is empty".to_owned());
    }
    if values.len() % row_len != 0 {
        return Err(format!(
            "{} values is not a multiple of row length {} (nfeat={}+1 target)",
            values.len(),
            row_len,
            nfeat
        ));
    }
    Ok(values
        .chunks(row_len)
        .map(|row| {
            let (target, features) = row.split_last().unwrap();
            (features.to_vec(), *target)
        })
        .collect())
}

/// Regression core: angle-encode each feature, one trainable Ry weight, <Z>
/// readout; Adam-optimize MSE. Works for any feature count by summing encoded
/// rotations (feature 0 encodes x0, feature i>0 adds Ry(x_i) — a simple but
/// valid linear-ish angle encoding).
fn run_regression(data: &[(Vec<f64>, f64)], label: &str, config: &TrainConfig) {
    let n_features = data[0].0.len();
    if data.iter().any(|(x, _)| x.len() != n_features) {
        return eprintln!("inconsistent feature counts in dataset");
    }
    if !(1..=8).contains(&n_features) {
        return eprintln!("feature count {n_features} out of range 1..=8");
    }
    let mut model = ParameterizedCircuit::with_inputs(n_features, 1);
    for f in 0..n_features {
        model.encode_rotation(RotationAxis::Y, f, f);
    }
    model.rotation(RotationAxis::Y, 0, 0);

    let init = vec![0.0];
    let qml = QmlTrainingConfig {
        learning_rate: config.lr,
        epochs: config.epochs,
        batch_size: data.len(),
        seed: 42,
    };
    let (loss0, _) =
        match model.mse_loss_and_gradient(data, &init, n_features, SingleQubitObservable::Z, 0) {
            Ok(v) => v,
            Err(e) => return eprintln!("train error: {e}"),
        };
    let weights = match model.train_regression_adam(
        data,
        &init,
        n_features,
        SingleQubitObservable::Z,
        0,
        qml,
    ) {
        Ok(w) => w,
        Err(e) => return eprintln!("train error: {e}"),
    };
    let (loss1, _) = match model.mse_loss_and_gradient(
        data,
        &weights,
        n_features,
        SingleQubitObservable::Z,
        0,
    ) {
        Ok(v) => v,
        Err(e) => return eprintln!("train error: {e}"),
    };
    let mean_y: f64 = data.iter().map(|(_, y)| y).sum::<f64>() / data.len() as f64;
    let ss_tot: f64 = data.iter().map(|(_, y)| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = loss1 * data.len() as f64;
    let r2 = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        f64::NAN
    };
    println!(
        "train regression ({label}): {n_features} feature(s), {} samples, 1 trainable weight",
        data.len()
    );
    println!(
        "  mse: {loss0:.6} -> {loss1:.6}   R² = {r2:.4}   weight = {:.6}",
        weights[0]
    );
    if r2 < 0.0 {
        println!(
            "  note: R²<0 — the <Z>∈[-1,1] readout can't fit these targets; try bounded targets"
        );
    }
}

/// Config for the interactive `qnn` command (target=, lr=, steps=, opt=).
#[derive(Clone, Copy)]
struct QnnConfig {
    target: f64,
    lr: f64,
    steps: usize,
    adam: bool,
}
impl Default for QnnConfig {
    fn default() -> Self {
        Self {
            target: 0.7,
            lr: 0.05,
            steps: 100,
            adam: true,
        }
    }
}

/// Split a `qnn` tail into the DATA expression and trailing `key=value` config.
fn split_qnn_config(tail: &str) -> (&str, QnnConfig) {
    let mut cfg = QnnConfig::default();
    let mut data_end = tail.len();
    for (i, _) in tail.match_indices(char::is_whitespace) {
        let word = tail[i..].split_whitespace().next().unwrap_or("");
        if let Some(v) = word
            .strip_prefix("target=")
            .and_then(|v| v.parse::<f64>().ok())
        {
            cfg.target = v;
            data_end = data_end.min(i);
        } else if let Some(v) = word.strip_prefix("lr=").and_then(|v| v.parse::<f64>().ok()) {
            cfg.lr = v;
            data_end = data_end.min(i);
        } else if let Some(v) = word
            .strip_prefix("steps=")
            .and_then(|v| v.parse::<usize>().ok())
        {
            cfg.steps = v;
            data_end = data_end.min(i);
        } else if word == "opt=adam" {
            cfg.adam = true;
            data_end = data_end.min(i);
        } else if word == "opt=gd" {
            cfg.adam = false;
            data_end = data_end.min(i);
        }
    }
    (tail[..data_end].trim(), cfg)
}

/// Fit the 1-parameter VariationalQnn to `data` with the chosen optimizer and
/// report initial/final MSE and the learned weight.
fn run_qnn(qnn: &VariationalQnn, data: &[(f64, f64)], cfg: &QnnConfig) {
    if data.is_empty() {
        return eprintln!("qnn: empty dataset");
    }
    let init = -1.2;
    let (l0, _) = match qnn.loss_and_gradient(data, init) {
        Ok(v) => v,
        Err(e) => return eprintln!("qnn error: {e}"),
    };
    let trained = if cfg.adam {
        qnn.train_adam(data, init, cfg.lr, cfg.steps, 2, 42)
    } else {
        qnn.train(data, init, cfg.lr, cfg.steps)
    };
    let trained = match trained {
        Ok(w) => w,
        Err(e) => return eprintln!("qnn error: {e}"),
    };
    let (l1, _) = match qnn.loss_and_gradient(data, trained) {
        Ok(v) => v,
        Err(e) => return eprintln!("qnn error: {e}"),
    };
    let opt = if cfg.adam { "Adam" } else { "GD" };
    println!(
        "qnn (1-param Ry-encode + trainable Ry, <Z> readout, {opt}, {} samples):",
        data.len()
    );
    println!(
        "  mse: {l0:.6} -> {l1:.6}   weight = {trained:.6}  (lr={}, steps={})",
        cfg.lr, cfg.steps
    );
}

/// Classification core: same ansatz, full-batch Adam on binary cross-entropy.
fn run_classification(data: &[(Vec<f64>, f64)], label: &str, config: &TrainConfig) {
    let n_features = data[0].0.len();
    if data.iter().any(|(x, _)| x.len() != n_features) {
        return eprintln!("inconsistent feature counts in dataset");
    }
    if data.iter().any(|(_, y)| !matches!(*y, 0.0 | 1.0)) {
        return eprintln!("classify labels must be 0 or 1");
    }
    if !(1..=8).contains(&n_features) {
        return eprintln!("feature count {n_features} out of range 1..=8");
    }
    let mut model = ParameterizedCircuit::with_inputs(n_features, 1);
    for f in 0..n_features {
        model.encode_rotation(RotationAxis::Y, f, f);
    }
    model.rotation(RotationAxis::Y, 0, 0);

    let mut weights = vec![0.0];
    let (lr, beta1, beta2, eps) = (config.lr, 0.9, 0.999, 1e-8);
    let mut m = vec![0.0; weights.len()];
    let mut v = vec![0.0; weights.len()];
    let loss_grad = |w: &[f64]| {
        model
            .binary_cross_entropy_and_gradient(data, w, n_features, SingleQubitObservable::Z, 0)
            .map(|(l, g)| {
                (
                    l / data.len() as f64,
                    g.iter().map(|x| x / data.len() as f64).collect::<Vec<_>>(),
                )
            })
    };
    let (loss0, _) = match loss_grad(&weights) {
        Ok(v) => v,
        Err(e) => return eprintln!("train error: {e}"),
    };
    for t in 1..=config.epochs {
        let (_, grad) = match loss_grad(&weights) {
            Ok(v) => v,
            Err(e) => return eprintln!("train error: {e}"),
        };
        let t = t as f64;
        for i in 0..weights.len() {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
            let m_hat = m[i] / (1.0 - beta1.powf(t));
            let v_hat = v[i] / (1.0 - beta2.powf(t));
            weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
    let (loss1, _) = match loss_grad(&weights) {
        Ok(v) => v,
        Err(e) => return eprintln!("train error: {e}"),
    };
    let mut correct = 0;
    for (x, y) in data {
        let z = model
            .expectation_with_inputs(x, &weights, n_features, SingleQubitObservable::Z, 0)
            .unwrap_or(f64::NAN);
        let pred = if z < 0.0 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.5 {
            correct += 1;
        }
    }
    let acc = correct as f64 / data.len() as f64;
    println!(
        "train classify ({label}): {n_features} feature(s), {} samples",
        data.len()
    );
    println!(
        "  cross-entropy: {loss0:.6} -> {loss1:.6}   accuracy = {:.0}% ({correct}/{})   weight = {:.6}",
        acc * 100.0,
        data.len(),
        weights[0]
    );
}

/// Built-in VQE Hamiltonians with exact rational coefficients (the exact-arithmetic
/// PauliSum kernel's sweet spot). Standard spin-model benchmarks on `qubits` sites:
///
/// - tfim: transverse-field Ising, H = -Σ Z_i Z_{i+1} - Σ X_i
/// - ising: pure Ising, H = -Σ Z_i Z_{i+1}
/// - heisenberg: XXX Heisenberg chain, H = Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
/// - xxz: anisotropic XXZ ring, H = Σ (X_iX_{i+1} + Y_iY_{i+1} - ½ Z_iZ_{i+1})
/// - hubbard: Hubbard ring (t=1, U=1) mapped to spins via Jordan-Wigner,
///   H = Σ (c†ᵢc_{i+1} + h.c.) + U Σ n_i n_{i+1} = -½Σ(X_iX_{i+1}+Y_iY_{i+1}) + ¼Σ(I_iI_{i+1} - Z_iZ_{i+1})
///
/// Returns None for unknown names or qubit counts < 2 (no bonds to form).
fn named_hamiltonian(name: &str, qubits: usize) -> Option<quantumforge::analytic::PauliSum> {
    use quantumforge::analytic::{ExactComplex, Pauli, PauliSum, PauliWord, Rational};

    fn c(num: i128, den: i128) -> ExactComplex {
        ExactComplex::real(Rational::new(num, den).expect("nonzero rational"))
    }
    fn add(sum: &mut PauliSum, num: i128, den: i128, paulis: &[(usize, Pauli)], qubits: usize) {
        let mut f = vec![Pauli::I; qubits];
        for &(q, p) in paulis {
            f[q] = p;
        }
        let _ = sum.add_term(c(num, den), PauliWord::new(f));
    }

    if qubits < 2 {
        return None;
    }
    let mut sum = PauliSum::zero(qubits);
    match name {
        "tfim" => {
            for i in 0..qubits - 1 {
                add(&mut sum, -1, 1, &[(i, Pauli::Z), (i + 1, Pauli::Z)], qubits);
            }
            for i in 0..qubits {
                add(&mut sum, -1, 1, &[(i, Pauli::X)], qubits);
            }
        }
        "ising" => {
            for i in 0..qubits - 1 {
                add(&mut sum, -1, 1, &[(i, Pauli::Z), (i + 1, Pauli::Z)], qubits);
            }
        }
        "heisenberg" => {
            for i in 0..qubits - 1 {
                for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                    add(&mut sum, 1, 1, &[(i, p), (i + 1, p)], qubits);
                }
            }
        }
        "xxz" => {
            // Anisotropic XXZ ring: Σ (XX + YY - ½ ZZ).
            for i in 0..qubits - 1 {
                add(&mut sum, 1, 1, &[(i, Pauli::X), (i + 1, Pauli::X)], qubits);
                add(&mut sum, 1, 1, &[(i, Pauli::Y), (i + 1, Pauli::Y)], qubits);
                add(&mut sum, -1, 2, &[(i, Pauli::Z), (i + 1, Pauli::Z)], qubits);
            }
        }
        "hubbard" => {
            // Hubbard ring (t=1, U=1), Jordan-Wigner to spins:
            //   hopping  -½(XX+YY) per bond;  interaction  ¼(II - ZZ) per bond.
            for i in 0..qubits - 1 {
                add(&mut sum, -1, 2, &[(i, Pauli::X), (i + 1, Pauli::X)], qubits);
                add(&mut sum, -1, 2, &[(i, Pauli::Y), (i + 1, Pauli::Y)], qubits);
                add(&mut sum, 1, 4, &[(i, Pauli::I), (i + 1, Pauli::I)], qubits);
                add(&mut sum, -1, 4, &[(i, Pauli::Z), (i + 1, Pauli::Z)], qubits);
            }
        }
        _ => return None,
    }
    Some(sum)
}
