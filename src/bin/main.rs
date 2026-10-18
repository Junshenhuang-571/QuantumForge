use quantumforge::analytic::{
    Assumptions, ExactComplex, ExprPattern, Pauli, PauliSum, PauliWord, Rational, RewriteRule,
    ScalarExpr, Simplifier, SymbolicPauliSum,
};
use quantumforge::parser::parse_circuit;
use quantumforge::parser::parse_scalar_expr;
use quantumforge::quantum::{HilbertSpace, Ket, Observable, Operator};
use quantumforge::{
    Circuit, DensityMatrix, Gate, NoiseChannel, ParameterizedCircuit, QmlTrainingConfig,
    RotationAxis, SmallRng, StateVector, VariationalQnn,
};
use std::collections::HashMap;
#[cfg(feature = "llvm")]
use std::collections::{BTreeMap, BTreeSet};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str).unwrap_or("demo") {
        "demo" => run_demo(),
        "eval" => run_eval(&args[1..].join(" ")),
        "simplify" => run_simplify(&args[1..].join(" ")),
        "shots" => run_shots(&args[1..]),
        #[cfg(feature = "llvm")]
        "llvm-ir" => run_llvm_ir(&args[1..].join(" ")),
        #[cfg(feature = "llvm")]
        "jit-eval" => run_jit_eval(&args[1..]),
        #[cfg(feature = "cuda")]
        "cuda-demo" => run_cuda_demo(),
        "help" | "--help" | "-h" => print_help(),
        command => {
            eprintln!("unknown command: {command}");
            print_help();
            std::process::exit(2);
        }
    }
}

fn run_shots(args: &[String]) {
    let result = (|| {
        if args.len() < 4 {
            return Err("usage: quantumforge shots QUBITS SHOTS SEED 'Circuit[...]'".to_owned());
        }
        let qubits = args[0]
            .parse::<usize>()
            .map_err(|_| "QUBITS must be a nonnegative integer".to_owned())?;
        let shots = args[1]
            .parse::<usize>()
            .map_err(|_| "SHOTS must be a nonnegative integer".to_owned())?;
        let seed = args[2]
            .parse::<u64>()
            .map_err(|_| "SEED must be a nonnegative integer".to_owned())?;
        let circuit = parse_circuit(&args[3..].join(" ")).map_err(|error| error.to_string())?;
        let initial = StateVector::zero(qubits).map_err(|error| error.to_string())?;
        let result = circuit
            .run_shots(&initial, shots, seed)
            .map_err(|error| error.to_string())?;
        Ok::<_, String>(result)
    })();
    match result {
        Ok(result) => {
            println!("shots: {}", result.shots);
            for (bits, count) in result.counts {
                let bit_string: String = bits
                    .iter()
                    .rev()
                    .map(|bit| char::from(b'0' + *bit))
                    .collect();
                println!("{bit_string}: {count}");
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

fn run_simplify(source: &str) {
    if let Err(error) = (|| {
        let expression = parse_scalar_expr(source).map_err(|error| error.to_string())?;
        let simplified = Simplifier::default()
            .simplify(&expression)
            .map_err(|error| error.to_string())?;
        println!("{simplified}");
        Ok::<(), String>(())
    })() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run_eval(source: &str) {
    if let Err(error) = (|| {
        let expression = parse_scalar_expr(source).map_err(|error| error.to_string())?;
        let simplified = Simplifier::default()
            .simplify(&expression)
            .map_err(|error| error.to_string())?;
        let value = simplified
            .evaluate(&HashMap::new())
            .map_err(|error| error.to_string())?;
        if value.im == 0.0 {
            println!("{simplified} = {}", value.re);
        } else {
            println!("{simplified} = {}{:+}i", value.re, value.im);
        }
        Ok::<(), String>(())
    })() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "llvm")]
fn run_llvm_ir(source: &str) {
    let result = (|| {
        let expression = parse_scalar_expr(source).map_err(|error| error.to_string())?;
        quantumforge::llvm_backend::lower_scalar_to_llvm_ir(&expression, &[])
            .map_err(|error| error.to_string())
    })();
    match result {
        Ok(ir) => print!("{ir}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "llvm")]
fn run_jit_eval(args: &[String]) {
    let result = (|| {
        if args.len() < 2 {
            return Err(
                "usage: quantumforge jit-eval OPT_LEVEL 'EXPR' [SYMBOL=VALUE ...]".to_owned(),
            );
        }
        let level = args[0]
            .parse::<u8>()
            .map_err(|_| "OPT_LEVEL must be an integer from 0 to 3".to_owned())?;
        if level > 3 {
            return Err("OPT_LEVEL must be an integer from 0 to 3".to_owned());
        }
        let expression =
            parse_scalar_expr(&args[1..2].join(" ")).map_err(|error| error.to_string())?;
        let mut symbols = BTreeSet::new();
        collect_scalar_symbols(&expression, &mut symbols);
        let mut bindings = BTreeMap::new();
        for binding in &args[2..] {
            let (name, value) = binding
                .split_once('=')
                .ok_or_else(|| format!("expected SYMBOL=VALUE, got '{binding}'"))?;
            if name.trim().is_empty() {
                return Err(format!("symbol name is empty in binding '{binding}'"));
            }
            let value = value
                .parse::<f64>()
                .map_err(|_| format!("invalid real value in binding '{binding}'"))?;
            if !value.is_finite() {
                return Err(format!("binding '{binding}' must be finite"));
            }
            if bindings.insert(name.to_owned(), value).is_some() {
                return Err(format!("symbol '{name}' was bound more than once"));
            }
        }
        for name in &symbols {
            if !bindings.contains_key(name) {
                return Err(format!("missing numeric value for symbol '{name}'"));
            }
        }
        if let Some(name) = bindings.keys().find(|name| !symbols.contains(*name)) {
            return Err(format!("symbol '{name}' does not occur in the expression"));
        }

        let symbol_names: Vec<&str> = symbols.iter().map(String::as_str).collect();
        let values: Vec<f64> = symbols.iter().map(|name| bindings[name]).collect();
        let program =
            quantumforge::numeric_ir::NumericProgram::compile_scalar(&expression, &symbol_names)
                .map_err(|error| error.to_string())?;
        let compiled = quantumforge::llvm_backend::compile_jit(&program, level)
            .map_err(|error| error.to_string())?;
        let value = compiled
            .evaluate(&values)
            .map_err(|error| error.to_string())?;
        Ok::<_, String>((expression, value, level))
    })();
    match result {
        Ok((expression, value, level)) => println!("{expression} = {value}  [LLVM -O{level}]"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(feature = "llvm")]
fn collect_scalar_symbols(expression: &ScalarExpr, symbols: &mut BTreeSet<String>) {
    match expression {
        ScalarExpr::Symbol(name) => {
            symbols.insert(name.clone());
        }
        ScalarExpr::Add(terms) | ScalarExpr::Mul(terms) => {
            for term in terms {
                collect_scalar_symbols(term, symbols);
            }
        }
        ScalarExpr::Pow(base, _) => collect_scalar_symbols(base, symbols),
        ScalarExpr::Abs(value)
        | ScalarExpr::Sqrt(value)
        | ScalarExpr::Sin(value)
        | ScalarExpr::Cos(value)
        | ScalarExpr::Exp(value)
        | ScalarExpr::Log(value) => collect_scalar_symbols(value, symbols),
        ScalarExpr::Exact(_) | ScalarExpr::Constant(_) => {}
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_demo() {
    use quantumforge::cuda::CudaBackend;
    let mut gpu_state = StateVector::zero(4).expect("state");
    gpu_state.apply(Gate::H, 0).expect("CPU preparation");
    let mut cpu_state = gpu_state.clone();
    let backend = CudaBackend::new(0).expect("initialize CUDA backend");
    backend
        .apply_ry(&mut gpu_state, 2, 0.73)
        .expect("execute CUDA Ry kernel");
    cpu_state.apply(Gate::Ry(0.73), 2).expect("CPU reference");
    let max_error = gpu_state
        .amplitudes()
        .iter()
        .zip(cpu_state.amplitudes())
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0_f64, f64::max);
    println!(
        "CUDA device {} Ry kernel; max amplitude error vs CPU = {:.3e}",
        backend.device_ordinal(),
        max_error
    );
}

fn run_demo() {
    println!("QuantumForge 0.1 — CPU quantum kernel and QNN demo");

    let x = ScalarExpr::symbol("x");
    let scalar = ScalarExpr::add([
        x.clone(),
        x,
        ScalarExpr::exact(ExactComplex::real(
            Rational::new(2, 1).expect("valid rational"),
        )),
        ScalarExpr::exact(ExactComplex::real(
            Rational::new(3, 1).expect("valid rational"),
        )),
    ])
    .expect("exact simplification");
    println!("symbolic simplification: x + x + 2 + 3 = {scalar}");

    let mut numeric_builder = quantumforge::numeric_ir::NumericBuilder::new(1);
    let numeric_x = numeric_builder.input(0).expect("valid input");
    let numeric_square = numeric_builder
        .powi(numeric_x, 2)
        .expect("valid numeric register");
    let numeric_output = numeric_builder
        .sin(numeric_square)
        .expect("valid numeric register");
    let numeric_program = numeric_builder
        .finish(numeric_output)
        .expect("valid numeric program");
    println!(
        "typed numeric IR reference: Sin[x^2] at x=0.5 -> {:.6}",
        numeric_program
            .evaluate(&[0.5])
            .expect("numeric evaluation")
    );

    let x_expr = ScalarExpr::symbol("x");
    let sqrt_square = ScalarExpr::sqrt(ScalarExpr::pow(x_expr.clone(), 2).expect("integer power"));
    let mut real_assumptions = Assumptions::default();
    real_assumptions.set_real("x");
    let real_simplifier = Simplifier::new(real_assumptions);
    println!(
        "assuming x is real: Sqrt[x^2] = {}",
        real_simplifier.simplify(&sqrt_square).expect("simplify")
    );
    let mut positive_assumptions = Assumptions::default();
    positive_assumptions.set_positive("x");
    let positive_simplifier = Simplifier::new(positive_assumptions);
    println!(
        "assuming x is positive: Sqrt[x^2] = {}",
        positive_simplifier
            .simplify(&sqrt_square)
            .expect("simplify")
    );

    let nested_abs = ScalarExpr::abs(ScalarExpr::abs(ScalarExpr::symbol("y")));
    let mut rule_simplifier = Simplifier::new(Assumptions::default());
    rule_simplifier.add_rule(RewriteRule::new(
        ExprPattern::Abs(Box::new(ExprPattern::Abs(Box::new(ExprPattern::Any(
            "u".into(),
        ))))),
        ScalarExpr::abs(ScalarExpr::symbol("u")),
    ));
    println!(
        "custom rewrite Abs[Abs[y]] = {}",
        rule_simplifier.simplify(&nested_abs).expect("rewrite")
    );

    let one_qubit = HilbertSpace::qubits(1).expect("one-qubit space");
    let zero_ket = Ket::basis(one_qubit, 0).expect("basis ket");
    let h_ket = Operator::hadamard()
        .apply(&zero_ket)
        .expect("Hadamard application");
    let z_observable = Observable::new(Operator::pauli_z(), 1e-12).expect("Hermitian observable");
    println!(
        "dimension-aware linear algebra: <H0|Z|H0> = {:.6}",
        h_ket.expectation(&z_observable).expect("expectation").re
    );

    let x =
        PauliSum::term(ExactComplex::ONE, PauliWord::new(vec![Pauli::X])).expect("one-qubit Pauli");
    let y =
        PauliSum::term(ExactComplex::ONE, PauliWord::new(vec![Pauli::Y])).expect("one-qubit Pauli");
    println!(
        "exact Pauli commutator [X, Y] = {}",
        x.commutator(&y).expect("exact arithmetic")
    );
    let ax = SymbolicPauliSum::term(ScalarExpr::symbol("a"), PauliWord::new(vec![Pauli::X]))
        .expect("symbolic Pauli");
    let by = SymbolicPauliSum::term(ScalarExpr::symbol("b"), PauliWord::new(vec![Pauli::Y]))
        .expect("symbolic Pauli");
    let symbolic_commutator = ax.commutator(&by).expect("symbolic algebra");
    println!("symbolic Pauli commutator [a X, b Y] = {symbolic_commutator}");
    let substitutions = HashMap::from([
        (
            "a".to_string(),
            ScalarExpr::exact(ExactComplex::real(
                Rational::new(2, 1).expect("valid rational"),
            )),
        ),
        (
            "b".to_string(),
            ScalarExpr::exact(ExactComplex::real(
                Rational::new(3, 1).expect("valid rational"),
            )),
        ),
    ]);
    println!(
        "after exact substitution a=2, b=3: {}",
        symbolic_commutator
            .substitute(&substitutions)
            .expect("substitution")
    );

    let mut state = StateVector::zero(1).expect("one-qubit state");
    state.apply(Gate::H, 0).expect("valid qubit");
    println!("Hadamard |0>: probabilities = {:?}", state.probabilities());
    println!(
        "Hadamard |0>: <Z> = {:.6}",
        state.expectation_z(0).expect("valid qubit")
    );

    let mut bell = StateVector::zero(2).expect("two-qubit state");
    let mut bell_circuit = quantumforge::Circuit::new();
    bell_circuit.push(Gate::H, 0);
    bell_circuit.cnot(0, 1);
    bell = bell_circuit.run(bell).expect("valid Bell circuit");
    println!("Bell state probabilities = {:?}", bell.probabilities());
    let reduced = DensityMatrix::from_state(&bell)
        .partial_trace(&[0])
        .expect("trace out qubit 1");
    println!(
        "Bell reduced state: trace = {:.6}, purity = {:.6}",
        reduced.trace().re,
        reduced.purity()
    );
    println!(
        "Bell reduced-state <Z> = {:.6}",
        z_observable
            .expectation_density(&reduced)
            .expect("density expectation")
            .re
    );

    let mut noisy_bell = DensityMatrix::from_state(&bell);
    noisy_bell
        .depolarize(0, 0.1)
        .expect("valid noise probability");
    println!(
        "after depolarizing noise: trace = {:.6}, purity = {:.6}",
        noisy_bell.trace().re,
        noisy_bell.purity()
    );
    let mut noisy_circuit = Circuit::new();
    noisy_circuit.push(Gate::X, 0);
    noisy_circuit.channel(0, NoiseChannel::AmplitudeDamping { gamma: 0.2 });
    noisy_circuit.cnot(0, 1);
    let noisy_pair = noisy_circuit
        .run_density(DensityMatrix::from_state(
            &StateVector::zero(2).expect("two-qubit state"),
        ))
        .expect("density circuit");
    println!(
        "noisy circuit result: trace = {:.6}, purity = {:.6}",
        noisy_pair.trace().re,
        noisy_pair.purity()
    );
    let mut excited = StateVector::zero(1).expect("one-qubit state");
    excited.apply(Gate::X, 0).expect("valid qubit");
    let mut damped = DensityMatrix::from_state(&excited);
    damped
        .apply_channel(NoiseChannel::AmplitudeDamping { gamma: 0.25 }, 0)
        .expect("valid damping parameter");
    println!(
        "amplitude damping: P(1) = {:.6}",
        damped.probability_one(0).expect("valid qubit")
    );
    let mut rng = SmallRng::seeded(42);
    println!(
        "density-matrix measurement = {}",
        damped.measure(0, &mut rng).expect("valid measurement")
    );

    let mut measured = bell.clone();
    let first = measured.measure(0, &mut rng).expect("valid qubit");
    let second = measured.measure(1, &mut rng).expect("valid qubit");
    println!("Bell measurement (seed 42) = |{first}{second}> (correlated bits)");

    let mut shot_circuit = Circuit::new();
    shot_circuit.push(Gate::H, 0);
    shot_circuit.cnot(0, 1);
    shot_circuit.measure(0, 0);
    shot_circuit.measure(1, 1);
    let shot_result = shot_circuit
        .run_shots(&StateVector::zero(2).expect("two-qubit state"), 256, 42)
        .expect("seeded circuit shots");
    println!(
        "seeded Bell circuit counts (classical bit 0 first) = {:?}",
        shot_result.counts
    );

    let qnn = VariationalQnn;
    let target_weight: f64 = 0.7;
    let data: Vec<(f64, f64)> = [-1.0, -0.5, 0.0, 0.5, 1.0]
        .into_iter()
        .map(|x| (x, (x + target_weight).cos()))
        .collect();
    let initial_weight = -1.2;
    let (initial_loss, _) = qnn
        .loss_and_gradient(&data, initial_weight)
        .expect("valid data");
    let learned_weight = qnn
        .train_adam(&data, initial_weight, 0.05, 100, 2, 42)
        .expect("training succeeds");
    let (final_loss, _) = qnn
        .loss_and_gradient(&data, learned_weight)
        .expect("valid data");
    println!("QNN target parameter = {target_weight:.4}");
    println!("QNN trained parameter = {learned_weight:.4}");
    println!("QNN Adam mini-batch loss: {initial_loss:.6} -> {final_loss:.6}");

    let mut variational = ParameterizedCircuit::new(2);
    variational.rotation(RotationAxis::Y, 0, 0);
    variational.rotation(RotationAxis::Y, 0, 1);
    let params = [0.3, -0.2];
    let output = variational
        .expectation_z(&params, 1, 0)
        .expect("valid variational circuit");
    let gradient = variational
        .parameter_shift_gradient(&params, 1, 0)
        .expect("valid parameter shift");
    println!("two-parameter circuit: <Z> = {output:.6}, parameter-shift gradient = {gradient:?}");
    let sampled_output = variational
        .expectation_shots(
            &params,
            1,
            quantumforge::SingleQubitObservable::Z,
            0,
            quantumforge::ShotConfig {
                shots: 1024,
                seed: 42,
            },
        )
        .expect("finite-shot QNN output");
    let sampled_gradient = variational
        .parameter_shift_gradient_shots(
            &params,
            1,
            quantumforge::SingleQubitObservable::Z,
            0,
            quantumforge::ShotConfig {
                shots: 1024,
                seed: 42,
            },
        )
        .expect("finite-shot parameter-shift gradient");
    println!("finite-shot QNN: <Z> = {sampled_output:.6}, gradient = {sampled_gradient:?}");

    let mut qnn_model = ParameterizedCircuit::with_inputs(1, 1);
    qnn_model.encode_rotation(RotationAxis::Y, 0, 0);
    qnn_model.rotation(RotationAxis::Y, 0, 0);
    let qnn_data: Vec<(Vec<f64>, f64)> = data.iter().map(|(x, y)| (vec![*x], *y)).collect();
    let (general_initial_loss, _) = qnn_model
        .mse_loss_and_gradient(
            &qnn_data,
            &[initial_weight],
            1,
            quantumforge::SingleQubitObservable::Z,
            0,
        )
        .expect("QNN batch");
    let trained_weights = qnn_model
        .train_regression_adam(
            &qnn_data,
            &[initial_weight],
            1,
            quantumforge::SingleQubitObservable::Z,
            0,
            QmlTrainingConfig {
                learning_rate: 0.05,
                epochs: 100,
                batch_size: 2,
                seed: 42,
            },
        )
        .expect("QNN model training");
    let (general_final_loss, _) = qnn_model
        .mse_loss_and_gradient(
            &qnn_data,
            &trained_weights,
            1,
            quantumforge::SingleQubitObservable::Z,
            0,
        )
        .expect("QNN batch");
    println!(
        "data-encoded QNN Adam loss: {general_initial_loss:.6} -> {general_final_loss:.6}; weights = {trained_weights:?}"
    );
}

fn print_help() {
    println!("QuantumForge — quantum computation kernel prototype");
    println!("\nUsage: quantumforge [demo|eval EXPR|simplify EXPR|help]");
    println!("       quantumforge shots QUBITS SHOTS SEED 'Circuit[...]'");
    #[cfg(feature = "llvm")]
    println!("       quantumforge llvm-ir EXPR");
    #[cfg(feature = "llvm")]
    println!("       quantumforge jit-eval OPT_LEVEL 'EXPR' [SYMBOL=VALUE ...]");
    #[cfg(feature = "cuda")]
    println!("       quantumforge cuda-demo");
    println!(
        "\nThe demo runs a one-qubit CPU simulation, seeded measurement, and a tiny variational QNN training loop."
    );
}
