use quantumforge::analytic::{
    Assumptions, ExactComplex, ExprPattern, Pauli, PauliSum, PauliWord, Rational, RewriteRule,
    ScalarExpr, Simplifier, SymbolicPauliSum,
};
use quantumforge::optimizer::{
    CircuitCostProfile, CouplingMap, OptimizeConfig, optimize_circuit, try_route_to_hardware,
    verify_unitary_equivalent,
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
        "optimize" => run_optimize(&args[1..]),
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

struct DeviceTarget {
    coupling: CouplingMap,
    native_gates: Option<std::collections::HashSet<String>>,
    directed_cx_edges: Option<std::collections::HashSet<(usize, usize)>>,
}

fn run_optimize(args: &[String]) {
    let result = (|| {
        if args.len() < 3 {
            return Err(
                "usage: quantumforge optimize QUBITS all|conservative [--line | --topology FILE] [--verify] [--input FILE] [--format json|qasm] [--report FILE] [Circuit[...]]"
                    .to_owned(),
            );
        }
        let qubits = args[0]
            .parse::<usize>()
            .map_err(|_| "QUBITS must be a positive integer".to_owned())?;
        if qubits == 0 {
            return Err("QUBITS must be a positive integer".to_owned());
        }
        let mode = args[1].as_str();
        let config = match mode {
            "all" => OptimizeConfig::all(),
            "conservative" => OptimizeConfig::conservative(),
            _ => return Err("mode must be 'all' or 'conservative'".to_owned()),
        };
        let mut use_line_topology = false;
        let mut verify = false;
        let mut topology_path = None;
        let mut input_path = None;
        let mut report_path = None;
        let mut output_format = "json";
        let mut circuit_parts = Vec::new();
        let mut index = 2;
        while index < args.len() {
            match args[index].as_str() {
                "--line" => use_line_topology = true,
                "--verify" => verify = true,
                "--topology" => {
                    index += 1;
                    topology_path = Some(
                        args.get(index)
                            .ok_or_else(|| "--topology requires a JSON file path".to_owned())?
                            .as_str(),
                    );
                }
                "--input" => {
                    index += 1;
                    let path = args.get(index).ok_or_else(|| {
                        "--input requires a file path (or - for stdin)".to_owned()
                    })?;
                    input_path = Some(path.as_str());
                }
                "--report" => {
                    index += 1;
                    report_path = Some(
                        args.get(index)
                            .ok_or_else(|| "--report requires a file path".to_owned())?
                            .as_str(),
                    );
                }
                "--format" => {
                    index += 1;
                    output_format = args
                        .get(index)
                        .ok_or_else(|| "--format requires 'json' or 'qasm'".to_owned())?;
                    if !matches!(output_format, "json" | "qasm") {
                        return Err("output format must be 'json' or 'qasm'".to_owned());
                    }
                }
                _ => circuit_parts.push(args[index].as_str()),
            }
            index += 1;
        }
        if input_path.is_some() && !circuit_parts.is_empty() {
            return Err("provide either --input or a Circuit[...] expression, not both".to_owned());
        }
        let circuit_source = if let Some(path) = input_path {
            if path == "-" {
                use std::io::Read as _;
                let mut source = String::new();
                std::io::stdin()
                    .read_to_string(&mut source)
                    .map_err(|error| format!("could not read stdin: {error}"))?;
                source
            } else {
                std::fs::read_to_string(path)
                    .map_err(|error| format!("could not read {path}: {error}"))?
            }
        } else {
            circuit_parts.join(" ")
        };
        if circuit_source.is_empty() {
            return Err("provide --input FILE or a Circuit[...] expression to optimize".to_owned());
        }

        let original = if circuit_source.contains("OPENQASM") {
            quantumforge::qasm::parse_qasm(&circuit_source).map_err(|error| error.to_string())?
        } else {
            parse_circuit(&circuit_source).map_err(|error| error.to_string())?
        };
        for operation in original.operations() {
            let out_of_range = match operation {
                quantumforge::CircuitOp::SingleQubit { target, .. }
                | quantumforge::CircuitOp::Channel { target, .. }
                | quantumforge::CircuitOp::Measure { target, .. }
                | quantumforge::CircuitOp::ConditionalSingleQubit { target, .. } => {
                    (*target >= qubits).then_some(*target)
                }
                quantumforge::CircuitOp::Cnot { control, target } => [*control, *target]
                    .into_iter()
                    .find(|qubit| *qubit >= qubits),
                quantumforge::CircuitOp::Cz { first, second }
                | quantumforge::CircuitOp::Swap { first, second } => {
                    [*first, *second].into_iter().find(|qubit| *qubit >= qubits)
                }
            };
            if let Some(qubit) = out_of_range {
                return Err(format!(
                    "circuit references qubit {qubit}, outside the declared {qubits}-qubit register"
                ));
            }
        }
        if use_line_topology && topology_path.is_some() {
            return Err("choose either --line or --topology, not both".to_owned());
        }
        let coupling_map = if use_line_topology {
            Some((
                "linear",
                DeviceTarget {
                    coupling: CouplingMap::linear(qubits),
                    native_gates: None,
                    directed_cx_edges: None,
                },
            ))
        } else if let Some(path) = topology_path {
            Some(("custom", read_coupling_map(path, qubits)?))
        } else {
            None
        };
        let original_profile = CircuitCostProfile::from_circuit(&original);
        let mut optimized = original.clone();
        let result = optimize_circuit(&mut optimized, &config);
        let routing_swaps = if let Some((_, target)) = &coupling_map {
            Some(
                try_route_to_hardware(&mut optimized, &target.coupling)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let direction_corrections = if let Some((_, target)) = &coupling_map {
            let corrections = correct_cnot_directions(&mut optimized, target)?;
            validate_device_target(&optimized, target)?;
            Some(corrections)
        } else {
            None
        };
        let optimized_profile = CircuitCostProfile::from_circuit(&optimized);
        let equivalence = if verify {
            Some(
                verify_unitary_equivalent(&original, &optimized, qubits, 1e-9)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };

        if output_format == "qasm"
            && optimized
                .operations()
                .iter()
                .any(|op| matches!(op, quantumforge::CircuitOp::Channel { .. }))
        {
            return Err(
                "OpenQASM 2.0 cannot represent noise-channel operations; use --format json"
                    .to_owned(),
            );
        }

        if let Some(path) = report_path {
            if let Some(input_file) = input_path.filter(|input| *input != "-") {
                if std::path::Path::new(path).exists()
                    && std::fs::canonicalize(path).ok() == std::fs::canonicalize(input_file).ok()
                {
                    return Err(
                        "report path resolves to the input file; choose a different report path"
                            .to_owned(),
                    );
                }
            }
            let resource_summary = |profile: &CircuitCostProfile, operations: usize| {
                serde_json::json!({
                    "operations": operations,
                    "single_qubit_gates": profile.single_qubit_gates,
                    "cnot_gates": profile.cnot_gates,
                    "cz_gates": profile.cz_gates,
                    "swap_gates": profile.swap_gates,
                    "t_gates": profile.t_gates,
                    "tdg_gates": profile.tdg_gates,
                    "two_qubit_gates": profile.cnot_gates + profile.cz_gates + profile.swap_gates,
                    "measurements": profile.measurements,
                    "conditional_gates": profile.conditional_gates,
                    "noise_channels": profile.channels,
                    "logical_depth": profile.depth,
                    "qubits": qubits,
                })
            };
            let report = serde_json::json!({
                "schema_version": 1,
                "tool": "quantumforge",
                "tool_version": env!("CARGO_PKG_VERSION"),
                "input_format": if circuit_source.contains("OPENQASM") { "openqasm2" } else { "quantumforge_expression" },
                "declared_qubits": qubits,
                "optimization_mode": mode,
                "passes": {
                    "remove_identity": config.remove_identity,
                    "cancel_inverses": config.cancel_inverses,
                    "fuse_rotations": config.fuse_rotations,
                    "commute_gates": config.commute_gates,
                },
                "target": match &coupling_map {
                    Some(("linear", _)) => serde_json::json!({ "kind": "linear", "qubits": qubits }),
                    Some((_, target)) => serde_json::json!({
                        "kind": "custom_device_target",
                        "qubits": target.coupling.num_qubits,
                        "edges": target.coupling.edges,
                        "native_gates": target.native_gates.as_ref().map(|gates| {
                            let mut names: Vec<_> = gates.iter().cloned().collect();
                            names.sort();
                            names
                        }),
                        "directed_cx_edges": target.directed_cx_edges.as_ref().map(|edges| {
                            let mut pairs: Vec<_> = edges.iter().copied().collect();
                            pairs.sort();
                            pairs
                        }),
                        "validation": "connectivity, native gate set when specified, and directed CNOT constraints",
                    }),
                    None => serde_json::Value::Null,
                },
                "routing_swaps_inserted": routing_swaps,
                "cnot_direction_corrections": direction_corrections,
                "output_format": output_format,
                "equivalence": match &equivalence {
                    Some(result) => serde_json::json!({
                        "status": if result.equivalent { "verified_equivalent" } else { "mismatch" },
                        "method": "exhaustive_computational_basis_unitary_check",
                        "tolerance": 1e-9,
                        "basis_states_checked": result.basis_states_checked,
                        "global_phase": [result.global_phase.0, result.global_phase.1],
                        "max_amplitude_error": result.max_amplitude_error,
                    }),
                    None => serde_json::json!({ "status": "not_requested" }),
                },
                "cost_model": "structural_counts_only; no hardware calibration or runtime prediction",
                "rewrites": {
                    "inverse_pairs_cancelled": result.cancellations,
                    "rotations_fused": result.fusions,
                    "identities_removed": result.identities_removed,
                },
                "before": resource_summary(&original_profile, original.operations().len()),
                "after": resource_summary(&optimized_profile, optimized.operations().len()),
            });
            let report_json = serde_json::to_vec_pretty(&report)
                .map_err(|error| format!("could not prepare optimization report: {error}"))?;
            std::fs::write(path, report_json)
                .map_err(|error| format!("could not write report to {path}: {error}"))?;
        }

        eprintln!("Optimization mode: {mode}");
        eprintln!(
            "Operations: {} -> {} | 2-qubit gates: {} -> {} | depth: {} -> {}",
            original_profile.total_gates(),
            optimized_profile.total_gates(),
            original_profile.cnot_gates + original_profile.cz_gates + original_profile.swap_gates,
            optimized_profile.cnot_gates
                + optimized_profile.cz_gates
                + optimized_profile.swap_gates,
            original_profile.depth,
            optimized_profile.depth,
        );
        eprintln!(
            "Rewrites: {} inverse pairs cancelled, {} rotations fused, {} identities removed",
            result.cancellations, result.fusions, result.identities_removed
        );
        if let Some(swaps) = routing_swaps {
            let topology = coupling_map
                .as_ref()
                .map(|(kind, _)| *kind)
                .unwrap_or("unknown");
            eprintln!("{topology} topology: {swaps} routing SWAPs inserted");
            eprintln!("Device target validation: passed");
        }
        if let Some(corrections) = direction_corrections {
            if corrections > 0 {
                eprintln!("Directed CNOTs corrected: {corrections} using native Hadamards");
            }
        }
        if let Some(result) = &equivalence {
            eprintln!(
                "Equivalence check: {} ({} basis states, max amplitude error {:.3e})",
                if result.equivalent {
                    "passed"
                } else {
                    "FAILED"
                },
                result.basis_states_checked,
                result.max_amplitude_error,
            );
        }
        if equivalence
            .as_ref()
            .is_some_and(|result| !result.equivalent)
        {
            return Err(
                "optimized circuit failed the requested equivalence check; output was not emitted"
                    .to_owned(),
            );
        }

        match output_format {
            "json" => {
                // stdout stays machine-readable so the circuit can be saved
                // directly and loaded with Circuit::from_json.
                let mut serialized: serde_json::Value =
                    serde_json::from_str(&optimized.to_json().map_err(|error| {
                        format!("could not serialize optimized circuit: {error}")
                    })?)
                    .map_err(|error| {
                        format!("could not prepare optimized circuit output: {error}")
                    })?;
                serialized["num_qubits"] = serde_json::Value::from(qubits);
                serde_json::to_writer_pretty(std::io::stdout().lock(), &serialized)
                    .map_err(|error| format!("could not write optimized circuit: {error}"))?;
                println!();
            }
            "qasm" => {
                println!("{}", quantumforge::qasm_export::export_qasm(&optimized));
            }
            _ => unreachable!(),
        }
        Ok::<(), String>(())
    })();
    if let Err(error) = result {
        eprintln!("optimization error: {error}");
        std::process::exit(2);
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
        // A circuit with no Measure operations would report one empty bitstring
        // carrying every shot. Sample all qubits instead, which is what a caller
        // asking for shots means.
        let mut circuit = circuit;
        if !circuit.has_measurements() {
            for qubit in 0..qubits {
                circuit.measure(qubit, qubit);
            }
        }
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
    println!(
        "QuantumForge {} — CPU quantum kernel and QNN demo",
        env!("CARGO_PKG_VERSION")
    );

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

    let qnn = VariationalQnn::new(2, 2);
    let target_weights: Vec<f64> = vec![0.7; 4]; // 2 qubits * 2 layers = 4 params
    let data: Vec<(f64, f64)> = [-1.0, -0.5, 0.0, 0.5, 1.0]
        .into_iter()
        .map(|x| (x, (x + target_weights[0]).cos()))
        .collect();
    let initial_weights = vec![-1.2; 4];
    let (initial_loss, _) = qnn
        .loss_and_gradient(&data, &initial_weights)
        .expect("valid data");
    let learned_weights = qnn
        .train_adam(&data, initial_weights.clone(), 0.05, 100, 2, 42)
        .expect("training succeeds");
    let (final_loss, _) = qnn
        .loss_and_gradient(&data, &learned_weights)
        .expect("valid data");
    println!("QNN target weights = {target_weights:?}");
    println!("QNN trained weights = {learned_weights:?}");
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
    // This model has exactly one parameter: `encode_rotation(Y, 0, 0)` binds
    // input 0 to parameter 0 and `rotation(Y, 0, 0)` reuses it. The four-element
    // `initial_weights` above belong to the earlier four-parameter `VariationalQnn`
    // and would be rejected as `InvalidParameterCount`.
    let qnn_initial = vec![0.0];
    let (general_initial_loss, _) = qnn_model
        .mse_loss_and_gradient(
            &qnn_data,
            &qnn_initial,
            1,
            quantumforge::SingleQubitObservable::Z,
            0,
        )
        .expect("QNN batch");
    let trained_weights = qnn_model
        .train_regression_adam(
            &qnn_data,
            &qnn_initial,
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

fn read_coupling_map(path: &str, expected_qubits: usize) -> Result<DeviceTarget, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|error| format!("could not read topology {path}: {error}"))?;
    let document: serde_json::Value = serde_json::from_str(&source)
        .map_err(|error| format!("invalid topology JSON in {path}: {error}"))?;
    let num_qubits = document
        .get("num_qubits")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "topology must have a nonnegative integer 'num_qubits'".to_owned())?;
    if num_qubits != expected_qubits {
        return Err(format!(
            "topology declares {num_qubits} qubits but the command declares {expected_qubits}"
        ));
    }
    if num_qubits > 4096 {
        return Err("routing supports at most 4096 qubits".to_owned());
    }
    let edge_values = document
        .get("edges")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            "topology must have an 'edges' array of [qubit_a, qubit_b] pairs".to_owned()
        })?;
    let mut edges = Vec::with_capacity(edge_values.len());
    let mut unique_edges = std::collections::HashSet::new();
    for (index, edge) in edge_values.iter().enumerate() {
        let pair = edge
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| format!("topology edge {index} must be a two-item array"))?;
        let first = pair[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("topology edge {index} has an invalid first qubit"))?;
        let second = pair[1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("topology edge {index} has an invalid second qubit"))?;
        if first >= num_qubits || second >= num_qubits {
            return Err(format!(
                "topology edge ({first}, {second}) is outside the {num_qubits}-qubit map"
            ));
        }
        if first == second {
            return Err(format!("topology edge {index} connects a qubit to itself"));
        }
        let normalized = (first.min(second), first.max(second));
        if !unique_edges.insert(normalized) {
            return Err(format!(
                "topology contains duplicate edge ({first}, {second})"
            ));
        }
        edges.push((first, second));
    }
    let native_gates = document
        .get("native_gates")
        .map(|value| {
            let gates = value
                .as_array()
                .ok_or_else(|| "target 'native_gates' must be an array of gate names".to_owned())?;
            let mut normalized = std::collections::HashSet::new();
            for (index, gate) in gates.iter().enumerate() {
                let gate = gate
                    .as_str()
                    .ok_or_else(|| format!("target native_gates entry {index} must be a string"))?;
                let gate = normalize_native_gate(gate)
                    .ok_or_else(|| format!("unsupported native gate name '{gate}' in target"))?;
                if !normalized.insert(gate.to_owned()) {
                    return Err(format!("target contains duplicate native gate '{gate}'"));
                }
            }
            Ok(normalized)
        })
        .transpose()?;
    let directed_cx_edges = document
        .get("directed_cx_edges")
        .map(|value| {
            let edge_values = value.as_array().ok_or_else(|| {
                "target 'directed_cx_edges' must be an array of [control, target] pairs".to_owned()
            })?;
            let mut directed = std::collections::HashSet::new();
            for (index, edge) in edge_values.iter().enumerate() {
                let pair = edge
                    .as_array()
                    .filter(|pair| pair.len() == 2)
                    .ok_or_else(|| format!("directed CNOT edge {index} must be a two-item array"))?;
                let control = pair[0].as_u64().and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| format!("directed CNOT edge {index} has an invalid control qubit"))?;
                let target = pair[1].as_u64().and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| format!("directed CNOT edge {index} has an invalid target qubit"))?;
                if control >= num_qubits || target >= num_qubits || control == target {
                    return Err(format!("directed CNOT edge ({control}, {target}) is invalid for a {num_qubits}-qubit target"));
                }
                let edge = (control, target);
                if !unique_edges.contains(&(control.min(target), control.max(target))) {
                    return Err(format!("directed CNOT edge ({control}, {target}) is absent from 'edges'"));
                }
                if !directed.insert(edge) {
                    return Err(format!("target contains duplicate directed CNOT edge ({control}, {target})"));
                }
            }
            Ok(directed)
        })
        .transpose()?;
    if directed_cx_edges.is_some()
        && native_gates
            .as_ref()
            .is_some_and(|gates| !gates.contains("cx"))
    {
        return Err(
            "target declares 'directed_cx_edges' but omits 'cx' from 'native_gates'".to_owned(),
        );
    }
    Ok(DeviceTarget {
        coupling: CouplingMap { num_qubits, edges },
        native_gates,
        directed_cx_edges,
    })
}

fn normalize_native_gate(gate: &str) -> Option<&'static str> {
    match gate.to_ascii_lowercase().as_str() {
        "i" | "id" | "idgate" => Some("id"),
        "x" | "y" | "z" | "h" | "s" | "sdg" | "t" | "tdg" | "rx" | "ry" | "rz" | "cx" | "cz"
        | "swap" | "measure" | "conditional" => Some(match gate.to_ascii_lowercase().as_str() {
            "x" => "x",
            "y" => "y",
            "z" => "z",
            "h" => "h",
            "s" => "s",
            "sdg" => "sdg",
            "t" => "t",
            "tdg" => "tdg",
            "rx" => "rx",
            "ry" => "ry",
            "rz" => "rz",
            "cx" => "cx",
            "cz" => "cz",
            "swap" => "swap",
            "measure" => "measure",
            "conditional" => "conditional",
            _ => "id",
        }),
        _ => None,
    }
}

fn validate_device_target(circuit: &Circuit, target: &DeviceTarget) -> Result<(), String> {
    let supports = |name: &str| {
        target
            .native_gates
            .as_ref()
            .map_or(true, |gates| gates.contains(name))
    };
    let has_edge = |a: usize, b: usize| {
        target
            .coupling
            .edges
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
    };
    for (index, operation) in circuit.operations().iter().enumerate() {
        let require = |name: &str| {
            if supports(name) {
                Ok(())
            } else {
                Err(format!(
                    "device target does not support operation {index}: native gate '{name}' is absent"
                ))
            }
        };
        match operation {
            quantumforge::CircuitOp::SingleQubit { gate, .. } => require(gate_name(*gate))?,
            quantumforge::CircuitOp::Cnot {
                control,
                target: target_qubit,
            } => {
                require("cx")?;
                if !has_edge(*control, *target_qubit) {
                    return Err(format!(
                        "device target operation {index}: CNOT operands ({control}, {target_qubit}) are not connected"
                    ));
                }
                if target
                    .directed_cx_edges
                    .as_ref()
                    .is_some_and(|edges| !edges.contains(&(*control, *target_qubit)))
                {
                    return Err(format!(
                        "device target operation {index}: CNOT direction {control}->{target_qubit} is unsupported"
                    ));
                }
            }
            quantumforge::CircuitOp::Cz { first, second } => {
                require("cz")?;
                if !has_edge(*first, *second) {
                    return Err(format!(
                        "device target operation {index}: CZ operands ({first}, {second}) are not connected"
                    ));
                }
            }
            quantumforge::CircuitOp::Swap { first, second } => {
                require("swap")?;
                if !has_edge(*first, *second) {
                    return Err(format!(
                        "device target operation {index}: SWAP operands ({first}, {second}) are not connected"
                    ));
                }
            }
            quantumforge::CircuitOp::Measure { .. } => require("measure")?,
            quantumforge::CircuitOp::ConditionalSingleQubit { gate, .. } => {
                require("conditional")?;
                require(gate_name(*gate))?;
            }
            quantumforge::CircuitOp::Channel { .. } => {
                return Err(format!(
                    "device target operation {index}: noise channels are simulator annotations and cannot be submitted as device gates"
                ));
            }
        }
    }
    Ok(())
}

fn correct_cnot_directions(circuit: &mut Circuit, target: &DeviceTarget) -> Result<usize, String> {
    let Some(directed_edges) = &target.directed_cx_edges else {
        return Ok(0);
    };
    let mut corrected = Vec::with_capacity(circuit.operations().len());
    let mut corrections = 0;
    for (index, operation) in circuit.operations().iter().copied().enumerate() {
        match operation {
            quantumforge::CircuitOp::Cnot {
                control,
                target: target_qubit,
            } if !directed_edges.contains(&(control, target_qubit)) => {
                if !directed_edges.contains(&(target_qubit, control)) {
                    return Err(format!(
                        "device target operation {index}: neither CNOT direction is supported for qubits {control} and {target_qubit}"
                    ));
                }
                if !target
                    .native_gates
                    .as_ref()
                    .is_some_and(|gates| gates.contains("h"))
                {
                    return Err(format!(
                        "device target operation {index}: reversing CNOT {control}->{target_qubit} requires native 'h' in the target's native_gates list"
                    ));
                }
                // CX(a,b) = H(a) H(b) CX(b,a) H(a) H(b).
                corrected.push(quantumforge::CircuitOp::SingleQubit {
                    gate: Gate::H,
                    target: control,
                });
                corrected.push(quantumforge::CircuitOp::SingleQubit {
                    gate: Gate::H,
                    target: target_qubit,
                });
                corrected.push(quantumforge::CircuitOp::Cnot {
                    control: target_qubit,
                    target: control,
                });
                corrected.push(quantumforge::CircuitOp::SingleQubit {
                    gate: Gate::H,
                    target: control,
                });
                corrected.push(quantumforge::CircuitOp::SingleQubit {
                    gate: Gate::H,
                    target: target_qubit,
                });
                corrections += 1;
            }
            _ => corrected.push(operation),
        }
    }
    *circuit.operations_mut() = corrected;
    Ok(corrections)
}

fn gate_name(gate: Gate) -> &'static str {
    match gate {
        Gate::I => "id",
        Gate::X => "x",
        Gate::Y => "y",
        Gate::Z => "z",
        Gate::H => "h",
        Gate::S => "s",
        Gate::Sdg => "sdg",
        Gate::T => "t",
        Gate::Tdg => "tdg",
        Gate::Rx(_) => "rx",
        Gate::Ry(_) => "ry",
        Gate::Rz(_) => "rz",
    }
}

fn print_help() {
    println!("QuantumForge — local quantum simulation and symbolic tools");
    println!("\nUsage: quantumforge [demo|eval EXPR|simplify EXPR|optimize ...|help]");
    println!("       quantumforge shots QUBITS SHOTS SEED 'Circuit[...]'");
    println!(
        "       quantumforge optimize QUBITS all|conservative [--line | --topology FILE] [--verify] [--input FILE] [--format json|qasm] [--report FILE] [Circuit[...]]"
    );
    #[cfg(feature = "llvm")]
    println!("       quantumforge llvm-ir EXPR");
    #[cfg(feature = "llvm")]
    println!("       quantumforge jit-eval OPT_LEVEL 'EXPR' [SYMBOL=VALUE ...]");
    #[cfg(feature = "cuda")]
    println!("       quantumforge cuda-demo");
    println!(
        "\nThe demo runs a one-qubit CPU simulation, seeded measurement, and a tiny variational QNN training loop."
    );
    println!(
        "\nExamples:\n\
         quantumforge shots 2 1000 42 'Circuit[H[0], CNOT[0,1]]'\n\
         quantumforge optimize 3 all --line 'Circuit[H[0], H[0], CNOT[0,2]]'\n\
         quantumforge optimize 3 all --input bell.qasm --line --format qasm\n\
         quantumforge optimize 3 all --input bell.qasm --report compile-report.json\n\
         quantumforge optimize 5 all --topology device-map.json --input circuit.qasm\n\
         quantumforge optimize 3 all --verify 'Circuit[H[0], H[0]]'\n\
         quantumforge eval '2 + 3*Pi'\n\
         qf  # interactive REPL\n\
         \n\
         shots samples every qubit when the circuit declares no Measure operations;\n\
         otherwise it uses the circuit's own Measure[q,c] mapping.\n\
         optimize writes reusable JSON to stdout by default; use --format qasm for OpenQASM 2.0, and --report FILE for a separate JSON provenance and metrics report.\n\
         The resource summary is written to stderr."
    );
}
