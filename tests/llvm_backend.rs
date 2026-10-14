#![cfg(feature = "llvm")]

use quantumforge::llvm_backend::{LlvmError, compile_object, emit_llvm_ir, optimize_ir};
use quantumforge::numeric_ir::NumericBuilder;
use quantumforge::parser::parse_scalar_expr;

fn sample_program() -> quantumforge::numeric_ir::NumericProgram {
    let mut builder = NumericBuilder::new(1);
    let x = builder.input(0).unwrap();
    let square = builder.powi(x, 2).unwrap();
    let sine = builder.sin(square).unwrap();
    let offset = builder.constant(0.25).unwrap();
    let output = builder.add(sine, offset).unwrap();
    builder.finish(output).unwrap()
}

#[test]
fn lowerer_emits_typed_ssa_and_intrinsic_calls() {
    let ir = emit_llvm_ir(&sample_program());
    assert!(ir.contains("define double @quantumforge_eval(double %input0)"));
    assert!(ir.contains("declare double @llvm.sin.f64(double)"));
    assert!(ir.contains("declare double @llvm.powi.f64.i32(double, i32)"));
    assert!(ir.contains("call double @llvm.powi.f64.i32(double %v0, i32 2)"));
    assert!(ir.contains("call double @llvm.sin.f64(double %v1)"));
    assert!(ir.contains("ret double %v4"));
}

#[test]
fn generated_program_agrees_with_reference_interpreter() {
    let program = sample_program();
    let interpreted = program.evaluate(&[0.5]).unwrap();
    let expected = 0.25_f64.sin() + 0.25;
    assert!((interpreted - expected).abs() < 1e-15);
    assert_eq!(program.input_count(), 1);
}

#[test]
fn rustc_llvm_runtime_compilation_matches_the_numeric_interpreter() {
    let program = sample_program();
    let compiled = quantumforge::llvm_backend::compile_jit(&program, 3).unwrap();
    assert_eq!(compiled.input_count(), program.input_count());
    assert!(compiled.module_path().exists());
    for input in [-2.0, -0.5, 0.0, 0.5, 1.7] {
        let interpreted = program.evaluate(&[input]).unwrap();
        let native = compiled.evaluate(&[input]).unwrap();
        assert!((native - interpreted).abs() < 1e-14);
    }
    assert!(matches!(
        compiled.evaluate(&[]),
        Err(
            quantumforge::numeric_ir::NumericIrError::InvalidInputCount {
                expected: 1,
                actual: 0
            }
        )
    ));
    assert!(matches!(
        compiled.evaluate(&[f64::INFINITY]),
        Err(quantumforge::numeric_ir::NumericIrError::NonFiniteValue)
    ));
    let module_path = compiled.module_path().to_path_buf();
    drop(compiled);
    assert!(!module_path.exists());
}

#[test]
fn rustc_llvm_runtime_module_preserves_numeric_errors() {
    let mut builder = NumericBuilder::new(1);
    let x = builder.input(0).unwrap();
    let zero = builder.constant(0.0).unwrap();
    let quotient = builder.divide(x, zero).unwrap();
    let program = builder.finish(quotient).unwrap();
    let compiled = quantumforge::llvm_backend::compile_jit(&program, 2).unwrap();
    assert!(matches!(
        compiled.evaluate(&[1.0]),
        Err(quantumforge::numeric_ir::NumericIrError::DivisionByZero)
    ));

    let mut constant_builder = NumericBuilder::new(0);
    let pi = constant_builder.constant(std::f64::consts::PI).unwrap();
    let constant_program = constant_builder.finish(pi).unwrap();
    let constant_native = quantumforge::llvm_backend::compile_jit(&constant_program, 3).unwrap();
    assert_eq!(constant_native.evaluate(&[]).unwrap(), std::f64::consts::PI);
}

#[test]
fn external_tool_drivers_validate_options_before_launch() {
    let ir = emit_llvm_ir(&sample_program());
    assert!(matches!(
        optimize_ir(&ir, "missing-opt", 4),
        Err(LlvmError::InvalidOptimizationLevel(4))
    ));
    assert!(matches!(
        compile_object(&ir, "missing-llc", "unused.o", 4),
        Err(LlvmError::InvalidOptimizationLevel(4))
    ));
}

#[test]
fn parsed_real_scalar_expressions_lower_to_numeric_and_llvm_ir() {
    let expression = parse_scalar_expr("Sin[Pi/4] + x^2 + Abs[y]").unwrap();
    let program =
        quantumforge::numeric_ir::NumericProgram::compile_scalar(&expression, &["x", "y"]).unwrap();
    let actual = program.evaluate(&[3.0, -2.0]).unwrap();
    let expected = std::f64::consts::FRAC_1_SQRT_2 + 9.0 + 2.0;
    assert!((actual - expected).abs() < 1e-14);

    let ir = emit_llvm_ir(&program);
    assert_eq!(
        ir,
        quantumforge::llvm_backend::lower_scalar_to_llvm_ir(&expression, &["x", "y"]).unwrap()
    );
    assert!(ir.contains("@llvm.sin.f64"));
    assert!(ir.contains("@llvm.fabs.f64"));
    assert!(ir.contains("@llvm.powi.f64.i32"));
}

#[test]
fn scalar_lowering_rejects_unbound_and_complex_values() {
    let unbound = parse_scalar_expr("x + y").unwrap();
    assert!(matches!(
        quantumforge::numeric_ir::NumericProgram::compile_scalar(&unbound, &["x"]),
        Err(quantumforge::numeric_ir::NumericIrError::UnknownSymbol(name)) if name == "y"
    ));

    let complex = parse_scalar_expr("I + 1").unwrap();
    assert!(matches!(
        quantumforge::numeric_ir::NumericProgram::compile_scalar(&complex, &[]),
        Err(quantumforge::numeric_ir::NumericIrError::ComplexValueUnsupported)
    ));
}
