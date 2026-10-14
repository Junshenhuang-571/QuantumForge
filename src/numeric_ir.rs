//! Typed scalar numeric IR and reference interpreter.
//!
//! The IR is a compact, validated SSA instruction stream intended as a stable
//! lowering boundary for future LLVM code generation. It deliberately contains
//! only pure `f64` operations; symbolic expressions remain in `analytic` until
//! explicitly lowered through the supported real-valued scalar subset.

use crate::analytic::{NamedConstant, ScalarExpr};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Register(usize);

impl Register {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NumericInstruction {
    Constant(f64),
    Input(usize),
    Add(Register, Register),
    Subtract(Register, Register),
    Multiply(Register, Register),
    Divide(Register, Register),
    PowerI(Register, i32),
    Negate(Register),
    Abs(Register),
    Sqrt(Register),
    Sin(Register),
    Cos(Register),
    Exp(Register),
    Log(Register),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericIrError {
    InvalidInputCount { expected: usize, actual: usize },
    InputOutOfRange(usize),
    RegisterOutOfRange(usize),
    DuplicateInputSymbol(String),
    UnknownSymbol(String),
    ComplexValueUnsupported,
    UnsupportedScalarExpression,
    NonFiniteValue,
    DivisionByZero,
}

impl fmt::Display for NumericIrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInputCount { expected, actual } => {
                write!(f, "numeric program expects {expected} inputs, got {actual}")
            }
            Self::InputOutOfRange(index) => {
                write!(f, "numeric input index {index} is out of range")
            }
            Self::RegisterOutOfRange(index) => {
                write!(f, "numeric IR register {index} is out of range")
            }
            Self::DuplicateInputSymbol(name) => {
                write!(f, "numeric input symbol '{name}' is bound more than once")
            }
            Self::UnknownSymbol(name) => {
                write!(f, "no numeric input was bound for symbol '{name}'")
            }
            Self::ComplexValueUnsupported => {
                write!(
                    f,
                    "complex-valued expressions cannot be lowered to real numeric IR"
                )
            }
            Self::UnsupportedScalarExpression => {
                write!(
                    f,
                    "this scalar expression is unsupported by real numeric IR"
                )
            }
            Self::NonFiniteValue => write!(f, "numeric IR values must remain finite"),
            Self::DivisionByZero => write!(f, "numeric IR division by zero"),
        }
    }
}
impl Error for NumericIrError {}

/// Pure `f64` SSA program with an interpreter-backed reference implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct NumericProgram {
    input_count: usize,
    instructions: Vec<NumericInstruction>,
    output: Register,
}

impl NumericProgram {
    /// Lower a scalar expression to real-valued IR using the supplied symbol
    /// order as the program's input order. Exact complex constants are accepted
    /// only when their imaginary component is zero.
    pub fn compile_scalar(
        expression: &ScalarExpr,
        input_symbols: &[&str],
    ) -> Result<Self, NumericIrError> {
        let mut builder = NumericBuilder::new(input_symbols.len());
        let mut inputs = HashMap::with_capacity(input_symbols.len());
        for (index, &name) in input_symbols.iter().enumerate() {
            if inputs.contains_key(name) {
                return Err(NumericIrError::DuplicateInputSymbol(name.to_owned()));
            }
            inputs.insert(name, builder.input(index)?);
        }
        let output = lower_scalar(expression, &mut builder, &inputs)?;
        builder.finish(output)
    }

    pub fn input_count(&self) -> usize {
        self.input_count
    }

    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    pub fn instructions(&self) -> &[NumericInstruction] {
        &self.instructions
    }

    pub fn output(&self) -> Register {
        self.output
    }

    pub fn evaluate(&self, inputs: &[f64]) -> Result<f64, NumericIrError> {
        if inputs.len() != self.input_count {
            return Err(NumericIrError::InvalidInputCount {
                expected: self.input_count,
                actual: inputs.len(),
            });
        }
        if inputs.iter().any(|value| !value.is_finite()) {
            return Err(NumericIrError::NonFiniteValue);
        }
        let mut values: Vec<f64> = Vec::with_capacity(self.instructions.len());
        for instruction in &self.instructions {
            let read = |register: Register| {
                values
                    .get(register.0)
                    .copied()
                    .ok_or(NumericIrError::RegisterOutOfRange(register.0))
            };
            let value = match *instruction {
                NumericInstruction::Constant(value) => value,
                NumericInstruction::Input(index) => *inputs
                    .get(index)
                    .ok_or(NumericIrError::InputOutOfRange(index))?,
                NumericInstruction::Add(a, b) => read(a)? + read(b)?,
                NumericInstruction::Subtract(a, b) => read(a)? - read(b)?,
                NumericInstruction::Multiply(a, b) => read(a)? * read(b)?,
                NumericInstruction::Divide(a, b) => {
                    let denominator = read(b)?;
                    if denominator == 0.0 {
                        return Err(NumericIrError::DivisionByZero);
                    }
                    read(a)? / denominator
                }
                NumericInstruction::PowerI(value, exponent) => read(value)?.powi(exponent),
                NumericInstruction::Negate(value) => -read(value)?,
                NumericInstruction::Abs(value) => read(value)?.abs(),
                NumericInstruction::Sqrt(value) => read(value)?.sqrt(),
                NumericInstruction::Sin(value) => read(value)?.sin(),
                NumericInstruction::Cos(value) => read(value)?.cos(),
                NumericInstruction::Exp(value) => read(value)?.exp(),
                NumericInstruction::Log(value) => read(value)?.ln(),
            };
            if !value.is_finite() {
                return Err(NumericIrError::NonFiniteValue);
            }
            values.push(value);
        }
        values
            .get(self.output.0)
            .copied()
            .ok_or(NumericIrError::RegisterOutOfRange(self.output.0))
    }
}

/// Builder that enforces SSA ordering and input/register bounds.
#[derive(Clone, Debug)]
pub struct NumericBuilder {
    input_count: usize,
    instructions: Vec<NumericInstruction>,
}

impl NumericBuilder {
    pub fn new(input_count: usize) -> Self {
        Self {
            input_count,
            instructions: Vec::new(),
        }
    }

    fn emit(&mut self, instruction: NumericInstruction) -> Register {
        let register = Register(self.instructions.len());
        self.instructions.push(instruction);
        register
    }

    fn check_register(&self, register: Register) -> Result<(), NumericIrError> {
        if register.0 < self.instructions.len() {
            Ok(())
        } else {
            Err(NumericIrError::RegisterOutOfRange(register.0))
        }
    }

    pub fn constant(&mut self, value: f64) -> Result<Register, NumericIrError> {
        if !value.is_finite() {
            return Err(NumericIrError::NonFiniteValue);
        }
        Ok(self.emit(NumericInstruction::Constant(value)))
    }

    pub fn input(&mut self, index: usize) -> Result<Register, NumericIrError> {
        if index >= self.input_count {
            return Err(NumericIrError::InputOutOfRange(index));
        }
        Ok(self.emit(NumericInstruction::Input(index)))
    }

    pub fn add(&mut self, a: Register, b: Register) -> Result<Register, NumericIrError> {
        self.binary(a, b, NumericInstruction::Add)
    }
    pub fn subtract(&mut self, a: Register, b: Register) -> Result<Register, NumericIrError> {
        self.binary(a, b, NumericInstruction::Subtract)
    }
    pub fn multiply(&mut self, a: Register, b: Register) -> Result<Register, NumericIrError> {
        self.binary(a, b, NumericInstruction::Multiply)
    }
    pub fn divide(&mut self, a: Register, b: Register) -> Result<Register, NumericIrError> {
        self.binary(a, b, NumericInstruction::Divide)
    }
    fn binary(
        &mut self,
        a: Register,
        b: Register,
        make: fn(Register, Register) -> NumericInstruction,
    ) -> Result<Register, NumericIrError> {
        self.check_register(a)?;
        self.check_register(b)?;
        Ok(self.emit(make(a, b)))
    }

    pub fn powi(&mut self, value: Register, exponent: i32) -> Result<Register, NumericIrError> {
        self.unary(value, |v| NumericInstruction::PowerI(v, exponent))
    }
    pub fn negate(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Negate)
    }
    pub fn abs(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Abs)
    }
    pub fn sqrt(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Sqrt)
    }
    pub fn sin(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Sin)
    }
    pub fn cos(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Cos)
    }
    pub fn exp(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Exp)
    }
    pub fn log(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Log)
    }
    fn unary(
        &mut self,
        value: Register,
        make: impl FnOnce(Register) -> NumericInstruction,
    ) -> Result<Register, NumericIrError> {
        self.check_register(value)?;
        Ok(self.emit(make(value)))
    }

    pub fn finish(self, output: Register) -> Result<NumericProgram, NumericIrError> {
        if output.0 >= self.instructions.len() {
            return Err(NumericIrError::RegisterOutOfRange(output.0));
        }
        Ok(NumericProgram {
            input_count: self.input_count,
            instructions: self.instructions,
            output,
        })
    }
}

fn lower_scalar(
    expression: &ScalarExpr,
    builder: &mut NumericBuilder,
    inputs: &HashMap<&str, Register>,
) -> Result<Register, NumericIrError> {
    match expression {
        ScalarExpr::Exact(value) => {
            if !value.im.is_zero() {
                return Err(NumericIrError::ComplexValueUnsupported);
            }
            builder.constant(value.re.numerator() as f64 / value.re.denominator() as f64)
        }
        ScalarExpr::Symbol(name) => inputs
            .get(name.as_str())
            .copied()
            .ok_or_else(|| NumericIrError::UnknownSymbol(name.clone())),
        ScalarExpr::Constant(NamedConstant::Pi) => builder.constant(std::f64::consts::PI),
        ScalarExpr::Constant(NamedConstant::E) => builder.constant(std::f64::consts::E),
        ScalarExpr::Add(terms) => {
            let mut terms = terms.iter();
            let Some(first) = terms.next() else {
                return builder.constant(0.0);
            };
            let mut sum = lower_scalar(first, builder, inputs)?;
            for term in terms {
                let next = lower_scalar(term, builder, inputs)?;
                sum = builder.add(sum, next)?;
            }
            Ok(sum)
        }
        ScalarExpr::Mul(factors) => {
            let mut factors = factors.iter();
            let Some(first) = factors.next() else {
                return builder.constant(1.0);
            };
            let mut product = lower_scalar(first, builder, inputs)?;
            for factor in factors {
                let next = lower_scalar(factor, builder, inputs)?;
                product = builder.multiply(product, next)?;
            }
            Ok(product)
        }
        ScalarExpr::Pow(base, exponent) => {
            let value = lower_scalar(base, builder, inputs)?;
            builder.powi(value, *exponent)
        }
        ScalarExpr::Abs(value) => {
            let value = lower_scalar(value, builder, inputs)?;
            builder.abs(value)
        }
        ScalarExpr::Sqrt(value) => {
            let value = lower_scalar(value, builder, inputs)?;
            builder.sqrt(value)
        }
        ScalarExpr::Sin(value) => {
            let value = lower_scalar(value, builder, inputs)?;
            builder.sin(value)
        }
        ScalarExpr::Cos(value) => {
            let value = lower_scalar(value, builder, inputs)?;
            builder.cos(value)
        }
        ScalarExpr::Exp(value) => {
            let value = lower_scalar(value, builder, inputs)?;
            builder.exp(value)
        }
        ScalarExpr::Log(value) => {
            let value = lower_scalar(value, builder, inputs)?;
            builder.log(value)
        }
    }
}
