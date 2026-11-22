//! Typed scalar numeric IR and reference interpreter.
//!
//! Numeric IR: a compact SSA instruction stream for LLVM code generation.
//!
//! # Relationship to other representations
//!
//! `ScalarExpr` (in `analytic.rs`) is the canonical symbolic representation.
//! `NumericProgram` is derived from it via `compile_scalar`.  Evaluation of
//! `ScalarExpr` (via `evaluate`) and `NumericProgram` (via `evaluate`) must
//! agree; if you change one, update the other and add a consistency test.
//!
//! A future unification would make `ScalarExpr` the single source of truth
//! with `NumericProgram` as a compiled view, eliminating the duplicate
//! operation dispatch.
//! `NumericProgram` is the lowering boundary for future LLVM code generation.
//! It deliberately contains only pure `f64` operations; symbolic expressions
//! remain in `analytic` until explicitly lowered through the supported
//! real-valued scalar subset.

use crate::analytic::{NamedConstant, ScalarExpr};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// SSA register identifier: the index of the instruction that produced a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Register(usize);

impl Register {
    /// Return the underlying instruction index this register refers to.
    pub fn index(self) -> usize {
        self.0
    }
}

/// A single pure-`f64` SSA instruction in a [`NumericProgram`].
#[derive(Clone, Debug, PartialEq)]
pub enum NumericInstruction {
    /// Literal `f64` constant value.
    Constant(f64),
    /// Read the program input at the given index.
    Input(usize),
    /// Add the values held in two registers.
    Add(Register, Register),
    /// Subtract the second register's value from the first.
    Subtract(Register, Register),
    /// Multiply the values held in two registers.
    Multiply(Register, Register),
    /// Divide the first register's value by the second.
    Divide(Register, Register),
    /// Raise a register's value to an integer power.
    PowerI(Register, i32),
    /// Negate a register's value.
    Negate(Register),
    /// Absolute value of a register's value.
    Abs(Register),
    /// Square root of a register's value.
    Sqrt(Register),
    /// Sine of a register's value.
    Sin(Register),
    /// Cosine of a register's value.
    Cos(Register),
    /// Natural exponential of a register's value.
    Exp(Register),
    /// Natural logarithm of a register's value.
    Log(Register),
}

/// Errors produced while lowering, building, or evaluating numeric IR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumericIrError {
    /// Evaluation received the wrong number of inputs.
    InvalidInputCount {
        /// Number of inputs the program expects.
        expected: usize,
        /// Number of inputs actually supplied.
        actual: usize,
    },
    /// An `Input` instruction referenced an index outside the program's inputs.
    InputOutOfRange(usize),
    /// A register index did not correspond to an emitted instruction.
    RegisterOutOfRange(usize),
    /// The same input symbol was bound more than once during lowering.
    DuplicateInputSymbol(String),
    /// A symbol in the expression had no bound input.
    UnknownSymbol(String),
    /// The expression is complex-valued and cannot be lowered to real IR.
    ComplexValueUnsupported,
    /// The scalar expression uses operations outside the real-valued subset.
    ///
    /// Currently **unreachable**: `ScalarExpr` is a closed enum of thirteen
    /// variants and `lower_scalar` matches all thirteen exhaustively, so no
    /// expression can produce this. It is kept so that adding a fourteenth
    /// variant forces a decision here rather than a silent `unreachable!()`.
    /// Do not write a test asserting this variant — there is nothing to call.
    UnsupportedScalarExpression,
    /// An input or computed value was not finite.
    NonFiniteValue,
    /// A division by zero was attempted during evaluation.
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

    /// Return the number of inputs the program expects.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Return the number of emitted instructions.
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Return the instruction stream in emission order.
    pub fn instructions(&self) -> &[NumericInstruction] {
        &self.instructions
    }

    /// Return the register holding the program output.
    pub fn output(&self) -> Register {
        self.output
    }

    /// Interpret the program on the given inputs and return the output value.
    ///
    /// The slice length must equal [`Self::input_count`], every input must be
    /// finite, and every intermediate result must stay finite; division by zero
    /// is rejected rather than producing infinity.
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
    /// Create an empty builder for a program with `input_count` inputs.
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

    /// Emit a constant, rejecting non-finite values.
    pub fn constant(&mut self, value: f64) -> Result<Register, NumericIrError> {
        if !value.is_finite() {
            return Err(NumericIrError::NonFiniteValue);
        }
        Ok(self.emit(NumericInstruction::Constant(value)))
    }

    /// Emit an input read, rejecting indices outside the declared input count.
    pub fn input(&mut self, index: usize) -> Result<Register, NumericIrError> {
        if index >= self.input_count {
            return Err(NumericIrError::InputOutOfRange(index));
        }
        Ok(self.emit(NumericInstruction::Input(index)))
    }

    /// Emit an addition of two existing registers.
    pub fn add(&mut self, a: Register, b: Register) -> Result<Register, NumericIrError> {
        self.binary(a, b, NumericInstruction::Add)
    }
    /// Emit a subtraction of two existing registers.
    pub fn subtract(&mut self, a: Register, b: Register) -> Result<Register, NumericIrError> {
        self.binary(a, b, NumericInstruction::Subtract)
    }
    /// Emit a multiplication of two existing registers.
    pub fn multiply(&mut self, a: Register, b: Register) -> Result<Register, NumericIrError> {
        self.binary(a, b, NumericInstruction::Multiply)
    }
    /// Emit a division of two existing registers.
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

    /// Emit an integer-power instruction on an existing register.
    pub fn powi(&mut self, value: Register, exponent: i32) -> Result<Register, NumericIrError> {
        self.unary(value, |v| NumericInstruction::PowerI(v, exponent))
    }
    /// Emit a negation of an existing register.
    pub fn negate(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Negate)
    }
    /// Emit an absolute-value instruction on an existing register.
    pub fn abs(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Abs)
    }
    /// Emit a square-root instruction on an existing register.
    pub fn sqrt(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Sqrt)
    }
    /// Emit a sine instruction on an existing register.
    pub fn sin(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Sin)
    }
    /// Emit a cosine instruction on an existing register.
    pub fn cos(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Cos)
    }
    /// Emit a natural-exponential instruction on an existing register.
    pub fn exp(&mut self, value: Register) -> Result<Register, NumericIrError> {
        self.unary(value, NumericInstruction::Exp)
    }
    /// Emit a natural-logarithm instruction on an existing register.
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

    /// Finish the program, designating `output` as its result register.
    ///
    /// Returns an error if `output` does not refer to an emitted instruction.
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
