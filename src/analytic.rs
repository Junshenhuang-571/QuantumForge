//! Exact finite-dimensional Pauli algebra.
//!
//! This module is independent of floating-point simulation: coefficients are
//! rational complex numbers and Pauli expressions simplify by combining equal
//! words exactly.

use crate::Complex64;
use crate::quantum::{HilbertSpace, Ket, LinearError, SparseKet, SparseObservable, SparseOperator};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnalyticError {
    ZeroDenominator,
    Overflow,
    NondifferentiableAbsoluteValue,
    QubitCountMismatch { left: usize, right: usize },
}
impl fmt::Display for AnalyticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator => write!(f, "rational denominator cannot be zero"),
            Self::Overflow => write!(f, "exact rational arithmetic overflowed i128"),
            Self::NondifferentiableAbsoluteValue => write!(
                f,
                "absolute-value derivative is undefined without a nonzero-domain assumption"
            ),
            Self::QubitCountMismatch { left, right } => write!(
                f,
                "Pauli expressions have different qubit counts ({left} and {right})"
            ),
        }
    }
}
impl Error for AnalyticError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rational {
    numerator: i128,
    denominator: i128,
}
impl Rational {
    pub const ZERO: Self = Self {
        numerator: 0,
        denominator: 1,
    };
    pub const ONE: Self = Self {
        numerator: 1,
        denominator: 1,
    };
    pub const MINUS_ONE: Self = Self {
        numerator: -1,
        denominator: 1,
    };

    pub fn new(numerator: i128, denominator: i128) -> Result<Self, AnalyticError> {
        if denominator == 0 {
            return Err(AnalyticError::ZeroDenominator);
        }
        let (n, d) = if denominator < 0 {
            (
                numerator.checked_neg().ok_or(AnalyticError::Overflow)?,
                denominator.checked_neg().ok_or(AnalyticError::Overflow)?,
            )
        } else {
            (numerator, denominator)
        };
        let divisor = gcd(n.unsigned_abs(), d as u128);
        let divisor = i128::try_from(divisor).map_err(|_| AnalyticError::Overflow)?;
        Ok(Self {
            numerator: n / divisor,
            denominator: d / divisor,
        })
    }
    pub fn numerator(self) -> i128 {
        self.numerator
    }
    pub fn denominator(self) -> i128 {
        self.denominator
    }
    pub fn is_zero(self) -> bool {
        self.numerator == 0
    }
    pub fn checked_add(self, rhs: Self) -> Result<Self, AnalyticError> {
        let g = gcd(self.denominator as u128, rhs.denominator as u128) as i128;
        let lmul = rhs.denominator / g;
        let rmul = self.denominator / g;
        let n = self
            .numerator
            .checked_mul(lmul)
            .and_then(|x| {
                rhs.numerator
                    .checked_mul(rmul)
                    .and_then(|y| x.checked_add(y))
            })
            .ok_or(AnalyticError::Overflow)?;
        let d = self
            .denominator
            .checked_mul(lmul)
            .ok_or(AnalyticError::Overflow)?;
        Self::new(n, d)
    }
    pub fn checked_sub(self, rhs: Self) -> Result<Self, AnalyticError> {
        self.checked_add(rhs.checked_neg()?)
    }
    pub fn checked_mul(self, rhs: Self) -> Result<Self, AnalyticError> {
        let g1 = gcd(self.numerator.unsigned_abs(), rhs.denominator as u128) as i128;
        let g2 = gcd(rhs.numerator.unsigned_abs(), self.denominator as u128) as i128;
        let n = (self.numerator / g1)
            .checked_mul(rhs.numerator / g2)
            .ok_or(AnalyticError::Overflow)?;
        let d = (self.denominator / g2)
            .checked_mul(rhs.denominator / g1)
            .ok_or(AnalyticError::Overflow)?;
        Self::new(n, d)
    }
    pub fn checked_div(self, rhs: Self) -> Result<Self, AnalyticError> {
        if rhs.numerator == 0 {
            return Err(AnalyticError::ZeroDenominator);
        }
        let numerator = self
            .numerator
            .checked_mul(rhs.denominator)
            .ok_or(AnalyticError::Overflow)?;
        let denominator = self
            .denominator
            .checked_mul(rhs.numerator)
            .ok_or(AnalyticError::Overflow)?;
        Self::new(numerator, denominator)
    }
    pub fn checked_neg(self) -> Result<Self, AnalyticError> {
        Ok(Self {
            numerator: self
                .numerator
                .checked_neg()
                .ok_or(AnalyticError::Overflow)?,
            denominator: self.denominator,
        })
    }
}
fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}
impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactComplex {
    pub re: Rational,
    pub im: Rational,
}
impl ExactComplex {
    pub const ZERO: Self = Self {
        re: Rational::ZERO,
        im: Rational::ZERO,
    };
    pub const ONE: Self = Self {
        re: Rational::ONE,
        im: Rational::ZERO,
    };
    pub const I: Self = Self {
        re: Rational::ZERO,
        im: Rational::ONE,
    };
    pub fn real(value: Rational) -> Self {
        Self {
            re: value,
            im: Rational::ZERO,
        }
    }
    pub fn is_zero(self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
    pub fn to_complex64(self) -> Complex64 {
        Complex64::new(
            self.re.numerator() as f64 / self.re.denominator() as f64,
            self.im.numerator() as f64 / self.im.denominator() as f64,
        )
    }
    pub fn checked_add(self, rhs: Self) -> Result<Self, AnalyticError> {
        Ok(Self {
            re: self.re.checked_add(rhs.re)?,
            im: self.im.checked_add(rhs.im)?,
        })
    }
    pub fn checked_sub(self, rhs: Self) -> Result<Self, AnalyticError> {
        Ok(Self {
            re: self.re.checked_sub(rhs.re)?,
            im: self.im.checked_sub(rhs.im)?,
        })
    }
    pub fn checked_mul(self, rhs: Self) -> Result<Self, AnalyticError> {
        let re = self
            .re
            .checked_mul(rhs.re)?
            .checked_sub(self.im.checked_mul(rhs.im)?)?;
        let im = self
            .re
            .checked_mul(rhs.im)?
            .checked_add(self.im.checked_mul(rhs.re)?)?;
        Ok(Self { re, im })
    }
    pub fn checked_neg(self) -> Result<Self, AnalyticError> {
        Ok(Self {
            re: self.re.checked_neg()?,
            im: self.im.checked_neg()?,
        })
    }
    pub fn checked_div(self, rhs: Self) -> Result<Self, AnalyticError> {
        let denominator = rhs
            .re
            .checked_mul(rhs.re)?
            .checked_add(rhs.im.checked_mul(rhs.im)?)?;
        if denominator.is_zero() {
            return Err(AnalyticError::ZeroDenominator);
        }
        Ok(Self {
            re: self
                .re
                .checked_mul(rhs.re)?
                .checked_add(self.im.checked_mul(rhs.im)?)?
                .checked_div(denominator)?,
            im: self
                .im
                .checked_mul(rhs.re)?
                .checked_sub(self.re.checked_mul(rhs.im)?)?
                .checked_div(denominator)?,
        })
    }
    pub fn checked_pow(self, exponent: i32) -> Result<Self, AnalyticError> {
        let mut power = exponent.unsigned_abs();
        let mut factor = if exponent < 0 {
            Self::ONE.checked_div(self)?
        } else {
            self
        };
        let mut result = Self::ONE;
        while power > 0 {
            if power & 1 == 1 {
                result = result.checked_mul(factor)?;
            }
            factor = factor.checked_mul(factor)?;
            power >>= 1;
        }
        Ok(result)
    }
}
impl fmt::Display for ExactComplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.re.is_zero(), self.im.is_zero()) {
            (true, true) => write!(f, "0"),
            (_, true) => write!(f, "{}", self.re),
            (true, _) if self.im == Rational::ONE => write!(f, "i"),
            (true, _) if self.im == Rational::MINUS_ONE => write!(f, "-i"),
            (true, _) => write!(f, "{}i", self.im),
            (_, _) if self.im == Rational::ONE => write!(f, "{}+i", self.re),
            (_, _) if self.im == Rational::MINUS_ONE => write!(f, "{}-i", self.re),
            (_, _) if self.im.numerator() < 0 => write!(f, "{}{}i", self.re, self.im),
            (_, _) => write!(f, "{}+{}i", self.re, self.im),
        }
    }
}

/// Canonical named transcendental constants supported by the scalar language.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NamedConstant {
    Pi,
    E,
}

impl fmt::Display for NamedConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pi => "Pi",
            Self::E => "E",
        })
    }
}

/// Canonical symbolic scalar expressions with exact complex constants.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarExpr {
    Exact(ExactComplex),
    Symbol(String),
    Constant(NamedConstant),
    Add(Vec<ScalarExpr>),
    Mul(Vec<ScalarExpr>),
    Pow(Box<ScalarExpr>, i32),
    Abs(Box<ScalarExpr>),
    Sqrt(Box<ScalarExpr>),
    Sin(Box<ScalarExpr>),
    Cos(Box<ScalarExpr>),
    Exp(Box<ScalarExpr>),
    Log(Box<ScalarExpr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnboundSymbol(pub String);
impl fmt::Display for UnboundSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "no value was provided for symbol {}", self.0)
    }
}
impl Error for UnboundSymbol {}

#[derive(Clone, Debug, PartialEq)]
pub enum NumericPauliError {
    Analytic(AnalyticError),
    LinearAlgebra(LinearError),
    UnboundSymbol(UnboundSymbol),
}
impl fmt::Display for NumericPauliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Analytic(error) => write!(f, "{error}"),
            Self::LinearAlgebra(error) => write!(f, "{error}"),
            Self::UnboundSymbol(error) => write!(f, "{error}"),
        }
    }
}
impl Error for NumericPauliError {}

impl ScalarExpr {
    pub fn exact(value: ExactComplex) -> Self {
        Self::Exact(value)
    }
    pub fn symbol(name: impl Into<String>) -> Self {
        Self::Symbol(name.into())
    }
    pub fn pi() -> Self {
        Self::Constant(NamedConstant::Pi)
    }
    pub fn e() -> Self {
        Self::Constant(NamedConstant::E)
    }
    pub fn zero() -> Self {
        Self::Exact(ExactComplex::ZERO)
    }
    pub fn one() -> Self {
        Self::Exact(ExactComplex::ONE)
    }
    pub fn abs(value: Self) -> Self {
        Self::Abs(Box::new(value))
    }
    pub fn sqrt(value: Self) -> Self {
        Self::Sqrt(Box::new(value))
    }
    pub fn sin(value: Self) -> Self {
        Self::Sin(Box::new(value))
    }
    pub fn cos(value: Self) -> Self {
        Self::Cos(Box::new(value))
    }
    pub fn exp(value: Self) -> Self {
        Self::Exp(Box::new(value))
    }
    pub fn log(value: Self) -> Self {
        Self::Log(Box::new(value))
    }

    /// Formal derivative with respect to one symbol. The absolute-value rule
    /// is intentionally rejected when the expression depends on that symbol;
    /// callers can simplify under sign assumptions before differentiating.
    pub fn differentiate(&self, variable: &str) -> Result<Self, AnalyticError> {
        match self {
            Self::Exact(_) | Self::Constant(_) => Ok(Self::zero()),
            Self::Symbol(name) => Ok(if name == variable {
                Self::one()
            } else {
                Self::zero()
            }),
            Self::Add(terms) => Self::add(
                terms
                    .iter()
                    .map(|term| term.differentiate(variable))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Self::Mul(factors) => {
                let mut terms = Vec::with_capacity(factors.len());
                for (index, factor) in factors.iter().enumerate() {
                    let derivative = factor.differentiate(variable)?;
                    if derivative == Self::zero() {
                        continue;
                    }
                    let mut product = Vec::with_capacity(factors.len());
                    product.extend(factors[..index].iter().cloned());
                    product.push(derivative);
                    product.extend(factors[index + 1..].iter().cloned());
                    terms.push(Self::mul(product)?);
                }
                Self::add(terms)
            }
            Self::Pow(base, exponent) => {
                let derivative = base.differentiate(variable)?;
                if *exponent == 0 || derivative == Self::zero() {
                    return Ok(Self::zero());
                }
                let reduced = exponent.checked_sub(1).ok_or(AnalyticError::Overflow)?;
                Self::mul([
                    Self::exact(ExactComplex::real(Rational::new(i128::from(*exponent), 1)?)),
                    Self::pow(base.as_ref().clone(), reduced)?,
                    derivative,
                ])
            }
            Self::Abs(value) => {
                if value.differentiate(variable)? == Self::zero() {
                    Ok(Self::zero())
                } else {
                    Err(AnalyticError::NondifferentiableAbsoluteValue)
                }
            }
            Self::Sqrt(value) => {
                let derivative = value.differentiate(variable)?;
                if derivative == Self::zero() {
                    return Ok(Self::zero());
                }
                Self::mul([
                    Self::exact(ExactComplex::real(Rational::new(1, 2)?)),
                    derivative,
                    Self::pow(Self::sqrt(value.as_ref().clone()), -1)?,
                ])
            }
            Self::Sin(value) => Self::mul([
                Self::cos(value.as_ref().clone()),
                value.differentiate(variable)?,
            ]),
            Self::Cos(value) => Self::mul([
                Self::exact(ExactComplex::real(Rational::MINUS_ONE)),
                Self::sin(value.as_ref().clone()),
                value.differentiate(variable)?,
            ]),
            Self::Exp(value) => Self::mul([
                Self::exp(value.as_ref().clone()),
                value.differentiate(variable)?,
            ]),
            Self::Log(value) => {
                let derivative = value.differentiate(variable)?;
                if derivative == Self::zero() {
                    return Ok(Self::zero());
                }
                Self::mul([derivative, Self::pow(value.as_ref().clone(), -1)?])
            }
        }
    }

    pub fn add(terms: impl IntoIterator<Item = Self>) -> Result<Self, AnalyticError> {
        let mut flattened = Vec::new();
        for term in terms {
            match term {
                Self::Add(xs) => flattened.extend(xs),
                other => flattened.push(other),
            }
        }
        let mut constant = ExactComplex::ZERO;
        let mut counts: BTreeMap<Self, u64> = BTreeMap::new();
        for term in flattened {
            match term {
                Self::Exact(value) => constant = constant.checked_add(value)?,
                other => {
                    let count = counts.entry(other).or_default();
                    *count = count.checked_add(1).ok_or(AnalyticError::Overflow)?;
                }
            }
        }
        let mut out = Vec::new();
        if !constant.is_zero() {
            out.push(Self::Exact(constant));
        }
        for (term, count) in counts {
            if count == 1 {
                out.push(term);
            } else {
                let coefficient = i128::from(count);
                out.push(Self::mul([
                    Self::Exact(ExactComplex::real(Rational::new(coefficient, 1)?)),
                    term,
                ])?);
            }
        }
        match out.len() {
            0 => Ok(Self::zero()),
            1 => Ok(out.pop().unwrap()),
            _ => Ok(Self::Add(out)),
        }
    }

    pub fn mul(factors: impl IntoIterator<Item = Self>) -> Result<Self, AnalyticError> {
        let mut flattened = Vec::new();
        for factor in factors {
            match factor {
                Self::Mul(xs) => flattened.extend(xs),
                other => flattened.push(other),
            }
        }
        let mut coefficient = ExactComplex::ONE;
        let mut counts: BTreeMap<Self, i32> = BTreeMap::new();
        for factor in flattened {
            match factor {
                Self::Exact(value) => {
                    coefficient = coefficient.checked_mul(value)?;
                    if coefficient.is_zero() {
                        return Ok(Self::zero());
                    }
                }
                Self::Pow(base, exponent) => {
                    let count = counts.entry(*base).or_default();
                    *count = count.checked_add(exponent).ok_or(AnalyticError::Overflow)?;
                }
                other => {
                    let count = counts.entry(other).or_default();
                    *count = count.checked_add(1).ok_or(AnalyticError::Overflow)?;
                }
            }
        }
        let mut out = Vec::new();
        if coefficient != ExactComplex::ONE || counts.is_empty() {
            out.push(Self::Exact(coefficient));
        }
        for (base, exponent) in counts {
            if exponent == 0 {
                continue;
            }
            out.push(if exponent == 1 {
                base
            } else {
                Self::Pow(Box::new(base), exponent)
            });
        }
        match out.len() {
            0 => Ok(Self::one()),
            1 => Ok(out.pop().unwrap()),
            _ => Ok(Self::Mul(out)),
        }
    }

    pub fn pow(base: Self, exponent: i32) -> Result<Self, AnalyticError> {
        if exponent == 0 {
            return Ok(Self::one());
        }
        if exponent == 1 {
            return Ok(base);
        }
        if let Self::Pow(inner, old) = base {
            let combined = old.checked_mul(exponent).ok_or(AnalyticError::Overflow)?;
            if combined == 0 {
                Ok(Self::one())
            } else {
                Ok(Self::Pow(inner, combined))
            }
        } else {
            Ok(Self::Pow(Box::new(base), exponent))
        }
    }

    pub fn subtract(left: Self, right: Self) -> Result<Self, AnalyticError> {
        Self::add([
            left,
            Self::mul([Self::Exact(ExactComplex::real(Rational::MINUS_ONE)), right])?,
        ])
    }

    pub fn substitute(&self, values: &HashMap<String, Self>) -> Result<Self, AnalyticError> {
        match self {
            Self::Exact(_) | Self::Constant(_) => Ok(self.clone()),
            Self::Symbol(name) => Ok(values.get(name).cloned().unwrap_or_else(|| self.clone())),
            Self::Add(terms) => Self::add(
                terms
                    .iter()
                    .map(|x| x.substitute(values))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Self::Mul(factors) => Self::mul(
                factors
                    .iter()
                    .map(|x| x.substitute(values))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Self::Pow(base, exponent) => Self::pow(base.substitute(values)?, *exponent),
            Self::Abs(value) => Ok(Self::abs(value.substitute(values)?)),
            Self::Sqrt(value) => Ok(Self::sqrt(value.substitute(values)?)),
            Self::Sin(value) => Ok(Self::sin(value.substitute(values)?)),
            Self::Cos(value) => Ok(Self::cos(value.substitute(values)?)),
            Self::Exp(value) => Ok(Self::exp(value.substitute(values)?)),
            Self::Log(value) => Ok(Self::log(value.substitute(values)?)),
        }
    }

    pub fn evaluate(
        &self,
        values: &HashMap<String, crate::Complex64>,
    ) -> Result<crate::Complex64, UnboundSymbol> {
        match self {
            Self::Exact(value) => Ok(crate::Complex64::new(
                value.re.numerator() as f64 / value.re.denominator() as f64,
                value.im.numerator() as f64 / value.im.denominator() as f64,
            )),
            Self::Constant(NamedConstant::Pi) => {
                Ok(crate::Complex64::new(std::f64::consts::PI, 0.0))
            }
            Self::Constant(NamedConstant::E) => Ok(crate::Complex64::new(std::f64::consts::E, 0.0)),
            Self::Symbol(name) => values
                .get(name)
                .copied()
                .ok_or_else(|| UnboundSymbol(name.clone())),
            Self::Add(terms) => {
                let mut sum = crate::Complex64::ZERO;
                for term in terms {
                    sum = sum + term.evaluate(values)?;
                }
                Ok(sum)
            }
            Self::Mul(factors) => {
                let mut product = crate::Complex64::ONE;
                for factor in factors {
                    product = product * factor.evaluate(values)?;
                }
                Ok(product)
            }
            Self::Pow(base, exponent) => Ok(complex_powi(base.evaluate(values)?, *exponent)),
            Self::Abs(value) => Ok(crate::Complex64::new(value.evaluate(values)?.abs(), 0.0)),
            Self::Sqrt(value) => {
                let z = value.evaluate(values)?;
                let r = z.abs();
                Ok(crate::Complex64::new(
                    ((r + z.re) * 0.5).max(0.0).sqrt(),
                    ((r - z.re) * 0.5).max(0.0).sqrt().copysign(z.im),
                ))
            }
            Self::Sin(value) => Ok(complex_sin(value.evaluate(values)?)),
            Self::Cos(value) => Ok(complex_cos(value.evaluate(values)?)),
            Self::Exp(value) => Ok(complex_exp(value.evaluate(values)?)),
            Self::Log(value) => {
                let z = value.evaluate(values)?;
                Ok(crate::Complex64::new(z.abs().ln(), z.im.atan2(z.re)))
            }
        }
    }
}

fn complex_powi(value: crate::Complex64, exponent: i32) -> crate::Complex64 {
    let mut power = exponent.unsigned_abs();
    let mut factor = value;
    let mut result = crate::Complex64::ONE;
    while power > 0 {
        if power & 1 == 1 {
            result = result * factor;
        }
        factor = factor * factor;
        power >>= 1;
    }
    if exponent < 0 {
        let norm = result.norm_sqr();
        crate::Complex64::new(result.re / norm, -result.im / norm)
    } else {
        result
    }
}

fn complex_exp(value: Complex64) -> Complex64 {
    let scale = value.re.exp();
    Complex64::new(scale * value.im.cos(), scale * value.im.sin())
}

fn complex_sin(value: Complex64) -> Complex64 {
    let cosh = (value.im.exp() + (-value.im).exp()) * 0.5;
    let sinh = (value.im.exp() - (-value.im).exp()) * 0.5;
    Complex64::new(value.re.sin() * cosh, value.re.cos() * sinh)
}

fn complex_cos(value: Complex64) -> Complex64 {
    let cosh = (value.im.exp() + (-value.im).exp()) * 0.5;
    let sinh = (value.im.exp() - (-value.im).exp()) * 0.5;
    Complex64::new(value.re.cos() * cosh, -value.re.sin() * sinh)
}

impl fmt::Display for ScalarExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(value) => write!(f, "{value}"),
            Self::Symbol(name) => write!(f, "{name}"),
            Self::Constant(value) => write!(f, "{value}"),
            Self::Add(terms) => {
                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{term}")?;
                }
                Ok(())
            }
            Self::Mul(factors) => {
                for (i, factor) in factors.iter().enumerate() {
                    if i > 0 {
                        write!(f, " * ")?;
                    }
                    match factor {
                        Self::Add(_) => write!(f, "({factor})")?,
                        _ => write!(f, "{factor}")?,
                    }
                }
                Ok(())
            }
            Self::Pow(base, exponent) => match base.as_ref() {
                Self::Add(_) | Self::Mul(_) => write!(f, "({base})^{exponent}"),
                _ => write!(f, "{base}^{exponent}"),
            },
            Self::Abs(value) => write!(f, "Abs[{value}]"),
            Self::Sqrt(value) => write!(f, "Sqrt[{value}]"),
            Self::Sin(value) => write!(f, "Sin[{value}]"),
            Self::Cos(value) => write!(f, "Cos[{value}]"),
            Self::Exp(value) => write!(f, "Exp[{value}]"),
            Self::Log(value) => write!(f, "Log[{value}]"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Sign {
    #[default]
    Unknown,
    NonNegative,
    Positive,
    NonPositive,
    Negative,
    Zero,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SymbolAssumptions {
    pub real: bool,
    pub integer: bool,
    pub sign: Sign,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assumptions {
    symbols: BTreeMap<String, SymbolAssumptions>,
}
impl Assumptions {
    pub fn set_real(&mut self, symbol: impl Into<String>) {
        self.symbols.entry(symbol.into()).or_default().real = true;
    }
    pub fn set_integer(&mut self, symbol: impl Into<String>) {
        let entry = self.symbols.entry(symbol.into()).or_default();
        entry.real = true;
        entry.integer = true;
    }
    pub fn set_positive(&mut self, symbol: impl Into<String>) {
        let entry = self.symbols.entry(symbol.into()).or_default();
        entry.real = true;
        entry.sign = Sign::Positive;
    }
    pub fn set_nonnegative(&mut self, symbol: impl Into<String>) {
        let entry = self.symbols.entry(symbol.into()).or_default();
        entry.real = true;
        entry.sign = Sign::NonNegative;
    }
    pub fn set_negative(&mut self, symbol: impl Into<String>) {
        let entry = self.symbols.entry(symbol.into()).or_default();
        entry.real = true;
        entry.sign = Sign::Negative;
    }
    pub fn set_nonpositive(&mut self, symbol: impl Into<String>) {
        let entry = self.symbols.entry(symbol.into()).or_default();
        entry.real = true;
        entry.sign = Sign::NonPositive;
    }
    pub fn symbol(&self, name: &str) -> SymbolAssumptions {
        self.symbols.get(name).copied().unwrap_or_default()
    }
    fn is_real(&self, expr: &ScalarExpr) -> bool {
        match expr {
            ScalarExpr::Exact(value) => value.im.is_zero(),
            ScalarExpr::Constant(_) => true,
            ScalarExpr::Symbol(name) => self.symbol(name).real,
            ScalarExpr::Pow(base, _) | ScalarExpr::Abs(base) | ScalarExpr::Sqrt(base) => {
                self.is_real(base)
            }
            ScalarExpr::Sin(value) | ScalarExpr::Cos(value) | ScalarExpr::Exp(value) => {
                self.is_real(value)
            }
            ScalarExpr::Log(value) => self.is_positive(value),
            ScalarExpr::Add(xs) | ScalarExpr::Mul(xs) => xs.iter().all(|x| self.is_real(x)),
        }
    }

    fn is_positive(&self, expr: &ScalarExpr) -> bool {
        match expr {
            ScalarExpr::Constant(NamedConstant::E) => true,
            ScalarExpr::Exact(value) => value.im.is_zero() && value.re.numerator() > 0,
            ScalarExpr::Symbol(name) => self.symbol(name).sign == Sign::Positive,
            _ => false,
        }
    }
}

/// Structural expression patterns. `Any(name)` captures an expression and
/// binds it into the replacement expression under the same symbol name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprPattern {
    Any(String),
    Exact(ScalarExpr),
    Constant(NamedConstant),
    Add(Vec<Self>),
    Mul(Vec<Self>),
    Pow(Box<Self>, i32),
    Abs(Box<Self>),
    Sqrt(Box<Self>),
    Sin(Box<Self>),
    Cos(Box<Self>),
    Exp(Box<Self>),
    Log(Box<Self>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RewriteRule {
    pub pattern: ExprPattern,
    pub replacement: ScalarExpr,
}
impl RewriteRule {
    pub fn new(pattern: ExprPattern, replacement: ScalarExpr) -> Self {
        Self {
            pattern,
            replacement,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    Arithmetic(AnalyticError),
    PassLimit(usize),
}
impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arithmetic(e) => write!(f, "{e}"),
            Self::PassLimit(n) => write!(f, "rewrite did not stabilize in {n} passes"),
        }
    }
}
impl Error for RewriteError {}
impl From<AnalyticError> for RewriteError {
    fn from(value: AnalyticError) -> Self {
        Self::Arithmetic(value)
    }
}

#[derive(Clone, Debug)]
pub struct Simplifier {
    pub assumptions: Assumptions,
    rules: Vec<RewriteRule>,
    max_passes: usize,
}
impl Default for Simplifier {
    fn default() -> Self {
        Self::new(Assumptions::default())
    }
}
impl Simplifier {
    pub fn new(assumptions: Assumptions) -> Self {
        Self {
            assumptions,
            rules: Vec::new(),
            max_passes: 64,
        }
    }
    pub fn add_rule(&mut self, rule: RewriteRule) {
        self.rules.push(rule);
    }
    pub fn set_max_passes(&mut self, max_passes: usize) {
        self.max_passes = max_passes;
    }

    pub fn simplify(&self, expression: &ScalarExpr) -> Result<ScalarExpr, RewriteError> {
        let mut current = expression.clone();
        for _ in 0..self.max_passes {
            let builtins = self.simplify_builtin(&current)?;
            let rewritten = self.apply_rules_once(&builtins)?;
            if rewritten == current {
                return Ok(rewritten);
            }
            current = rewritten;
        }
        Err(RewriteError::PassLimit(self.max_passes))
    }

    fn simplify_builtin(&self, expr: &ScalarExpr) -> Result<ScalarExpr, RewriteError> {
        match expr {
            ScalarExpr::Exact(_) | ScalarExpr::Symbol(_) | ScalarExpr::Constant(_) => {
                Ok(expr.clone())
            }
            ScalarExpr::Add(xs) => ScalarExpr::add(
                xs.iter()
                    .map(|x| self.simplify_builtin(x))
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .map_err(Into::into),
            ScalarExpr::Mul(xs) => ScalarExpr::mul(
                xs.iter()
                    .map(|x| self.simplify_builtin(x))
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .map_err(Into::into),
            ScalarExpr::Pow(base, exponent) => {
                let base = self.simplify_builtin(base)?;
                if let ScalarExpr::Exact(value) = base {
                    Ok(ScalarExpr::exact(value.checked_pow(*exponent)?))
                } else {
                    ScalarExpr::pow(base, *exponent).map_err(Into::into)
                }
            }
            ScalarExpr::Abs(value) => {
                let inner = self.simplify_builtin(value)?;
                if let ScalarExpr::Exact(exact) = inner {
                    if exact.im.is_zero() {
                        if exact.re.numerator() >= 0 {
                            return Ok(ScalarExpr::Exact(exact));
                        }
                        return Ok(ScalarExpr::Exact(exact.checked_neg()?));
                    }
                    return Ok(ScalarExpr::Abs(Box::new(ScalarExpr::Exact(exact))));
                }
                if let ScalarExpr::Symbol(name) = &inner {
                    match self.assumptions.symbol(name).sign {
                        Sign::Positive | Sign::NonNegative | Sign::Zero => return Ok(inner),
                        Sign::Negative | Sign::NonPositive => {
                            return ScalarExpr::mul([
                                ScalarExpr::exact(ExactComplex::real(Rational::MINUS_ONE)),
                                inner,
                            ])
                            .map_err(Into::into);
                        }
                        Sign::Unknown => {}
                    }
                }
                Ok(ScalarExpr::Abs(Box::new(inner)))
            }
            ScalarExpr::Sqrt(value) => {
                let inner = self.simplify_builtin(value)?;
                if let ScalarExpr::Exact(exact) = inner {
                    if let Some(root) = exact_sqrt(exact)? {
                        return Ok(root);
                    }
                    return Ok(ScalarExpr::Sqrt(Box::new(ScalarExpr::Exact(exact))));
                }
                if let ScalarExpr::Pow(base, 2) = &inner
                    && self.assumptions.is_real(base)
                {
                    return self.simplify_builtin(&ScalarExpr::Abs(base.clone()));
                }
                Ok(ScalarExpr::Sqrt(Box::new(inner)))
            }
            ScalarExpr::Sin(value) => {
                let inner = self.simplify_builtin(value)?;
                if is_exact_zero(&inner) {
                    return Ok(ScalarExpr::zero());
                }
                if let Some(result) = simplify_pi_trig(&inner, true)? {
                    return Ok(result);
                }
                if let Some(positive) = remove_negative_sign(&inner)? {
                    return ScalarExpr::mul([
                        ScalarExpr::exact(ExactComplex::real(Rational::MINUS_ONE)),
                        self.simplify_builtin(&ScalarExpr::sin(positive))?,
                    ])
                    .map_err(Into::into);
                }
                Ok(ScalarExpr::sin(inner))
            }
            ScalarExpr::Cos(value) => {
                let inner = self.simplify_builtin(value)?;
                if is_exact_zero(&inner) {
                    return Ok(ScalarExpr::one());
                }
                if let Some(result) = simplify_pi_trig(&inner, false)? {
                    return Ok(result);
                }
                if let Some(positive) = remove_negative_sign(&inner)? {
                    return self.simplify_builtin(&ScalarExpr::cos(positive));
                }
                Ok(ScalarExpr::cos(inner))
            }
            ScalarExpr::Exp(value) => {
                let inner = self.simplify_builtin(value)?;
                if is_exact_zero(&inner) {
                    return Ok(ScalarExpr::one());
                }
                if inner == ScalarExpr::one() {
                    return Ok(ScalarExpr::e());
                }
                Ok(ScalarExpr::exp(inner))
            }
            ScalarExpr::Log(value) => {
                let inner = self.simplify_builtin(value)?;
                if inner == ScalarExpr::one() {
                    return Ok(ScalarExpr::zero());
                }
                if inner == ScalarExpr::e() {
                    return Ok(ScalarExpr::one());
                }
                if let ScalarExpr::Exp(argument) = &inner
                    && self.assumptions.is_real(argument)
                {
                    return Ok(argument.as_ref().clone());
                }
                Ok(ScalarExpr::log(inner))
            }
        }
    }

    fn apply_rules_once(&self, expr: &ScalarExpr) -> Result<ScalarExpr, RewriteError> {
        let rebuilt = match expr {
            ScalarExpr::Exact(_) | ScalarExpr::Symbol(_) | ScalarExpr::Constant(_) => expr.clone(),
            ScalarExpr::Add(xs) => ScalarExpr::add(
                xs.iter()
                    .map(|x| self.apply_rules_once(x))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
            ScalarExpr::Mul(xs) => ScalarExpr::mul(
                xs.iter()
                    .map(|x| self.apply_rules_once(x))
                    .collect::<Result<Vec<_>, _>>()?,
            )?,
            ScalarExpr::Pow(base, exponent) => {
                ScalarExpr::pow(self.apply_rules_once(base)?, *exponent)?
            }
            ScalarExpr::Abs(value) => ScalarExpr::abs(self.apply_rules_once(value)?),
            ScalarExpr::Sqrt(value) => ScalarExpr::sqrt(self.apply_rules_once(value)?),
            ScalarExpr::Sin(value) => ScalarExpr::sin(self.apply_rules_once(value)?),
            ScalarExpr::Cos(value) => ScalarExpr::cos(self.apply_rules_once(value)?),
            ScalarExpr::Exp(value) => ScalarExpr::exp(self.apply_rules_once(value)?),
            ScalarExpr::Log(value) => ScalarExpr::log(self.apply_rules_once(value)?),
        };
        for rule in &self.rules {
            let mut bindings = HashMap::new();
            if match_pattern(&rule.pattern, &rebuilt, &mut bindings) {
                return Ok(rule.replacement.substitute(&bindings)?);
            }
        }
        Ok(rebuilt)
    }
}

fn match_pattern(
    pattern: &ExprPattern,
    expr: &ScalarExpr,
    bindings: &mut HashMap<String, ScalarExpr>,
) -> bool {
    match (pattern, expr) {
        (ExprPattern::Any(name), value) => match bindings.get(name) {
            Some(previous) => previous == value,
            None => {
                bindings.insert(name.clone(), value.clone());
                true
            }
        },
        (ExprPattern::Exact(a), value) => a == value,
        (ExprPattern::Constant(a), ScalarExpr::Constant(b)) => a == b,
        (ExprPattern::Add(ps), ScalarExpr::Add(xs))
        | (ExprPattern::Mul(ps), ScalarExpr::Mul(xs)) => {
            ps.len() == xs.len()
                && ps
                    .iter()
                    .zip(xs)
                    .all(|(p, x)| match_pattern(p, x, bindings))
        }
        (ExprPattern::Pow(p, pe), ScalarExpr::Pow(x, xe)) => {
            pe == xe && match_pattern(p, x, bindings)
        }
        (ExprPattern::Abs(p), ScalarExpr::Abs(x))
        | (ExprPattern::Sqrt(p), ScalarExpr::Sqrt(x))
        | (ExprPattern::Sin(p), ScalarExpr::Sin(x))
        | (ExprPattern::Cos(p), ScalarExpr::Cos(x))
        | (ExprPattern::Exp(p), ScalarExpr::Exp(x))
        | (ExprPattern::Log(p), ScalarExpr::Log(x)) => match_pattern(p, x, bindings),
        _ => false,
    }
}

fn is_exact_zero(expression: &ScalarExpr) -> bool {
    matches!(expression, ScalarExpr::Exact(value) if value.is_zero())
}

fn remove_negative_sign(expression: &ScalarExpr) -> Result<Option<ScalarExpr>, AnalyticError> {
    let ScalarExpr::Mul(factors) = expression else {
        return Ok(None);
    };
    let Some(ScalarExpr::Exact(coefficient)) = factors.first() else {
        return Ok(None);
    };
    if !coefficient.im.is_zero() || coefficient.re.numerator() >= 0 {
        return Ok(None);
    }
    let positive = ExactComplex::real(coefficient.re.checked_neg()?);
    let rest = &factors[1..];
    let mut rebuilt = Vec::with_capacity(rest.len() + 1);
    if positive != ExactComplex::ONE || rest.is_empty() {
        rebuilt.push(ScalarExpr::Exact(positive));
    }
    rebuilt.extend(rest.iter().cloned());
    Ok(Some(ScalarExpr::mul(rebuilt)?))
}

fn rational_pi_multiple(expression: &ScalarExpr) -> Result<Option<Rational>, AnalyticError> {
    if matches!(expression, ScalarExpr::Constant(NamedConstant::Pi)) {
        return Ok(Some(Rational::ONE));
    }
    let ScalarExpr::Mul(factors) = expression else {
        return Ok(None);
    };
    let mut coefficient = Rational::ONE;
    let mut found_pi = false;
    for factor in factors {
        match factor {
            ScalarExpr::Exact(value) if value.im.is_zero() => {
                coefficient = coefficient.checked_mul(value.re)?;
            }
            ScalarExpr::Constant(NamedConstant::Pi) if !found_pi => found_pi = true,
            _ => return Ok(None),
        }
    }
    Ok(found_pi.then_some(coefficient))
}

fn simplify_pi_trig(
    expression: &ScalarExpr,
    sine: bool,
) -> Result<Option<ScalarExpr>, AnalyticError> {
    let Some(angle) = rational_pi_multiple(expression)? else {
        return Ok(None);
    };
    let exact = |value| ScalarExpr::exact(ExactComplex::real(value));
    let one = Rational::ONE;
    let minus_one = Rational::MINUS_ONE;
    let zero = Rational::ZERO;
    let result = match angle.denominator() {
        1 => {
            if sine {
                Some(exact(zero))
            } else if angle.numerator().rem_euclid(2) == 0 {
                Some(exact(one))
            } else {
                Some(exact(minus_one))
            }
        }
        2 => {
            if sine {
                Some(exact(if angle.numerator().rem_euclid(4) == 1 {
                    one
                } else {
                    minus_one
                }))
            } else {
                Some(exact(zero))
            }
        }
        4 => {
            let residue = angle.numerator().rem_euclid(8);
            let half_sqrt_two = || {
                ScalarExpr::mul([
                    exact(Rational::new(1, 2).expect("nonzero denominator")),
                    ScalarExpr::sqrt(exact(Rational::new(2, 1).expect("nonzero denominator"))),
                ])
                .expect("small exact scalar multiplication cannot overflow")
            };
            let signed_half_sqrt_two = |positive| {
                if positive {
                    half_sqrt_two()
                } else {
                    ScalarExpr::mul([exact(minus_one), half_sqrt_two()])
                        .expect("small exact scalar multiplication cannot overflow")
                }
            };
            if sine {
                match residue {
                    0 | 4 => Some(exact(zero)),
                    2 => Some(exact(one)),
                    6 => Some(exact(minus_one)),
                    1 | 3 => Some(signed_half_sqrt_two(true)),
                    5 | 7 => Some(signed_half_sqrt_two(false)),
                    _ => unreachable!(),
                }
            } else {
                match residue {
                    0 => Some(exact(one)),
                    2 | 6 => Some(exact(zero)),
                    4 => Some(exact(minus_one)),
                    1 | 7 => Some(signed_half_sqrt_two(true)),
                    3 | 5 => Some(signed_half_sqrt_two(false)),
                    _ => unreachable!(),
                }
            }
        }
        _ => None,
    };
    Ok(result)
}

fn exact_sqrt(value: ExactComplex) -> Result<Option<ScalarExpr>, AnalyticError> {
    if !value.im.is_zero() {
        return Ok(None);
    }
    let numerator = value.re.numerator();
    let n_abs = numerator.unsigned_abs();
    let n_root = integer_sqrt(n_abs);
    let d_root = integer_sqrt(value.re.denominator() as u128);
    if n_root * n_root != n_abs || d_root * d_root != value.re.denominator() as u128 {
        return Ok(None);
    }
    let n_root = i128::try_from(n_root).map_err(|_| AnalyticError::Overflow)?;
    let d_root = i128::try_from(d_root).map_err(|_| AnalyticError::Overflow)?;
    let root = Rational::new(n_root, d_root)?;
    if numerator >= 0 {
        Ok(Some(ScalarExpr::exact(ExactComplex::real(root))))
    } else {
        Ok(Some(ScalarExpr::exact(ExactComplex {
            re: Rational::ZERO,
            im: root,
        })))
    }
}
fn integer_sqrt(n: u128) -> u128 {
    if n < 2 {
        return n;
    }
    let (mut low, mut high) = (1u128, 1u128 << 64);
    while low + 1 < high {
        let mid = low + (high - low) / 2;
        if mid <= n / mid {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}
impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::I => 'I',
                Self::X => 'X',
                Self::Y => 'Y',
                Self::Z => 'Z',
            }
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PauliWord(Vec<Pauli>);
impl PauliWord {
    pub fn new(factors: Vec<Pauli>) -> Self {
        Self(factors)
    }
    pub fn factors(&self) -> &[Pauli] {
        &self.0
    }
    pub fn qubits(&self) -> usize {
        self.0.len()
    }
    pub fn multiply(&self, rhs: &Self) -> Result<(ExactComplex, Self), AnalyticError> {
        if self.qubits() != rhs.qubits() {
            return Err(AnalyticError::QubitCountMismatch {
                left: self.qubits(),
                right: rhs.qubits(),
            });
        }
        let mut phase = ExactComplex::ONE;
        let mut product = Vec::with_capacity(self.qubits());
        for (&a, &b) in self.0.iter().zip(&rhs.0) {
            let (factor, local_phase) = pauli_product(a, b);
            phase = phase.checked_mul(local_phase)?;
            product.push(factor);
        }
        Ok((phase, Self(product)))
    }

    pub fn apply_ket(&self, ket: &Ket) -> Result<Ket, LinearError> {
        validate_pauli_space(ket.space(), self.qubits())?;
        let mut output = vec![Complex64::ZERO; ket.space().dimension()];
        for (index, &amplitude) in ket.amplitudes().iter().enumerate() {
            let (destination, phase) = self.apply_basis_index(index);
            output[destination] = output[destination] + phase * amplitude;
        }
        Ket::new(ket.space().clone(), output)
    }

    pub fn apply_sparse_ket(&self, ket: &SparseKet) -> Result<SparseKet, LinearError> {
        validate_pauli_space(ket.space(), self.qubits())?;
        let entries = ket.amplitudes().iter().map(|(&index, &amplitude)| {
            let (destination, phase) = self.apply_basis_index(index);
            (destination, phase * amplitude)
        });
        SparseKet::new(ket.space().clone(), entries)
    }

    fn apply_basis_index(&self, mut index: usize) -> (usize, Complex64) {
        let mut phase = Complex64::ONE;
        for (qubit, pauli) in self.0.iter().enumerate() {
            let mask = 1usize << qubit;
            let bit_is_one = index & mask != 0;
            match pauli {
                Pauli::I => {}
                Pauli::X => index ^= mask,
                Pauli::Y => {
                    phase = phase
                        * if bit_is_one {
                            Complex64::new(0.0, -1.0)
                        } else {
                            Complex64::I
                        };
                    index ^= mask;
                }
                Pauli::Z if bit_is_one => phase = phase * -1.0,
                Pauli::Z => {}
            }
        }
        (index, phase)
    }
}

fn validate_pauli_space(space: &HilbertSpace, qubits: usize) -> Result<(), LinearError> {
    if space
        .subsystem_dimensions()
        .iter()
        .any(|&dimension| dimension != 2)
    {
        return Err(LinearError::NotQubitSpace);
    }
    if space.subsystem_dimensions().len() != qubits {
        return Err(LinearError::DimensionMismatch {
            expected: qubits,
            actual: space.subsystem_dimensions().len(),
        });
    }
    Ok(())
}
impl fmt::Display for PauliWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() || self.0.iter().all(|p| *p == Pauli::I) {
            return write!(f, "I");
        }
        for (index, pauli) in self.0.iter().enumerate() {
            if *pauli != Pauli::I {
                write!(f, "{pauli}{index}")?;
            }
        }
        Ok(())
    }
}
fn pauli_product(a: Pauli, b: Pauli) -> (Pauli, ExactComplex) {
    use Pauli::*;
    match (a, b) {
        (I, p) | (p, I) => (p, ExactComplex::ONE),
        (X, X) | (Y, Y) | (Z, Z) => (I, ExactComplex::ONE),
        (X, Y) => (Z, ExactComplex::I),
        (Y, X) => (
            Z,
            ExactComplex {
                re: Rational::ZERO,
                im: Rational::MINUS_ONE,
            },
        ),
        (Y, Z) => (X, ExactComplex::I),
        (Z, Y) => (
            X,
            ExactComplex {
                re: Rational::ZERO,
                im: Rational::MINUS_ONE,
            },
        ),
        (Z, X) => (Y, ExactComplex::I),
        (X, Z) => (
            Y,
            ExactComplex {
                re: Rational::ZERO,
                im: Rational::MINUS_ONE,
            },
        ),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PauliSum {
    qubits: usize,
    terms: BTreeMap<PauliWord, ExactComplex>,
}
impl PauliSum {
    pub fn zero(qubits: usize) -> Self {
        Self {
            qubits,
            terms: BTreeMap::new(),
        }
    }
    pub fn identity(qubits: usize) -> Self {
        let mut sum = Self::zero(qubits);
        let _ = sum.add_term(ExactComplex::ONE, PauliWord::new(vec![Pauli::I; qubits]));
        sum
    }
    pub fn term(coefficient: ExactComplex, word: PauliWord) -> Result<Self, AnalyticError> {
        let qubits = word.qubits();
        let mut sum = Self::zero(qubits);
        sum.add_term(coefficient, word)?;
        Ok(sum)
    }
    pub fn qubits(&self) -> usize {
        self.qubits
    }
    pub fn terms(&self) -> &BTreeMap<PauliWord, ExactComplex> {
        &self.terms
    }
    pub fn to_sparse_operator(&self) -> Result<SparseOperator, LinearError> {
        let space = HilbertSpace::qubits(self.qubits)?;
        let dimension = space.dimension();
        let mut entries = Vec::with_capacity(self.terms.len().saturating_mul(dimension));
        for (word, coefficient) in &self.terms {
            let coefficient = coefficient.to_complex64();
            for column in 0..dimension {
                let (row, phase) = word.apply_basis_index(column);
                entries.push((row, column, coefficient * phase));
            }
        }
        SparseOperator::new(space.clone(), space, entries)
    }
    pub fn to_sparse_observable(
        &self,
        tolerance: f64,
    ) -> Result<SparseObservable, NumericPauliError> {
        let operator = self
            .to_sparse_operator()
            .map_err(NumericPauliError::LinearAlgebra)?;
        SparseObservable::new(operator, tolerance).map_err(NumericPauliError::LinearAlgebra)
    }
    pub fn expectation_ket(&self, ket: &Ket) -> Result<Complex64, LinearError> {
        let operator = self.to_sparse_operator()?;
        ket.expectation_map(&operator)
    }
    pub fn expectation_sparse_ket(&self, ket: &SparseKet) -> Result<Complex64, LinearError> {
        validate_pauli_space(ket.space(), self.qubits)?;
        let mut value = Complex64::ZERO;
        for (word, coefficient) in &self.terms {
            let transformed = word.apply_sparse_ket(ket)?;
            value = value + coefficient.to_complex64() * ket.inner(&transformed)?;
        }
        Ok(value)
    }
    pub fn add_term(
        &mut self,
        coefficient: ExactComplex,
        word: PauliWord,
    ) -> Result<(), AnalyticError> {
        if word.qubits() != self.qubits {
            return Err(AnalyticError::QubitCountMismatch {
                left: self.qubits,
                right: word.qubits(),
            });
        }
        let updated = self
            .terms
            .get(&word)
            .copied()
            .unwrap_or(ExactComplex::ZERO)
            .checked_add(coefficient)?;
        if updated.is_zero() {
            self.terms.remove(&word);
        } else {
            self.terms.insert(word, updated);
        }
        Ok(())
    }
    pub fn checked_add(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.check_qubits(rhs)?;
        let mut out = self.clone();
        for (word, coefficient) in &rhs.terms {
            out.add_term(*coefficient, word.clone())?;
        }
        Ok(out)
    }
    pub fn checked_sub(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.check_qubits(rhs)?;
        let mut out = self.clone();
        for (word, coefficient) in &rhs.terms {
            out.add_term(coefficient.checked_neg()?, word.clone())?;
        }
        Ok(out)
    }
    pub fn checked_mul(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.check_qubits(rhs)?;
        let mut out = Self::zero(self.qubits);
        for (left_word, left_coefficient) in &self.terms {
            for (right_word, right_coefficient) in &rhs.terms {
                let (phase, word) = left_word.multiply(right_word)?;
                let coefficient = left_coefficient
                    .checked_mul(*right_coefficient)?
                    .checked_mul(phase)?;
                out.add_term(coefficient, word)?;
            }
        }
        Ok(out)
    }
    pub fn commutator(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.checked_mul(rhs)?.checked_sub(&rhs.checked_mul(self)?)
    }
    fn check_qubits(&self, rhs: &Self) -> Result<(), AnalyticError> {
        if self.qubits != rhs.qubits {
            Err(AnalyticError::QubitCountMismatch {
                left: self.qubits,
                right: rhs.qubits,
            })
        } else {
            Ok(())
        }
    }
}
impl fmt::Display for PauliSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (index, (word, coefficient)) in self.terms.iter().enumerate() {
            if index > 0 {
                write!(f, " + ")?;
            }
            write!(f, "({coefficient}) {word}")?;
        }
        Ok(())
    }
}

/// Pauli algebra with symbolic scalar coefficients, such as `a X + b Z`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicPauliSum {
    qubits: usize,
    terms: BTreeMap<PauliWord, ScalarExpr>,
}
impl SymbolicPauliSum {
    pub fn zero(qubits: usize) -> Self {
        Self {
            qubits,
            terms: BTreeMap::new(),
        }
    }
    pub fn term(coefficient: ScalarExpr, word: PauliWord) -> Result<Self, AnalyticError> {
        let qubits = word.qubits();
        let mut sum = Self::zero(qubits);
        sum.add_term(coefficient, word)?;
        Ok(sum)
    }
    pub fn qubits(&self) -> usize {
        self.qubits
    }
    pub fn terms(&self) -> &BTreeMap<PauliWord, ScalarExpr> {
        &self.terms
    }
    pub fn evaluate_sparse_operator(
        &self,
        values: &HashMap<String, Complex64>,
    ) -> Result<SparseOperator, NumericPauliError> {
        let space = HilbertSpace::qubits(self.qubits).map_err(NumericPauliError::LinearAlgebra)?;
        let dimension = space.dimension();
        let mut entries = Vec::with_capacity(self.terms.len().saturating_mul(dimension));
        for (word, coefficient) in &self.terms {
            let coefficient = coefficient
                .evaluate(values)
                .map_err(NumericPauliError::UnboundSymbol)?;
            for column in 0..dimension {
                let (row, phase) = word.apply_basis_index(column);
                entries.push((row, column, coefficient * phase));
            }
        }
        SparseOperator::new(space.clone(), space, entries).map_err(NumericPauliError::LinearAlgebra)
    }
    pub fn evaluate_sparse_observable(
        &self,
        values: &HashMap<String, Complex64>,
        tolerance: f64,
    ) -> Result<SparseObservable, NumericPauliError> {
        let operator = self.evaluate_sparse_operator(values)?;
        SparseObservable::new(operator, tolerance).map_err(NumericPauliError::LinearAlgebra)
    }
    pub fn substitute(&self, values: &HashMap<String, ScalarExpr>) -> Result<Self, AnalyticError> {
        let mut out = Self::zero(self.qubits);
        for (word, coefficient) in &self.terms {
            out.add_term(coefficient.substitute(values)?, word.clone())?;
        }
        Ok(out)
    }
    pub fn add_term(
        &mut self,
        coefficient: ScalarExpr,
        word: PauliWord,
    ) -> Result<(), AnalyticError> {
        if word.qubits() != self.qubits {
            return Err(AnalyticError::QubitCountMismatch {
                left: self.qubits,
                right: word.qubits(),
            });
        }
        let combined = ScalarExpr::add([
            self.terms
                .get(&word)
                .cloned()
                .unwrap_or_else(ScalarExpr::zero),
            coefficient,
        ])?;
        if combined == ScalarExpr::zero() {
            self.terms.remove(&word);
        } else {
            self.terms.insert(word, combined);
        }
        Ok(())
    }
    pub fn checked_add(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.check_qubits(rhs)?;
        let mut out = self.clone();
        for (word, coefficient) in &rhs.terms {
            out.add_term(coefficient.clone(), word.clone())?;
        }
        Ok(out)
    }
    pub fn checked_sub(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.check_qubits(rhs)?;
        let mut out = self.clone();
        for (word, coefficient) in &rhs.terms {
            let neg = ScalarExpr::mul([
                ScalarExpr::exact(ExactComplex::real(Rational::MINUS_ONE)),
                coefficient.clone(),
            ])?;
            out.add_term(neg, word.clone())?;
        }
        Ok(out)
    }
    pub fn checked_mul(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.check_qubits(rhs)?;
        let mut out = Self::zero(self.qubits);
        for (left_word, left_coefficient) in &self.terms {
            for (right_word, right_coefficient) in &rhs.terms {
                let (phase, word) = left_word.multiply(right_word)?;
                let coefficient = ScalarExpr::mul([
                    left_coefficient.clone(),
                    right_coefficient.clone(),
                    ScalarExpr::exact(phase),
                ])?;
                out.add_term(coefficient, word)?;
            }
        }
        Ok(out)
    }
    pub fn commutator(&self, rhs: &Self) -> Result<Self, AnalyticError> {
        self.checked_mul(rhs)?.checked_sub(&rhs.checked_mul(self)?)
    }
    fn check_qubits(&self, rhs: &Self) -> Result<(), AnalyticError> {
        if self.qubits != rhs.qubits {
            Err(AnalyticError::QubitCountMismatch {
                left: self.qubits,
                right: rhs.qubits,
            })
        } else {
            Ok(())
        }
    }
}
impl fmt::Display for SymbolicPauliSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (index, (word, coefficient)) in self.terms.iter().enumerate() {
            if index > 0 {
                write!(f, " + ")?;
            }
            write!(f, "({coefficient}) {word}")?;
        }
        Ok(())
    }
}
