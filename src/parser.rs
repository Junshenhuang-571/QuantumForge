//! Small parser for Wolfram-style scalar expressions.
//!
//! Supports exact integer/decimal literals, `I`, `Pi`, `E`, symbols, basic
//! arithmetic, implicit multiplication, groups, and scalar function heads.

use crate::analytic::{AnalyticError, ExactComplex, NamedConstant, Rational, ScalarExpr};
use crate::{Circuit, Gate, NoiseChannel};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    offset: usize,
    message: String,
}
impl ParseError {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
    pub fn offset(&self) -> usize {
        self.offset
    }
    pub fn message(&self) -> &str {
        &self.message
    }
}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}
impl Error for ParseError {}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TokenKind {
    Number(String),
    Identifier(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Comma,
}
#[derive(Clone, Debug)]
struct Token {
    kind: TokenKind,
    offset: usize,
}

pub fn parse_scalar_expr(source: &str) -> Result<ScalarExpr, ParseError> {
    let mut parser = Parser {
        tokens: tokenize(source)?,
        position: 0,
        source_len: source.len(),
        depth: 0,
    };
    if parser.tokens.is_empty() {
        return Err(ParseError::new(0, "expected an expression"));
    }
    let expression = parser.parse_expression(0)?;
    if let Some(token) = parser.peek() {
        return Err(ParseError::new(
            token.offset,
            "unexpected token after expression",
        ));
    }
    Ok(expression)
}

/// Parse a small Wolfram-style circuit expression such as
/// `Circuit[H[0], CNOT[0,1], Measure[0,0]]`.
///
/// Supported heads are `I`, `X`, `Y`, `Z`, `H`, `Identity`, `Rx`, `Ry`, `Rz`,
/// `CNOT`, `CZ`, `SWAP`, `Measure`, `IfBit`, `Depolarize`, `AmplitudeDamping`,
/// and `PhaseDamping`. Rotation angles and channel probabilities accept scalar
/// expressions that evaluate to finite real values.
pub fn parse_circuit(source: &str) -> Result<Circuit, ParseError> {
    let tokens = tokenize(source)?;
    let mut parser = CircuitParser {
        source,
        tokens,
        position: 0,
    };
    parser.parse()
}

struct CircuitParser<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    position: usize,
}

impl CircuitParser<'_> {
    fn parse(&mut self) -> Result<Circuit, ParseError> {
        if self.tokens.is_empty() {
            return Err(ParseError::new(0, "expected Circuit[...]"));
        }
        let head = self.next().unwrap();
        if head.kind != TokenKind::Identifier("Circuit".into()) {
            return Err(ParseError::new(head.offset, "expected Circuit[...]"));
        }
        self.expect(TokenKind::LeftBracket, "expected '[' after Circuit")?;
        let mut circuit = Circuit::new();
        if matches!(self.peek(), Some(TokenKind::RightBracket)) {
            self.next();
        } else {
            loop {
                self.parse_operation(&mut circuit)?;
                match self.next() {
                    Some(Token {
                        kind: TokenKind::Comma,
                        ..
                    }) => continue,
                    Some(Token {
                        kind: TokenKind::RightBracket,
                        ..
                    }) => break,
                    Some(token) => {
                        return Err(ParseError::new(
                            token.offset,
                            "expected ',' or ']' after circuit operation",
                        ));
                    }
                    None => {
                        return Err(ParseError::new(
                            self.source.len(),
                            "expected ']' to close circuit",
                        ));
                    }
                }
            }
        }
        if let Some(token) = self.tokens.get(self.position) {
            return Err(ParseError::new(
                token.offset,
                "unexpected token after circuit",
            ));
        }
        Ok(circuit)
    }

    fn parse_operation(&mut self, circuit: &mut Circuit) -> Result<(), ParseError> {
        let head = self
            .next()
            .ok_or_else(|| ParseError::new(self.source.len(), "expected a circuit operation"))?;
        let TokenKind::Identifier(name) = head.kind else {
            return Err(ParseError::new(
                head.offset,
                "expected a circuit operation name",
            ));
        };
        self.expect(TokenKind::LeftBracket, "expected '[' after operation name")?;
        match name.as_str() {
            "I" | "Identity" | "X" | "Y" | "Z" | "H" | "S" | "S†" | "Sdg" | "T" | "T†" | "Tdg" =>
            {
                let target = self.read_index()?;
                self.close_operation()?;
                let gate = fixed_gate(&name)
                    .ok_or_else(|| ParseError::new(head.offset, "unknown fixed gate"))?;
                circuit.push(gate, target);
            }
            "Rx" | "Ry" | "Rz" => {
                let angle = self.read_real()?;
                self.comma()?;
                let target = self.read_index()?;
                self.close_operation()?;
                let gate = match name.as_str() {
                    "Rx" => Gate::Rx(angle),
                    "Ry" => Gate::Ry(angle),
                    _ => Gate::Rz(angle),
                };
                circuit.push(gate, target);
            }
            "CNOT" | "CZ" | "SWAP" => {
                let first = self.read_index()?;
                self.comma()?;
                let second = self.read_index()?;
                self.close_operation()?;
                match name.as_str() {
                    "CNOT" => circuit.cnot(first, second),
                    "CZ" => circuit.cz(first, second),
                    _ => circuit.swap(first, second),
                }
            }
            "Measure" => {
                let target = self.read_index()?;
                self.comma()?;
                let classical_bit = self.read_index()?;
                self.close_operation()?;
                circuit.measure(target, classical_bit);
            }
            "IfBit" => {
                let classical_bit = self.read_index()?;
                self.comma()?;
                let value = self.read_index()?;
                if value > 1 {
                    return Err(ParseError::new(head.offset, "IfBit value must be 0 or 1"));
                }
                self.comma()?;
                let (gate_name, offset) = self.read_identifier()?;
                let gate = fixed_gate(&gate_name).ok_or_else(|| {
                    ParseError::new(offset, "IfBit supports only fixed one-qubit gates")
                })?;
                self.comma()?;
                let target = self.read_index()?;
                self.close_operation()?;
                circuit.if_classical_bit(classical_bit, value as u8, gate, target);
            }
            "Depolarize" => {
                let target = self.read_index()?;
                self.comma()?;
                let probability = self.read_real()?;
                self.close_operation()?;
                circuit.channel(target, NoiseChannel::Depolarizing { probability });
            }
            "AmplitudeDamping" | "PhaseDamping" => {
                let target = self.read_index()?;
                self.comma()?;
                let gamma = self.read_real()?;
                self.close_operation()?;
                let channel = if name == "AmplitudeDamping" {
                    NoiseChannel::AmplitudeDamping { gamma }
                } else {
                    NoiseChannel::PhaseDamping { gamma }
                };
                circuit.channel(target, channel);
            }
            _ => {
                return Err(ParseError::new(
                    head.offset,
                    format!("unknown circuit operation '{name}'"),
                ));
            }
        }
        Ok(())
    }

    fn read_index(&mut self) -> Result<usize, ParseError> {
        let offset = self
            .peek_token()
            .map_or(self.source.len(), |token| token.offset);
        let expression = self.read_scalar()?;
        let ScalarExpr::Exact(value) = expression else {
            return Err(ParseError::new(
                offset,
                "index must be an exact nonnegative integer",
            ));
        };
        if !value.im.is_zero() || value.re.denominator() != 1 || value.re.numerator() < 0 {
            return Err(ParseError::new(
                offset,
                "index must be an exact nonnegative integer",
            ));
        }
        usize::try_from(value.re.numerator())
            .map_err(|_| ParseError::new(offset, "index is too large"))
    }

    fn read_real(&mut self) -> Result<f64, ParseError> {
        let offset = self
            .peek_token()
            .map_or(self.source.len(), |token| token.offset);
        let expression = self.read_scalar()?;
        let value = expression
            .evaluate(&HashMap::new())
            .map_err(|error| ParseError::new(offset, error.to_string()))?;
        if value.im != 0.0 || !value.re.is_finite() {
            return Err(ParseError::new(
                offset,
                "value must evaluate to a finite real number",
            ));
        }
        Ok(value.re)
    }

    fn read_scalar(&mut self) -> Result<ScalarExpr, ParseError> {
        let Some(first) = self.peek_token() else {
            return Err(ParseError::new(self.source.len(), "expected an argument"));
        };
        let start = first.offset;
        let mut parentheses = 0usize;
        let mut brackets = 0usize;
        let mut end_position = self.position;
        while let Some(token) = self.tokens.get(end_position) {
            match token.kind {
                TokenKind::LeftParen => parentheses += 1,
                TokenKind::RightParen => {
                    if parentheses == 0 {
                        return Err(ParseError::new(token.offset, "unmatched ')' in argument"));
                    }
                    parentheses -= 1;
                }
                TokenKind::LeftBracket => brackets += 1,
                TokenKind::RightBracket if brackets > 0 => brackets -= 1,
                TokenKind::RightBracket if parentheses == 0 => break,
                TokenKind::Comma if parentheses == 0 && brackets == 0 => break,
                _ => {}
            }
            end_position += 1;
        }
        if parentheses != 0 || brackets != 0 {
            return Err(ParseError::new(
                self.source.len(),
                "unclosed group in argument",
            ));
        }
        let end = self
            .tokens
            .get(end_position)
            .map_or(self.source.len(), |token| token.offset);
        if start == end {
            return Err(ParseError::new(start, "expected an argument"));
        }
        let expression = parse_scalar_expr(&self.source[start..end])
            .map_err(|error| ParseError::new(start + error.offset(), error.message()))?;
        self.position = end_position;
        Ok(expression)
    }

    fn read_identifier(&mut self) -> Result<(String, usize), ParseError> {
        let token = self
            .next()
            .ok_or_else(|| ParseError::new(self.source.len(), "expected a gate name"))?;
        match token.kind {
            TokenKind::Identifier(name) => Ok((name, token.offset)),
            _ => Err(ParseError::new(token.offset, "expected a gate name")),
        }
    }

    fn comma(&mut self) -> Result<(), ParseError> {
        self.expect(TokenKind::Comma, "expected ',' between operation arguments")
    }

    fn close_operation(&mut self) -> Result<(), ParseError> {
        self.expect(
            TokenKind::RightBracket,
            "expected ']' after operation arguments",
        )
    }

    fn expect(&mut self, expected: TokenKind, message: &str) -> Result<(), ParseError> {
        match self.next() {
            Some(token) if token.kind == expected => Ok(()),
            Some(token) => Err(ParseError::new(token.offset, message)),
            None => Err(ParseError::new(self.source.len(), message)),
        }
    }

    fn peek(&self) -> Option<&TokenKind> {
        self.peek_token().map(|token| &token.kind)
    }

    fn peek_token(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        self.position += usize::from(token.is_some());
        token
    }
}

fn fixed_gate(name: &str) -> Option<Gate> {
    match name {
        "I" | "Identity" => Some(Gate::I),
        "X" => Some(Gate::X),
        "Y" => Some(Gate::Y),
        "Z" => Some(Gate::Z),
        "H" => Some(Gate::H),
        "S" => Some(Gate::S),
        "S†" | "Sdg" => Some(Gate::Sdg),
        "T" => Some(Gate::T),
        "T†" | "Tdg" => Some(Gate::Tdg),
        _ => None,
    }
}

struct Parser {
    tokens: Vec<Token>,
    position: usize,
    source_len: usize,
    /// Current recursion depth, to bound stack use on deeply nested input.
    depth: usize,
}

/// Maximum expression nesting depth before the parser bails out. Deep enough
/// for any realistic expression, shallow enough that recursive descent cannot
/// overflow the stack (a pasted `((((…))))` returns an error, not a crash).
const MAX_PARSE_DEPTH: usize = 256;
impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }
    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.position).cloned();
        self.position += usize::from(token.is_some());
        token
    }
    fn parse_expression(&mut self, minimum_bp: u8) -> Result<ScalarExpr, ParseError> {
        // Bound recursion so deeply nested input returns an error, not a
        // stack overflow.
        if self.depth >= MAX_PARSE_DEPTH {
            return Err(ParseError::new(
                self.source_len,
                format!("expression nests deeper than {MAX_PARSE_DEPTH} levels"),
            ));
        }
        self.depth += 1;
        let result = self.parse_expression_inner(minimum_bp);
        self.depth -= 1;
        result
    }

    fn parse_expression_inner(&mut self, minimum_bp: u8) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_prefix()?;
        loop {
            let (left_bp, right_bp, operation) = match self.peek().map(|token| &token.kind) {
                Some(TokenKind::Plus) => (10, 11, Binary::Add),
                Some(TokenKind::Minus) => (10, 11, Binary::Subtract),
                Some(TokenKind::Star) => (20, 21, Binary::Multiply),
                Some(TokenKind::Slash) => (20, 21, Binary::Divide),
                Some(TokenKind::Caret) => (30, 30, Binary::Power),
                Some(kind) if begins_primary(kind) => (20, 21, Binary::Multiply),
                _ => break,
            };
            if left_bp < minimum_bp {
                break;
            }
            let explicit = matches!(
                self.peek().map(|token| &token.kind),
                Some(
                    TokenKind::Plus
                        | TokenKind::Minus
                        | TokenKind::Star
                        | TokenKind::Slash
                        | TokenKind::Caret
                )
            );
            if explicit {
                self.next();
            }
            let right = self.parse_expression(right_bp)?;
            left = match operation {
                Binary::Add => ScalarExpr::add([left, right]),
                Binary::Subtract => ScalarExpr::subtract(left, right),
                Binary::Multiply => ScalarExpr::mul([left, right]),
                Binary::Divide => divide(left, right),
                Binary::Power => {
                    let exponent = integer_exponent(&right).ok_or_else(|| {
                        ParseError::new(
                            self.previous_offset(),
                            "an exponent must be an exact integer representable as i32",
                        )
                    })?;
                    ScalarExpr::pow(left, exponent)
                }
            }
            .map_err(|error| self.arithmetic_error(error))?;
        }
        Ok(left)
    }
    fn parse_prefix(&mut self) -> Result<ScalarExpr, ParseError> {
        let token = self
            .next()
            .ok_or_else(|| ParseError::new(self.source_len, "expected an expression"))?;
        match token.kind {
            TokenKind::Number(value) => parse_number(&value, token.offset),
            TokenKind::Identifier(name) => self.parse_identifier(name, token.offset),
            TokenKind::Plus => self.parse_expression(25),
            TokenKind::Minus => {
                let value = self.parse_expression(25)?;
                ScalarExpr::mul([
                    ScalarExpr::exact(ExactComplex::real(Rational::MINUS_ONE)),
                    value,
                ])
                .map_err(|error| self.arithmetic_error(error))
            }
            TokenKind::LeftParen => {
                let value = self.parse_expression(0)?;
                self.expect(TokenKind::RightParen, "expected ')' to close group")?;
                Ok(value)
            }
            _ => Err(ParseError::new(
                token.offset,
                "expected a number, symbol, or group",
            )),
        }
    }
    fn parse_identifier(&mut self, name: String, offset: usize) -> Result<ScalarExpr, ParseError> {
        let constant = match name.as_str() {
            "I" => Some(ScalarExpr::exact(ExactComplex::I)),
            "Pi" => Some(ScalarExpr::Constant(NamedConstant::Pi)),
            "E" => Some(ScalarExpr::Constant(NamedConstant::E)),
            _ => None,
        };
        if let Some(value) = constant {
            if matches!(
                self.peek().map(|token| &token.kind),
                Some(TokenKind::LeftBracket)
            ) {
                return Err(ParseError::new(
                    offset,
                    "constants cannot be called as functions",
                ));
            }
            return Ok(value);
        }
        if !matches!(
            self.peek().map(|token| &token.kind),
            Some(TokenKind::LeftBracket)
        ) {
            return Ok(ScalarExpr::symbol(name));
        }
        self.next();
        let argument = self.parse_expression(0)?;
        self.expect(
            TokenKind::RightBracket,
            "expected ']' after function argument",
        )?;
        match name.as_str() {
            "Abs" => Ok(ScalarExpr::abs(argument)),
            "Sqrt" => Ok(ScalarExpr::sqrt(argument)),
            "Sin" => Ok(ScalarExpr::sin(argument)),
            "Cos" => Ok(ScalarExpr::cos(argument)),
            "Exp" => Ok(ScalarExpr::exp(argument)),
            "Log" => Ok(ScalarExpr::log(argument)),
            _ => Err(ParseError::new(
                offset,
                format!("unknown scalar function '{name}'"),
            )),
        }
    }
    fn expect(&mut self, expected: TokenKind, message: &str) -> Result<(), ParseError> {
        match self.next() {
            Some(token) if token.kind == expected => Ok(()),
            Some(token) => Err(ParseError::new(token.offset, message)),
            None => Err(ParseError::new(self.source_len, message)),
        }
    }
    fn previous_offset(&self) -> usize {
        self.position
            .checked_sub(1)
            .and_then(|p| self.tokens.get(p))
            .map_or(self.source_len, |token| token.offset)
    }
    fn arithmetic_error(&self, error: AnalyticError) -> ParseError {
        ParseError::new(self.previous_offset(), error.to_string())
    }
}

#[derive(Clone, Copy)]
enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}
fn begins_primary(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Number(_) | TokenKind::Identifier(_) | TokenKind::LeftParen
    )
}
fn integer_exponent(expression: &ScalarExpr) -> Option<i32> {
    let ScalarExpr::Exact(value) = expression else {
        return None;
    };
    if !value.im.is_zero() || value.re.denominator() != 1 {
        return None;
    }
    i32::try_from(value.re.numerator()).ok()
}
fn divide(left: ScalarExpr, right: ScalarExpr) -> Result<ScalarExpr, AnalyticError> {
    if let (ScalarExpr::Exact(a), ScalarExpr::Exact(b)) = (&left, &right) {
        return Ok(ScalarExpr::exact(a.checked_div(*b)?));
    }
    ScalarExpr::mul([left, ScalarExpr::pow(right, -1)?])
}
fn parse_number(value: &str, offset: usize) -> Result<ScalarExpr, ParseError> {
    let parse_integer = |digits: &str| -> Result<i128, ParseError> {
        if digits.is_empty() {
            Ok(0)
        } else {
            digits
                .parse::<i128>()
                .map_err(|_| ParseError::new(offset, "numeric literal is outside i128 range"))
        }
    };
    let (numerator, denominator) = if let Some((whole, fractional)) = value.split_once('.') {
        let numerator = parse_integer(&format!("{whole}{fractional}"))?;
        let denominator = 10i128
            .checked_pow(fractional.len() as u32)
            .ok_or_else(|| ParseError::new(offset, "decimal literal precision is too large"))?;
        (numerator, denominator)
    } else {
        (parse_integer(value)?, 1)
    };
    let rational = Rational::new(numerator, denominator)
        .map_err(|error| ParseError::new(offset, error.to_string()))?;
    Ok(ScalarExpr::exact(ExactComplex::real(rational)))
}

fn tokenize(source: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if byte.is_ascii_whitespace() {
            cursor += 1;
            continue;
        }
        let offset = cursor;
        let punctuation = match byte {
            b'+' => Some(TokenKind::Plus),
            b'-' => Some(TokenKind::Minus),
            b'*' => Some(TokenKind::Star),
            b'/' => Some(TokenKind::Slash),
            b'^' => Some(TokenKind::Caret),
            b'(' => Some(TokenKind::LeftParen),
            b')' => Some(TokenKind::RightParen),
            b'[' => Some(TokenKind::LeftBracket),
            b']' => Some(TokenKind::RightBracket),
            b',' => Some(TokenKind::Comma),
            _ => None,
        };
        if let Some(kind) = punctuation {
            tokens.push(Token { kind, offset });
            cursor += 1;
            continue;
        }
        if byte.is_ascii_digit()
            || (byte == b'.' && bytes.get(cursor + 1).is_some_and(u8::is_ascii_digit))
        {
            let mut saw_dot = false;
            while cursor < bytes.len() && (bytes[cursor].is_ascii_digit() || bytes[cursor] == b'.')
            {
                if bytes[cursor] == b'.' && saw_dot {
                    return Err(ParseError::new(
                        cursor,
                        "a numeric literal cannot contain two decimal points",
                    ));
                }
                saw_dot |= bytes[cursor] == b'.';
                cursor += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Number(source[offset..cursor].to_string()),
                offset,
            });
            continue;
        }
        if byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$') {
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric()
                    || matches!(bytes[cursor], b'_' | b'$' | b'`'))
            {
                cursor += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Identifier(source[offset..cursor].to_string()),
                offset,
            });
            continue;
        }
        return Err(ParseError::new(
            offset,
            format!("unsupported character {:?}", byte as char),
        ));
    }
    Ok(tokens)
}
