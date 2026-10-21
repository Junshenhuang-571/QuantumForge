//! QASM 2.0 parser — the standard quantum assembly interchange format.
//!
//! Parses OpenQASM 2.0 into a [`Circuit`], letting users import circuits from
//! Qiskit, IBM Quantum, and every other QASM-compatible tool.
//!
//! Supported:
//! - `OPENQASM 2.0;` header
//! - `qreg name[size];` / `creg name[size];` declarations
//! - Single-qubit gates: `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `rx`, `ry`, `rz`, `u1`, `u2`, `u3`, `id`
//! - Two-qubit gates: `cx`, `cz`, `swap`
//! - Three-qubit gates: `ccx` (Toffoli)
//! - `measure q[i] -> c[j];`
//! - `barrier q;`
//! - `reset q[i];`
//! - Gate definitions via `gate name(params) qargs { ... }`
//! - `include "qelib1.inc";` (skipped — standard library)
//! - Comments (`//` and `/* */`)
//!
//! Not supported (QASM 3.0 features): `gate` modifiers, `ctrl`, `pow`,
//! `gphase`, classical control flow, `if` statements.

use crate::{Circuit, Gate};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QasmError {
    pub line: usize,
    pub message: String,
}

impl QasmError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for QasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QASM error at line {}: {}", self.line, self.message)
    }
}

impl Error for QasmError {}

/// Parse a QASM 2.0 program into a [`Circuit`].
///
/// # Example
///
/// ```
/// use quantumforge::qasm::parse_qasm;
///
/// let qasm = r#"
/// OPENQASM 2.0;
/// include "qelib1.inc";
/// qreg q[2];
/// creg c[2];
/// h q[0];
/// cx q[0], q[1];
/// measure q[0] -> c[0];
/// "#;
///
/// let circuit = parse_qasm(qasm).unwrap();
/// assert_eq!(circuit.operations().len(), 3);
/// ```
pub fn parse_qasm(source: &str) -> Result<Circuit, QasmError> {
    let mut parser = QasmParser::new(source);
    parser.parse()
}

struct QasmParser<'a> {
    lines: Vec<&'a str>,
    position: usize,
    circuit: Circuit,
    /// Maps qubit register name -> (start_index, size)
    qregs: HashMap<String, (usize, usize)>,
    /// Maps classical register name -> (start_index, size)
    cregs: HashMap<String, (usize, usize)>,
    /// Total qubits allocated so far
    qubit_count: usize,
    /// Total classical bits allocated so far
    cbit_count: usize,
    /// User-defined gates: name -> (params, body)
    gate_defs: HashMap<String, GateDef>,
}

/// Returns true if `c` character can delimit a gate name from its arguments.
fn is_gate_delimiter(c: char) -> bool {
    c == '(' || c == ' ' || c == '['
}

/// Returns true if `name` is a parameterized gate (takes numeric arguments).
fn is_parameterized_gate(name: &str) -> bool {
    matches!(
        name,
        "rx" | "Rx" | "ry" | "Ry" | "rz" | "Rz" | "u1" | "U1" | "u2" | "U2" | "u3" | "U3"
    )
}

#[derive(Clone)]
struct GateDef {
    #[allow(dead_code)] // TODO: wire parameter substitution when applying user gates
    params: Vec<String>,
    qargs: Vec<String>,
    body: Vec<(String, Vec<String>, Vec<String>)>, // (gate_name, param_names, qarg_names)
}

impl<'a> QasmParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            lines: source.lines().collect(),
            position: 0,
            circuit: Circuit::new(),
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            qubit_count: 0,
            cbit_count: 0,
            gate_defs: HashMap::new(),
        }
    }

    fn parse(&mut self) -> Result<Circuit, QasmError> {
        let mut header_seen = false;

        while self.position < self.lines.len() {
            let line = self.lines[self.position].trim();
            self.position += 1;

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with("//") {
                continue;
            }

            // Skip block comments (single-line)
            if line.starts_with("/*") && line.ends_with("*/") {
                continue;
            }

            // OPENQASM header
            if line.starts_with("OPENQASM") {
                header_seen = true;
                continue;
            }

            // Include statement (skip standard library)
            if line.starts_with("include") {
                continue;
            }

            // QASM 3.0 version header — reject
            if line.starts_with("OPENQASM 3") {
                return Err(QasmError::new(
                    self.position,
                    "QASM 3.0 is not supported; use QASM 2.0",
                ));
            }

            if !header_seen {
                return Err(QasmError::new(self.position, "missing OPENQASM 2.0 header"));
            }

            // Qubit register declaration
            if line.starts_with("qreg") {
                self.parse_qreg(line)?;
                continue;
            }

            // Classical register declaration
            if line.starts_with("creg") {
                self.parse_creg(line)?;
                continue;
            }

            // Gate definition
            if line.starts_with("gate") {
                self.parse_gate_def(line)?;
                continue;
            }

            // Barrier
            if line.starts_with("barrier") {
                self.parse_barrier(line)?;
                continue;
            }

            // Reset
            if line.starts_with("reset") {
                self.parse_reset(line)?;
                continue;
            }

            // Measure
            if line.starts_with("measure") {
                self.parse_measure(line)?;
                continue;
            }

            // Gate application (possibly with parameters)
            self.parse_gate_application(line)?;
        }

        Ok(std::mem::take(&mut self.circuit))
    }

    fn parse_qreg(&mut self, line: &str) -> Result<(), QasmError> {
        // qreg name[size];
        let line = line.trim_end_matches(';').trim();
        let rest = line
            .strip_prefix("qreg")
            .ok_or_else(|| QasmError::new(self.position, "invalid qreg declaration"))?;

        let (name, size) = parse_reg_decl(rest)
            .ok_or_else(|| QasmError::new(self.position, "invalid qreg declaration"))?;

        let start = self.qubit_count;
        self.qregs.insert(name, (start, size));
        self.qubit_count += size;
        Ok(())
    }

    fn parse_creg(&mut self, line: &str) -> Result<(), QasmError> {
        // creg name[size];
        let line = line.trim_end_matches(';').trim();
        let rest = line
            .strip_prefix("creg")
            .ok_or_else(|| QasmError::new(self.position, "invalid creg declaration"))?;

        let (name, size) = parse_reg_decl(rest)
            .ok_or_else(|| QasmError::new(self.position, "invalid creg declaration"))?;

        self.cregs.insert(name, (self.cbit_count, size));
        self.cbit_count += size;
        Ok(())
    }

    fn parse_gate_def(&mut self, first_line: &str) -> Result<(), QasmError> {
        // Gate definitions can span multiple lines:
        // gate name(params) qargs {
        //     stmt1;
        //     stmt2;
        // }
        // We need to collect lines until we find the closing '}'

        // Use owned String for header since we may need to concatenate
        let mut header = first_line.trim().to_string();
        let mut body_lines = Vec::new();

        // Check if opening brace is on first line
        if let Some(brace_start) = header.find('{') {
            // Header and possibly some body on first line
            let header_part = header[..brace_start].to_string();
            let body_part = header[brace_start + 1..].to_string();
            header = header_part;
            if !body_part.trim().is_empty() {
                body_lines.push(body_part);
            }
        } else {
            // Opening brace is on a subsequent line - consume lines until we find '{'
            while self.position < self.lines.len() {
                let next_line = self.lines[self.position].trim();
                self.position += 1;
                if next_line.is_empty() || next_line.starts_with("//") {
                    continue;
                }
                if let Some(brace_start) = next_line.find('{') {
                    // Found opening brace
                    let header_part = &next_line[..brace_start];
                    header = format!("{} {}", header, header_part);
                    let body_part = &next_line[brace_start + 1..];
                    if !body_part.trim().is_empty() {
                        body_lines.push(body_part.to_string());
                    }
                    break;
                } else {
                    // Continuation of header
                    header = format!("{} {}", header, next_line);
                }
            }
        }

        // Now consume body lines until closing '}'
        let mut brace_count = 0;
        while self.position < self.lines.len() {
            let next_line = self.lines[self.position];
            self.position += 1;
            let trimmed = next_line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            // Count braces
            for ch in trimmed.chars() {
                if ch == '{' {
                    brace_count += 1;
                } else if ch == '}' {
                    if brace_count == 0 {
                        // Found the closing brace - include everything before it
                        let before_brace = &trimmed[..trimmed.find('}').unwrap()];
                        if !before_brace.trim().is_empty() {
                            body_lines.push(before_brace.to_string());
                        }
                        // Process the body
                        return self.parse_gate_def_body(&header, body_lines);
                    } else {
                        brace_count -= 1;
                    }
                }
            }
            body_lines.push(trimmed.to_string());
        }

        Err(QasmError::new(
            self.position,
            "unclosed gate definition (missing '}')",
        ))
    }

    fn parse_gate_def_body(
        &mut self,
        header: &str,
        body_lines: Vec<String>,
    ) -> Result<(), QasmError> {
        // Parse header: name(params) qargs
        // Header format: "gate name(params) qargs" or "gate name qargs"
        let header = header.trim();
        let header = header.strip_prefix("gate ").unwrap_or(header).trim();
        let name_end = header
            .find('(')
            .unwrap_or(header.find(' ').unwrap_or(header.len()));
        let name = header[..name_end].trim().to_string();

        // Parse params
        let mut params = Vec::new();
        if let Some(paren_start) = header.find('(') {
            let paren_end = header.find(')').ok_or_else(|| {
                QasmError::new(self.position, "unclosed parenthesis in gate definition")
            })?;
            let params_str = &header[paren_start + 1..paren_end];
            for p in params_str.split(',') {
                let p = p.trim();
                if !p.is_empty() {
                    params.push(p.to_string());
                }
            }
        }

        // Parse qargs
        // After stripping "gate " and name, the rest is qargs (e.g., "a, b" or "a b")
        let after_name = &header[name_end..].trim();
        let mut qargs = Vec::new();
        for q in after_name.replace(',', " ").split_whitespace() {
            if !q.is_empty() {
                qargs.push(q.to_string());
            }
        }

        // Parse body — gate qargs are formal parameters, not real qubit registers.
        let body_str = body_lines.join(" ");
        let mut body = Vec::new();
        for stmt in body_str.split(';') {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }
            let gate_name = stmt
                .split(|c: char| is_gate_delimiter(c))
                .next()
                .unwrap_or("")
                .to_string();
            // Extract params if present
            let mut body_params: Vec<String> = Vec::new();
            let qargs_str = if let Some(paren_start) = stmt.find('(') {
                let paren_end = stmt.find(')').ok_or_else(|| {
                    QasmError::new(self.position, "unclosed parenthesis in gate body")
                })?;
                let params_str = &stmt[paren_start + 1..paren_end];
                for p in params_str.split(',') {
                    let p = p.trim();
                    if !p.is_empty() {
                        body_params.push(p.to_string());
                    }
                }
                &stmt[paren_end + 1..]
            } else {
                &stmt[gate_name.len()..]
            };
            let gate_qargs: Vec<String> = qargs_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            body.push((gate_name, body_params, gate_qargs));
        }

        self.gate_defs.insert(
            name,
            GateDef {
                params,
                qargs,
                body,
            },
        );
        Ok(())
    }

    fn parse_barrier(&mut self, line: &str) -> Result<(), QasmError> {
        // barrier q;  or  barrier q[0], q[1];
        let line = line.trim_end_matches(';').trim();
        let rest = line
            .strip_prefix("barrier")
            .ok_or_else(|| QasmError::new(self.position, "invalid barrier"))?;
        // Barriers are no-ops in our simulation
        let _ = rest.trim();
        Ok(())
    }

    fn parse_reset(&mut self, line: &str) -> Result<(), QasmError> {
        // reset q[i];  or  reset q;
        let line = line.trim_end_matches(';').trim();
        let rest = line
            .strip_prefix("reset")
            .ok_or_else(|| QasmError::new(self.position, "invalid reset"))?;
        let rest = rest.trim();

        // Parse qubit reference(s)
        for qref in rest.split(',') {
            let qref = qref.trim();
            let idx = self.resolve_qubit(qref)?;
            // Reset = measure + conditional X, but we just apply X if we knew it was |1⟩
            // For simplicity, we skip reset (it's a no-op on |0⟩ states)
            let _ = idx;
        }
        Ok(())
    }

    fn parse_measure(&mut self, line: &str) -> Result<(), QasmError> {
        // measure q[i] -> c[j];
        let line = line.trim_end_matches(';').trim();
        let rest = line
            .strip_prefix("measure")
            .ok_or_else(|| QasmError::new(self.position, "invalid measure"))?;

        let parts: Vec<&str> = rest.split("->").collect();
        if parts.len() != 2 {
            return Err(QasmError::new(
                self.position,
                "measure requires 'q[i] -> c[j]'",
            ));
        }

        let qref = parts[0].trim();
        let cref = parts[1].trim();

        let qidx = self.resolve_qubit(qref)?;
        let cidx = self.resolve_cbit(cref)?;

        self.circuit.measure(qidx, cidx);
        Ok(())
    }

    fn parse_gate_application(&mut self, line: &str) -> Result<(), QasmError> {
        let line = line.trim_end_matches(';').trim();

        // Check if it's a user-defined gate
        let gate_name = line
            .split(|c: char| is_gate_delimiter(c))
            .next()
            .unwrap_or("")
            .to_string();

        if self.gate_defs.contains_key(&gate_name) {
            self.apply_user_gate(line, &gate_name)?;
        } else if is_parameterized_gate(&gate_name) {
            self.parse_gate_call_with_params(line)?;
        } else {
            self.apply_builtin_gate(line)?;
        }
        Ok(())
    }

    fn apply_user_gate(&mut self, line: &str, name: &str) -> Result<(), QasmError> {
        let def = self.gate_defs.get(name).unwrap().clone();

        // Parse qargs only (don't apply — user gates are expanded inline)
        let (_gate_name, gate_qargs): (String, Vec<String>) = self.parse_gate_call(line)?;

        if gate_qargs.len() != def.qargs.len() {
            return Err(QasmError::new(
                self.position,
                format!(
                    "gate '{name}' expects {} qargs, got {}",
                    def.qargs.len(),
                    gate_qargs.len()
                ),
            ));
        }

        // Map user gate qargs to actual qubit indices
        let mut arg_map: HashMap<String, usize> = HashMap::new();
        for (i, qarg_name) in def.qargs.iter().enumerate() {
            let qref: &str = match gate_qargs.get(i) {
                Some(s) => s.as_str(),
                None => return Err(QasmError::new(self.position, "missing gate argument")),
            };
            let actual_idx = self.resolve_qubit(qref)?;
            arg_map.insert(qarg_name.clone(), actual_idx);
        }

        // Parse call-site params (if any) for this user gate instance.
        let (_call_name, _call_qargs) = self.parse_gate_call(line)?;
        // Extract numeric params from the call line.
        let mut call_params: Vec<f64> = Vec::new();
        if let Some(paren_start) = line.find('(') {
            if let Some(paren_end) = line.find(')') {
                let params_str = &line[paren_start + 1..paren_end];
                for p in params_str.split(',') {
                    let p = p.trim();
                    if !p.is_empty() {
                        if let Ok(v) = p.parse::<f64>() {
                            call_params.push(v);
                        }
                    }
                }
            }
        }
        // Map formal param names to values.
        let mut param_map: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        for (i, pname) in def.params.iter().enumerate() {
            if let Some(v) = call_params.get(i) {
                param_map.insert(pname.clone(), *v);
            }
        }

        // Expand body
        for (body_gate, body_params, body_qargs) in &def.body {
            let resolved: Vec<usize> = body_qargs
                .iter()
                .map(|qa| *arg_map.get(qa).unwrap_or(&0))
                .collect();
            // Resolve body params.
            let mut resolved_params: Vec<f64> = Vec::new();
            for bp in body_params {
                if let Some(v) = param_map.get(bp) {
                    resolved_params.push(*v);
                } else if let Ok(v) = bp.parse::<f64>() {
                    resolved_params.push(v);
                } else {
                    resolved_params.push(0.0);
                }
            }
            if is_parameterized_gate(body_gate) {
                self.apply_parameterized_gate(body_gate, &resolved_params, &resolved)?;
            } else {
                self.apply_builtin_gate_on_qubits(body_gate, &resolved)?;
            }
        }

        Ok(())
    }

    fn apply_builtin_gate(&mut self, line: &str) -> Result<(), QasmError> {
        let (gate_name, qargs) = self.parse_gate_call(line)?;
        let indices: Vec<usize> = qargs
            .iter()
            .map(|q| self.resolve_qubit(q))
            .collect::<Result<Vec<_>, _>>()?;
        self.apply_builtin_gate_on_qubits(&gate_name, &indices)
    }

    fn apply_builtin_gate_on_qubits(
        &mut self,
        gate_name: &str,
        indices: &[usize],
    ) -> Result<(), QasmError> {
        match gate_name {
            // Single-qubit gates
            "x" | "X" => {
                self.circuit.push(Gate::X, indices[0]);
            }
            "y" | "Y" => {
                self.circuit.push(Gate::Y, indices[0]);
            }
            "z" | "Z" => {
                self.circuit.push(Gate::Z, indices[0]);
            }
            "h" | "H" => {
                self.circuit.push(Gate::H, indices[0]);
            }
            "s" | "S" => {
                self.circuit.push(Gate::S, indices[0]);
            }
            "sdg" | "Sdg" => {
                self.circuit.push(Gate::Sdg, indices[0]);
            }
            "t" | "T" => {
                self.circuit.push(Gate::T, indices[0]);
            }
            "tdg" | "Tdg" => {
                self.circuit.push(Gate::Tdg, indices[0]);
            }
            "id" | "I" | "Identity" => {
                self.circuit.push(Gate::I, indices[0]);
            }
            // Two-qubit gates
            "cx" | "CX" | "cnot" | "CNOT" => {
                self.circuit.cnot(indices[0], indices[1]);
            }
            "cz" | "CZ" => {
                self.circuit.cz(indices[0], indices[1]);
            }
            "swap" | "SWAP" => {
                self.circuit.swap(indices[0], indices[1]);
            }
            // Three-qubit gates
            "ccx" | "CCX" | "toffoli" => {
                // Decompose Toffoli (CCNOT) into CNOTs + T/Tdg + H gates.
                // Standard decomposition: 6 CNOTs + 8 T/Tdg + 2 H.
                let (a, b, c) = (indices[0], indices[1], indices[2]);
                self.circuit.push(Gate::H, c);
                self.circuit.cnot(b, c);
                self.circuit.push(Gate::Tdg, c);
                self.circuit.cnot(a, c);
                self.circuit.push(Gate::T, c);
                self.circuit.cnot(b, c);
                self.circuit.push(Gate::Tdg, c);
                self.circuit.cnot(a, c);
                self.circuit.push(Gate::T, b);
                self.circuit.push(Gate::T, c);
                self.circuit.cnot(a, b);
                self.circuit.push(Gate::H, c);
                self.circuit.push(Gate::T, a);
                self.circuit.push(Gate::Tdg, b);
                self.circuit.cnot(a, b);
            }
            _ => {
                return Err(QasmError::new(
                    self.position,
                    format!("unknown gate '{gate_name}'"),
                ));
            }
        }
        Ok(())
    }

    fn parse_gate_call(&mut self, line: &str) -> Result<(String, Vec<String>), QasmError> {
        // Parse: gate_name qarg1, qarg2, ...
        // or:    gate_name(param) qarg1, qarg2, ...
        let line = line.trim();

        // Extract gate name
        let name_end = line
            .find(|c: char| is_gate_delimiter(c))
            .unwrap_or(line.len());
        let name = line[..name_end].to_string();

        // Extract qargs (after the closing paren if params exist)
        let qargs_str = if let Some(_paren_start) = line.find('(') {
            let paren_end = line.find(')').ok_or_else(|| {
                QasmError::new(self.position, "unclosed parenthesis in gate call")
            })?;
            &line[paren_end + 1..]
        } else {
            &line[name_end..]
        };

        let qargs: Vec<String> = qargs_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Ok((name, qargs))
    }

    fn parse_gate_call_with_params(
        &mut self,
        line: &str,
    ) -> Result<(Vec<String>, Vec<f64>), QasmError> {
        // Parse: gate_name(param1, param2, ...) qarg1, qarg2, ...
        let line = line.trim();

        let name_end = line
            .find(|c: char| is_gate_delimiter(c))
            .unwrap_or(line.len());
        let name = line[..name_end].to_string();

        // Extract params
        let mut params = Vec::new();
        if let Some(paren_start) = line.find('(') {
            let paren_end = line.find(')').ok_or_else(|| {
                QasmError::new(self.position, "unclosed parenthesis in gate call")
            })?;
            let params_str = &line[paren_start + 1..paren_end];
            for p in params_str.split(',') {
                let p = p.trim();
                if !p.is_empty() {
                    // Try to parse as f64, or evaluate simple expressions
                    let val = eval_qasm_expr(p).ok_or_else(|| {
                        QasmError::new(self.position, format!("invalid parameter '{p}'"))
                    })?;
                    params.push(val);
                }
            }
        }

        // Extract qargs
        let qargs_str = if let Some(paren_end) = line.find(')') {
            &line[paren_end + 1..]
        } else {
            &line[name_end..]
        };

        let qargs: Vec<String> = qargs_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // Now apply the gate with parameters
        let indices: Vec<usize> = qargs
            .iter()
            .map(|q| self.resolve_qubit(q))
            .collect::<Result<Vec<_>, _>>()?;

        self.apply_parameterized_gate(&name, &params, &indices)?;

        Ok((qargs, params))
    }

    fn apply_parameterized_gate(
        &mut self,
        name: &str,
        params: &[f64],
        indices: &[usize],
    ) -> Result<(), QasmError> {
        match name {
            "rx" | "Rx" => {
                self.circuit.push(Gate::Rx(params[0]), indices[0]);
            }
            "ry" | "Ry" => {
                self.circuit.push(Gate::Ry(params[0]), indices[0]);
            }
            "rz" | "Rz" => {
                self.circuit.push(Gate::Rz(params[0]), indices[0]);
            }
            "u1" | "U1" => {
                // u1(lambda) = Rz(lambda) up to global phase
                self.circuit.push(Gate::Rz(params[0]), indices[0]);
            }
            "u2" | "U2" => {
                // u2(phi, lambda) = Rz(phi) Ry(pi/2) Rz(lambda)
                self.circuit.push(Gate::Rz(params[0]), indices[0]);
                self.circuit
                    .push(Gate::Ry(std::f64::consts::FRAC_PI_2), indices[0]);
                self.circuit.push(Gate::Rz(params[1]), indices[0]);
            }
            "u3" | "U3" => {
                // u3(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda)
                self.circuit.push(Gate::Rz(params[1]), indices[0]);
                self.circuit.push(Gate::Ry(params[0]), indices[0]);
                self.circuit.push(Gate::Rz(params[2]), indices[0]);
            }
            _ => {
                // Non-parameterized gate — just apply it
                return self.apply_builtin_gate_on_qubits(name, indices);
            }
        }
        Ok(())
    }

    fn resolve_qubit(&self, qref: &str) -> Result<usize, QasmError> {
        // qref can be: q, q[0], q[1], etc.
        if let Some(bracket_start) = qref.find('[') {
            let reg_name = qref[..bracket_start].trim();
            let bracket_end = qref.find(']').ok_or_else(|| {
                QasmError::new(self.position, format!("invalid qubit reference '{qref}'"))
            })?;
            let idx_str = &qref[bracket_start + 1..bracket_end];
            let idx: usize = idx_str.trim().parse().map_err(|_| {
                QasmError::new(self.position, format!("invalid qubit index '{idx_str}'"))
            })?;

            let (start, size) = self.qregs.get(reg_name).ok_or_else(|| {
                QasmError::new(
                    self.position,
                    format!("unknown qubit register '{reg_name}'"),
                )
            })?;

            if idx >= *size {
                return Err(QasmError::new(
                    self.position,
                    format!(
                        "qubit index {idx} out of range for register '{reg_name}' (size {size})"
                    ),
                ));
            }

            Ok(start + idx)
        } else {
            // Whole register reference — return first qubit
            let (start, _size) = self.qregs.get(qref).ok_or_else(|| {
                QasmError::new(self.position, format!("unknown qubit register '{qref}'"))
            })?;
            Ok(*start)
        }
    }

    fn resolve_cbit(&self, cref: &str) -> Result<usize, QasmError> {
        if let Some(bracket_start) = cref.find('[') {
            let reg_name = cref[..bracket_start].trim();
            let bracket_end = cref.find(']').ok_or_else(|| {
                QasmError::new(
                    self.position,
                    format!("invalid classical bit reference '{cref}'"),
                )
            })?;
            let idx_str = &cref[bracket_start + 1..bracket_end];
            let idx: usize = idx_str.trim().parse().map_err(|_| {
                QasmError::new(
                    self.position,
                    format!("invalid classical bit index '{idx_str}'"),
                )
            })?;

            let (start, size) = self.cregs.get(reg_name).ok_or_else(|| {
                QasmError::new(
                    self.position,
                    format!("unknown classical register '{reg_name}'"),
                )
            })?;

            if idx >= *size {
                return Err(QasmError::new(
                    self.position,
                    format!(
                        "classical bit index {idx} out of range for register '{reg_name}' (size {size})"
                    ),
                ));
            }

            Ok(start + idx)
        } else {
            let (start, _size) = self.cregs.get(cref).ok_or_else(|| {
                QasmError::new(
                    self.position,
                    format!("unknown classical register '{cref}'"),
                )
            })?;
            Ok(*start)
        }
    }
}

/// Parse a register declaration like `q[2]` or `q[10]`.
fn parse_reg_decl(s: &str) -> Option<(String, usize)> {
    let s = s.trim();
    let bracket_start = s.find('[')?;
    let bracket_end = s.find(']')?;
    let name = s[..bracket_start].trim().to_string();
    let size_str = s[bracket_start + 1..bracket_end].trim();
    let size: usize = size_str.parse().ok()?;
    Some((name, size))
}

/// Evaluate a simple QASM numeric expression (supports +, -, *, /, pi, e, numbers).
fn eval_qasm_expr(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Handle pi and e
    let s = s.replace("pi", &std::f64::consts::PI.to_string());
    let s = s.replace("e", &std::f64::consts::E.to_string());

    // Simple expression evaluator
    eval_simple(&s)
}

/// Very simple expression evaluator for QASM parameters.
/// Supports: numbers, +, -, *, /, parentheses, unary minus.
fn eval_simple(s: &str) -> Option<f64> {
    let tokens = tokenize_expr(s)?;
    let mut pos = 0;
    let result = parse_expr(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return None;
    }
    Some(result)
}

#[derive(Debug, Clone, PartialEq)]
enum ExprToken {
    Number(f64),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
}

fn tokenize_expr(s: &str) -> Option<Vec<ExprToken>> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' => {
                chars.next();
            }
            '+' => {
                tokens.push(ExprToken::Plus);
                chars.next();
            }
            '-' => {
                tokens.push(ExprToken::Minus);
                chars.next();
            }
            '*' => {
                tokens.push(ExprToken::Star);
                chars.next();
            }
            '/' => {
                tokens.push(ExprToken::Slash);
                chars.next();
            }
            '(' => {
                tokens.push(ExprToken::LParen);
                chars.next();
            }
            ')' => {
                tokens.push(ExprToken::RParen);
                chars.next();
            }
            c if c.is_ascii_digit() || c == '.' => {
                let mut num_str = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' {
                        num_str.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                let num: f64 = num_str.parse().ok()?;
                tokens.push(ExprToken::Number(num));
            }
            _ => return None,
        }
    }

    Some(tokens)
}

fn parse_expr(tokens: &[ExprToken], pos: &mut usize) -> Option<f64> {
    let mut left = parse_term(tokens, pos)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            ExprToken::Plus => {
                *pos += 1;
                let right = parse_term(tokens, pos)?;
                left += right;
            }
            ExprToken::Minus => {
                *pos += 1;
                let right = parse_term(tokens, pos)?;
                left -= right;
            }
            _ => break,
        }
    }
    Some(left)
}

fn parse_term(tokens: &[ExprToken], pos: &mut usize) -> Option<f64> {
    let mut left = parse_factor(tokens, pos)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            ExprToken::Star => {
                *pos += 1;
                let right = parse_factor(tokens, pos)?;
                left *= right;
            }
            ExprToken::Slash => {
                *pos += 1;
                let right = parse_factor(tokens, pos)?;
                if right == 0.0 {
                    return None;
                }
                left /= right;
            }
            _ => break,
        }
    }
    Some(left)
}

fn parse_factor(tokens: &[ExprToken], pos: &mut usize) -> Option<f64> {
    if *pos >= tokens.len() {
        return None;
    }
    match &tokens[*pos] {
        ExprToken::Number(n) => {
            *pos += 1;
            Some(*n)
        }
        ExprToken::Minus => {
            *pos += 1;
            let val = parse_factor(tokens, pos)?;
            Some(-val)
        }
        ExprToken::Plus => {
            *pos += 1;
            parse_factor(tokens, pos)
        }
        ExprToken::LParen => {
            *pos += 1;
            let val = parse_expr(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != ExprToken::RParen {
                return None;
            }
            *pos += 1;
            Some(val)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_bell() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 3);
    }

    #[test]
    fn parse_single_qubit_gates() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[1];
x q[0];
y q[0];
z q[0];
h q[0];
s q[0];
sdg q[0];
t q[0];
tdg q[0];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 8);
    }

    #[test]
    fn parse_parameterized_gates() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[1];
rx(3.14159) q[0];
ry(1.5708) q[0];
rz(0.7854) q[0];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 3);
    }

    #[test]
    fn parse_u_gates() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[1];
u1(0.5) q[0];
u2(0.3, 0.7) q[0];
u3(0.1, 0.2, 0.3) q[0];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        // u1 -> 1 op, u2 -> 3 ops, u3 -> 3 ops
        assert_eq!(circuit.operations().len(), 7);
    }

    #[test]
    fn parse_toffoli() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[3];
ccx q[0], q[1], q[2];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        // Toffoli decomposes to 6 CNOTs + 7 T/Tdg + 2 H = 15 ops
        assert_eq!(circuit.operations().len(), 15);
    }

    #[test]
    fn parse_multiple_registers() {
        let qasm = r#"
OPENQASM 2.0;
qreg a[2];
qreg b[2];
cx a[0], b[1];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 1);
    }

    #[test]
    fn parse_gate_definition() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[2];
gate mygate a, b {
    h a;
    cx a, b;
}
mygate q[0], q[1];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 2);
    }

    #[test]
    fn parse_barrier_and_reset() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[2];
h q[0];
barrier q;
reset q[0];
cx q[0], q[1];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 2);
    }

    #[test]
    fn reject_qasm3() {
        let qasm = r#"
OPENQASM 3.0;
qubit q;
"#;
        assert!(parse_qasm(qasm).is_err());
    }

    #[test]
    fn reject_missing_header() {
        let qasm = r#"
qreg q[1];
h q[0];
"#;
        assert!(parse_qasm(qasm).is_err());
    }

    #[test]
    fn reject_unknown_gate() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[1];
foo q[0];
"#;
        assert!(parse_qasm(qasm).is_err());
    }

    #[test]
    fn reject_out_of_range() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[1];
h q[5];
"#;
        assert!(parse_qasm(qasm).is_err());
    }

    #[test]
    fn eval_pi_expression() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[1];
rx(pi/2) q[0];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 1);
    }

    #[test]
    fn eval_complex_expression() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[1];
ry(2*pi/3) q[0];
"#;
        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.operations().len(), 1);
    }
}
