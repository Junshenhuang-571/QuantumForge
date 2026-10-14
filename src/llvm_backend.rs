//! Optional textual LLVM IR lowering and command-line backend integration.
//!
//! The core crate does not link LLVM. This module emits target-independent IR
//! and can drive `opt` and `llc` when those executables are installed.

use crate::analytic::ScalarExpr;
use crate::numeric_ir::{NumericInstruction, NumericIrError, NumericProgram, Register};
use std::error::Error;
use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub enum LlvmError {
    InvalidOptimizationLevel(u8),
    ToolUnavailable(String),
    ToolFailed {
        tool: String,
        diagnostics: String,
    },
    RustcUnavailable(String),
    RustcFailed {
        compiler: String,
        diagnostics: String,
    },
    DynamicLibraryUnavailable(String),
    DynamicLibrarySymbolMissing,
    UnsupportedDynamicLibraryPlatform,
    Io(std::io::Error),
}

impl fmt::Display for LlvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptimizationLevel(level) => {
                write!(
                    f,
                    "LLVM optimization level must be between 0 and 3, got {level}"
                )
            }
            Self::ToolUnavailable(tool) => write!(f, "LLVM tool '{tool}' could not be started"),
            Self::ToolFailed { tool, diagnostics } => {
                write!(f, "LLVM tool '{tool}' failed: {diagnostics}")
            }
            Self::RustcUnavailable(compiler) => {
                write!(f, "Rust compiler '{compiler}' could not be started")
            }
            Self::RustcFailed {
                compiler,
                diagnostics,
            } => write!(f, "Rust compiler '{compiler}' failed: {diagnostics}"),
            Self::DynamicLibraryUnavailable(message) => {
                write!(f, "could not load compiled numeric module: {message}")
            }
            Self::DynamicLibrarySymbolMissing => {
                write!(f, "compiled numeric module does not export its entry point")
            }
            Self::UnsupportedDynamicLibraryPlatform => {
                write!(f, "runtime native modules are unsupported on this platform")
            }
            Self::Io(error) => write!(f, "LLVM backend I/O error: {error}"),
        }
    }
}
impl Error for LlvmError {}
impl From<std::io::Error> for LlvmError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// Lower a validated scalar program to LLVM IR using one `double` argument per
/// numeric input. The function is named `quantumforge_eval` and returns `double`.
pub fn emit_llvm_ir(program: &NumericProgram) -> String {
    let mut ir = String::from(
        "; QuantumForge numeric IR lowering\n; requires LLVM 10 or newer\n\
declare double @llvm.sin.f64(double)\n\
declare double @llvm.cos.f64(double)\n\
declare double @llvm.exp.f64(double)\n\
declare double @llvm.log.f64(double)\n\
declare double @llvm.fabs.f64(double)\n\
declare double @llvm.sqrt.f64(double)\n\
declare double @llvm.powi.f64.i32(double, i32)\n",
    );
    let mut arguments = Vec::with_capacity(program.input_count());
    for i in 0..program.input_count() {
        arguments.push(format!("double %input{i}"));
    }
    ir.push_str(&format!(
        "define double @quantumforge_eval({}) {{\nentry:\n",
        arguments.join(", ")
    ));

    for (index, instruction) in program.instructions().iter().enumerate() {
        let output = format!("%v{index}");
        match *instruction {
            NumericInstruction::Constant(value) => {
                ir.push_str(&format!(
                    "  {output} = select i1 true, double {}, double undef\n",
                    llvm_float(value)
                ));
            }
            NumericInstruction::Input(input) => {
                ir.push_str(&format!(
                    "  {output} = select i1 true, double %input{input}, double undef\n"
                ));
            }
            NumericInstruction::Add(a, b) => emit_binary(&mut ir, &output, "fadd", a, b),
            NumericInstruction::Subtract(a, b) => emit_binary(&mut ir, &output, "fsub", a, b),
            NumericInstruction::Multiply(a, b) => emit_binary(&mut ir, &output, "fmul", a, b),
            NumericInstruction::Divide(a, b) => emit_binary(&mut ir, &output, "fdiv", a, b),
            NumericInstruction::PowerI(value, exponent) => {
                ir.push_str(&format!(
                    "  {output} = call double @llvm.powi.f64.i32(double {}, i32 {exponent})\n",
                    value_name(value)
                ));
            }
            NumericInstruction::Negate(value) => {
                ir.push_str(&format!("  {output} = fneg double {}\n", value_name(value)))
            }
            NumericInstruction::Abs(value) => emit_intrinsic(&mut ir, &output, "fabs", value),
            NumericInstruction::Sqrt(value) => emit_intrinsic(&mut ir, &output, "sqrt", value),
            NumericInstruction::Sin(value) => emit_intrinsic(&mut ir, &output, "sin", value),
            NumericInstruction::Cos(value) => emit_intrinsic(&mut ir, &output, "cos", value),
            NumericInstruction::Exp(value) => emit_intrinsic(&mut ir, &output, "exp", value),
            NumericInstruction::Log(value) => emit_intrinsic(&mut ir, &output, "log", value),
        }
    }
    ir.push_str(&format!(
        "  ret double %v{}\n}}\n",
        program.output().index()
    ));
    ir
}

/// Compile a supported real scalar expression directly to LLVM IR.
pub fn lower_scalar_to_llvm_ir(
    expression: &ScalarExpr,
    input_symbols: &[&str],
) -> Result<String, NumericIrError> {
    let program = NumericProgram::compile_scalar(expression, input_symbols)?;
    Ok(emit_llvm_ir(&program))
}

/// Run `opt` on textual LLVM IR and return the optimized IR.
pub fn optimize_ir(ir: &str, opt_executable: &str, level: u8) -> Result<String, LlvmError> {
    if level > 3 {
        return Err(LlvmError::InvalidOptimizationLevel(level));
    }
    let child = Command::new(opt_executable)
        .args([
            "-S",
            &format!("-passes=default<O{level}>"),
            "-verify-each",
            "-o",
            "-",
        ])
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| LlvmError::ToolUnavailable(opt_executable.to_owned()))?;
    let output = run_with_ir(child, ir.as_bytes(), opt_executable)?;
    if !output.status.success() {
        return Err(LlvmError::ToolFailed {
            tool: opt_executable.to_owned(),
            diagnostics: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    String::from_utf8(output.stdout).map_err(|error| LlvmError::ToolFailed {
        tool: opt_executable.to_owned(),
        diagnostics: format!("`opt` returned non-UTF-8 IR: {error}"),
    })
}

/// Compile textual LLVM IR to an object file with `llc` at optimization level
/// 0 through 3. The caller chooses the destination path and target triple.
pub fn compile_object(
    ir: &str,
    llc_executable: &str,
    output: impl AsRef<Path>,
    level: u8,
) -> Result<(), LlvmError> {
    if level > 3 {
        return Err(LlvmError::InvalidOptimizationLevel(level));
    }
    let output = output.as_ref();
    let result = Command::new(llc_executable)
        .arg(format!("-O={level}"))
        .args(["-filetype=obj", "-o"])
        .arg(output)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn();
    let child = result.map_err(|_| LlvmError::ToolUnavailable(llc_executable.to_owned()))?;
    let output_result = run_with_ir(child, ir.as_bytes(), llc_executable)?;
    if !output_result.status.success() {
        return Err(LlvmError::ToolFailed {
            tool: llc_executable.to_owned(),
            diagnostics: String::from_utf8_lossy(&output_result.stderr).into_owned(),
        });
    }
    Ok(())
}

/// Compile a numeric SSA program at runtime into an optimized native module
/// using `rustc`'s LLVM code generator, then load it for repeated execution.
/// This path works without separately installed `opt`, `llc`, or LLVM DLLs.
pub fn compile_jit(
    program: &NumericProgram,
    level: u8,
) -> Result<CompiledNumericProgram, LlvmError> {
    compile_jit_with_rustc(program, "rustc", level)
}

/// Variant of `compile_jit` with an explicit Rust compiler executable.
pub fn compile_jit_with_rustc(
    program: &NumericProgram,
    rustc_executable: &str,
    level: u8,
) -> Result<CompiledNumericProgram, LlvmError> {
    if level > 3 {
        return Err(LlvmError::InvalidOptimizationLevel(level));
    }
    let artifact = JitArtifact::create()?;
    let source_path = artifact.path.join("numeric_kernel.rs");
    let library_path = artifact.path.join(dynamic_library_name());
    fs::write(&source_path, emit_rust_kernel(program))?;
    let output = Command::new(rustc_executable)
        .args([
            "--crate-name",
            "quantumforge_numeric_kernel",
            "--crate-type",
            "cdylib",
            "--edition",
            "2024",
            "-C",
            &format!("opt-level={level}"),
            "-o",
        ])
        .arg(&library_path)
        .arg(&source_path)
        .output()
        .map_err(|_| LlvmError::RustcUnavailable(rustc_executable.to_owned()))?;
    if !output.status.success() {
        return Err(LlvmError::RustcFailed {
            compiler: rustc_executable.to_owned(),
            diagnostics: compiler_diagnostics(&output),
        });
    }

    let library = DynamicLibrary::open(&library_path)?;
    let symbol = library
        .symbol("quantumforge_eval")
        .ok_or(LlvmError::DynamicLibrarySymbolMissing)?;
    let entry = unsafe { std::mem::transmute::<*mut c_void, NumericEntry>(symbol) };
    Ok(CompiledNumericProgram {
        // Drop the library before deleting the file, which is required on Windows.
        library,
        artifact,
        entry,
        input_count: program.input_count(),
    })
}

type NumericEntry = unsafe extern "C" fn(*const f64, usize, *mut f64) -> c_int;

/// A loaded native version of a validated `NumericProgram`.
pub struct CompiledNumericProgram {
    library: DynamicLibrary,
    artifact: JitArtifact,
    entry: NumericEntry,
    input_count: usize,
}

impl CompiledNumericProgram {
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Evaluate compiled code. Domain and arithmetic errors map to the same
    /// `NumericIrError` categories as the reference interpreter.
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
        let mut output = f64::NAN;
        let status = unsafe { (self.entry)(inputs.as_ptr(), inputs.len(), &mut output) };
        match status {
            0 if output.is_finite() => Ok(output),
            1 => Err(NumericIrError::InvalidInputCount {
                expected: self.input_count,
                actual: inputs.len(),
            }),
            2 => Err(NumericIrError::DivisionByZero),
            _ => Err(NumericIrError::NonFiniteValue),
        }
    }

    /// Path of the temporary native module. It exists for the lifetime of this
    /// handle and is removed when the handle is dropped.
    pub fn module_path(&self) -> &Path {
        let _keep_library_alive = &self.library;
        &self.artifact.library_path
    }
}

fn emit_rust_kernel(program: &NumericProgram) -> String {
    let mut source = format!(
        "#[unsafe(no_mangle)]\npub unsafe extern \"C\" fn quantumforge_eval(inputs: *const f64, input_count: usize, output: *mut f64) -> i32 {{\n    if input_count != {} {{ return 1; }}\n    if output.is_null() {{ return 3; }}\n    if input_count != 0 {{\n        if inputs.is_null() {{ return 3; }}\n        let input_slice = unsafe {{ core::slice::from_raw_parts(inputs, input_count) }};\n        if input_slice.iter().any(|value| !value.is_finite()) {{ return 3; }}\n    }}\n",
        program.input_count()
    );
    for (index, instruction) in program.instructions().iter().enumerate() {
        let output = format!("v{index}");
        let expression = match *instruction {
            NumericInstruction::Constant(value) => {
                format!("f64::from_bits(0x{:016x}_u64)", value.to_bits())
            }
            NumericInstruction::Input(input) => format!("unsafe {{ *inputs.add({input}) }}"),
            NumericInstruction::Add(a, b) => {
                format!("{} + {}", rust_value_name(a), rust_value_name(b))
            }
            NumericInstruction::Subtract(a, b) => {
                format!("{} - {}", rust_value_name(a), rust_value_name(b))
            }
            NumericInstruction::Multiply(a, b) => {
                format!("{} * {}", rust_value_name(a), rust_value_name(b))
            }
            NumericInstruction::Divide(a, b) => format!(
                "{{ let denominator = {}; if denominator == 0.0 {{ return 2; }} {} / denominator }}",
                rust_value_name(b),
                rust_value_name(a)
            ),
            NumericInstruction::PowerI(value, exponent) => {
                format!("{}.powi({exponent})", rust_value_name(value))
            }
            NumericInstruction::Negate(value) => format!("-{}", rust_value_name(value)),
            NumericInstruction::Abs(value) => format!("{}.abs()", rust_value_name(value)),
            NumericInstruction::Sqrt(value) => format!("{}.sqrt()", rust_value_name(value)),
            NumericInstruction::Sin(value) => format!("{}.sin()", rust_value_name(value)),
            NumericInstruction::Cos(value) => format!("{}.cos()", rust_value_name(value)),
            NumericInstruction::Exp(value) => format!("{}.exp()", rust_value_name(value)),
            NumericInstruction::Log(value) => format!("{}.ln()", rust_value_name(value)),
        };
        source.push_str(&format!(
            "    let {output} = {expression};\n    if !{output}.is_finite() {{ return 3; }}\n"
        ));
    }
    source.push_str(&format!(
        "    unsafe {{ *output = {}; }}\n    0\n}}\n",
        rust_value_name(program.output())
    ));
    source
}

fn rust_value_name(register: Register) -> String {
    format!("v{}", register.index())
}

fn dynamic_library_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "quantumforge_numeric_kernel.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libquantumforge_numeric_kernel.dylib"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "libquantumforge_numeric_kernel.so"
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        "quantumforge_numeric_kernel.native"
    }
}

fn compiler_diagnostics(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stderr.is_empty() {
        stdout.into_owned()
    } else if stdout.is_empty() {
        stderr.into_owned()
    } else {
        format!("{stderr}\n{stdout}")
    }
}

static JIT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct JitArtifact {
    path: PathBuf,
    library_path: PathBuf,
}

impl JitArtifact {
    fn create() -> Result<Self, LlvmError> {
        let sequence = JIT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "quantumforge-jit-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        let library_path = path.join(dynamic_library_name());
        Ok(Self { path, library_path })
    }
}

impl Drop for JitArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct DynamicLibrary {
    handle: *mut c_void,
}

impl DynamicLibrary {
    fn open(path: &Path) -> Result<Self, LlvmError> {
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::ffi::OsStrExt;
            let wide_path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            let handle = unsafe { LoadLibraryW(wide_path.as_ptr()) };
            if handle.is_null() {
                return Err(LlvmError::DynamicLibraryUnavailable(
                    path.display().to_string(),
                ));
            }
            Ok(Self { handle })
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let c_path = CString::new(path.as_os_str().as_bytes())
                .map_err(|error| LlvmError::DynamicLibraryUnavailable(error.to_string()))?;
            let handle = unsafe { dlopen(c_path.as_ptr(), 2) };
            if handle.is_null() {
                return Err(LlvmError::DynamicLibraryUnavailable(
                    path.display().to_string(),
                ));
            }
            Ok(Self { handle })
        }
        #[cfg(not(any(unix, target_os = "windows")))]
        {
            let _ = path;
            Err(LlvmError::UnsupportedDynamicLibraryPlatform)
        }
    }

    fn symbol(&self, name: &str) -> Option<*mut c_void> {
        let name = CString::new(name).ok()?;
        #[cfg(target_os = "windows")]
        {
            let symbol = unsafe { GetProcAddress(self.handle, name.as_ptr()) };
            (!symbol.is_null()).then_some(symbol)
        }
        #[cfg(unix)]
        {
            let symbol = unsafe { dlsym(self.handle, name.as_ptr()) };
            (!symbol.is_null()).then_some(symbol)
        }
        #[cfg(not(any(unix, target_os = "windows")))]
        {
            let _ = name;
            None
        }
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        #[cfg(target_os = "windows")]
        unsafe {
            FreeLibrary(self.handle);
        }
        #[cfg(unix)]
        unsafe {
            dlclose(self.handle);
        }
    }
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(filename: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
}

#[cfg(all(unix, target_os = "linux"))]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

#[cfg(all(unix, not(target_os = "linux")))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

fn run_with_ir(mut child: Child, ir: &[u8], tool: &str) -> Result<Output, LlvmError> {
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| LlvmError::ToolUnavailable(tool.to_owned()))?;
    let input = ir.to_vec();
    let writer = std::thread::spawn(move || stdin.write_all(&input));
    let output = child.wait_with_output()?;
    let write_result = writer.join().map_err(|_| LlvmError::ToolFailed {
        tool: tool.to_owned(),
        diagnostics: "IR input writer thread failed".to_owned(),
    })?;
    if !output.status.success() {
        return Ok(output);
    }
    write_result?;
    Ok(output)
}

fn value_name(register: Register) -> String {
    format!("%v{}", register.index())
}

fn llvm_float(value: f64) -> String {
    format!("{value:.17e}")
}

fn emit_binary(ir: &mut String, output: &str, operation: &str, a: Register, b: Register) {
    ir.push_str(&format!(
        "  {output} = {operation} double {}, {}\n",
        value_name(a),
        value_name(b)
    ));
}

fn emit_intrinsic(ir: &mut String, output: &str, intrinsic: &str, value: Register) {
    ir.push_str(&format!(
        "  {output} = call double @llvm.{intrinsic}.f64(double {})\n",
        value_name(value)
    ));
}
