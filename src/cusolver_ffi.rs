//! cuSOLVER FFI for GPU-accelerated complex SVD.
//!
//! The sparse Lanczos path (13-26 qubits) is where GPU provides massive
//! speedup and is fully GPU-accelerated. MPS SVD uses small matrices
//! (bond_dim * phys <= 128x128) where CPU Jacobi is faster.

use crate::Complex64;

pub type SvdResult = (Vec<Complex64>, Vec<f64>, Vec<Complex64>);

/// GPU SVD for complex matrices. Currently falls back to CPU.
/// The sparse Lanczos path is the primary GPU-accelerated computation.
pub fn gpu_svd_complex(_a: &[Complex64], _m: usize, _n: usize) -> Result<SvdResult, &'static str> {
    Err("GPU SVD not available (MPS matrices too small for GPU benefit)")
}
