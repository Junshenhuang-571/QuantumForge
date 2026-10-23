//! cuSOLVER FFI for GPU-accelerated complex SVD.
//!
//! Provides `gpu_svd_complex` which computes the full SVD of a complex matrix
//! on the GPU using cuSOLVER's zgesvd. Falls back to CPU on any error.

use crate::Complex64;
use std::ffi::c_void;

type CuSolverHandle = *mut c_void;

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum JobSvd {
    All = 0,
}

#[link(name = "cusolver")]
unsafe extern "C" {
    fn cusolverDnCreate(handle: *mut CuSolverHandle) -> i32;
    fn cusolverDnDestroy(handle: CuSolverHandle) -> i32;
    fn cusolverDnZgesvd_bufferSize(
        handle: CuSolverHandle,
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        lwork: *mut i32,
    ) -> i32;
    fn cusolverDnZgesvd(
        handle: CuSolverHandle,
        jobu: i32,
        jobvt: i32,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        s: *mut f64,
        u: *mut c_void,
        ldu: i32,
        vt: *mut c_void,
        ldvt: i32,
        work: *mut c_void,
        lwork: i32,
        rwork: *mut f64,
        devinfo: *mut i32,
    ) -> i32;
}

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: u32) -> i32;
    fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
    fn cuCtxCreate_v2(ctx: *mut *mut c_void, flags: u32, dev: i32) -> i32;
    fn cuCtxSetCurrent(ctx: *mut c_void) -> i32;
    fn cuMemAlloc_v2(dptr: *mut *mut c_void, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: *mut c_void) -> i32;
    fn cuMemcpyHtoD_v2(dst: *mut c_void, src: *const c_void, bytesize: usize) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: *const c_void, bytesize: usize) -> i32;
}

type SvdResult = (Vec<Complex64>, Vec<f64>, Vec<Complex64>);

/// GPU SVD for complex matrices. Returns (U, S, Vt) with full SVD.
/// U is m×m, S is min(m,n), Vt is n×n.
pub fn gpu_svd_complex(
    a: &[Complex64],
    m: usize,
    n: usize,
) -> Result<SvdResult, &'static str> {
    if m == 0 || n == 0 || a.len() != m * n {
        return Err("Invalid dimensions");
    }

    let min_mn = m.min(n);

    // Initialize CUDA
    unsafe {
        if cuInit(0) != 0 {
            return Err("cuInit failed");
        }
    }
    let mut device = 0i32;
    unsafe {
        if cuDeviceGet(&mut device, 0) != 0 {
            return Err("cuDeviceGet failed");
        }
    }
    let mut ctx: *mut c_void = std::ptr::null_mut();
    unsafe {
        if cuCtxCreate_v2(&mut ctx, 0, device) != 0 {
            return Err("cuCtxCreate failed");
        }
        if cuCtxSetCurrent(ctx) != 0 {
            return Err("cuCtxSetCurrent failed");
        }
    }

    // Create cuSOLVER handle
    let mut handle: CuSolverHandle = std::ptr::null_mut();
    unsafe {
        if cusolverDnCreate(&mut handle) != 0 {
            return Err("cusolverDnCreate failed");
        }
    }

    // Query buffer size
    let mut lwork: i32 = 0;
    unsafe {
        if cusolverDnZgesvd_bufferSize(handle, 0, 0, m as i32, n as i32, &mut lwork) != 0 {
            cusolverDnDestroy(handle);
            return Err("bufferSize failed");
        }
    }

    // Allocate device memory
    let a_bytes = m * n * 16; // Complex64 = 16 bytes
    let u_bytes = m * m * 16;
    let vt_bytes = n * n * 16;
    let s_bytes = min_mn * 8;
    let work_bytes = lwork as usize * 16;
    let rwork_bytes = (min_mn.max(1)) * 8;

    let mut d_a: *mut c_void = std::ptr::null_mut();
    let mut d_u: *mut c_void = std::ptr::null_mut();
    let mut d_vt: *mut c_void = std::ptr::null_mut();
    let mut d_s: *mut c_void = std::ptr::null_mut();
    let mut d_work: *mut c_void = std::ptr::null_mut();
    let mut d_rwork: *mut c_void = std::ptr::null_mut();
    let mut d_info: *mut c_void = std::ptr::null_mut();

    macro_rules! alloc {
        ($ptr:expr, $size:expr) => {
            unsafe {
                if cuMemAlloc_v2($ptr, $size) != 0 {
                    return Err("cuMemAlloc failed");
                }
            }
        };
    }

    alloc!(&mut d_a, a_bytes);
    alloc!(&mut d_u, u_bytes);
    alloc!(&mut d_vt, vt_bytes);
    alloc!(&mut d_s, s_bytes);
    alloc!(&mut d_work, work_bytes);
    alloc!(&mut d_rwork, rwork_bytes);
    alloc!(&mut d_info, 4);

    // Copy A to device
    unsafe {
        if cuMemcpyHtoD_v2(d_a, a.as_ptr() as *const c_void, a_bytes) != 0 {
            return Err("HtoD copy failed");
        }
    }

    // Run SVD
    let mut info: i32 = 0;
    let status = unsafe {
        cusolverDnZgesvd(
            handle,
            0, // jobu = all
            0, // jobvt = all
            m as i32,
            n as i32,
            d_a,
            m as i32,
            d_s as *mut f64,
            d_u,
            m as i32,
            d_vt,
            n as i32,
            d_work,
            lwork,
            d_rwork as *mut f64,
            &mut info as *mut i32,
        )
    };

    if status != 0 || info != 0 {
        // Cleanup
        unsafe {
            cuMemFree_v2(d_a);
            cuMemFree_v2(d_u);
            cuMemFree_v2(d_vt);
            cuMemFree_v2(d_s);
            cuMemFree_v2(d_work);
            cuMemFree_v2(d_rwork);
            cuMemFree_v2(d_info);
            cusolverDnDestroy(handle);
        }
        return Err("SVD computation failed");
    }

    // Copy results back
    let mut u = vec![Complex64::new(0.0, 0.0); m * m];
    let mut s = vec![0.0f64; min_mn];
    let mut vt = vec![Complex64::new(0.0, 0.0); n * n];

    unsafe {
        if cuMemcpyDtoH_v2(u.as_mut_ptr() as *mut c_void, d_u, u_bytes) != 0 {
            return Err("DtoH U failed");
        }
        if cuMemcpyDtoH_v2(s.as_mut_ptr() as *mut c_void, d_s, s_bytes) != 0 {
            return Err("DtoH S failed");
        }
        if cuMemcpyDtoH_v2(vt.as_mut_ptr() as *mut c_void, d_vt, vt_bytes) != 0 {
            return Err("DtoH Vt failed");
        }
    }

    // Cleanup
    unsafe {
        cuMemFree_v2(d_a);
        cuMemFree_v2(d_u);
        cuMemFree_v2(d_vt);
        cuMemFree_v2(d_s);
        cuMemFree_v2(d_work);
        cuMemFree_v2(d_rwork);
        cuMemFree_v2(d_info);
        cusolverDnDestroy(handle);
    }

    Ok((u, s, vt))
}
