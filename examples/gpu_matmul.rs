//! Example: GPU-accelerated matrix operations using cudarc.
//!
//! Run with: cargo run --example gpu_matmul --features gpu

#[cfg(feature = "gpu")]
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
#[cfg(feature = "gpu")]
use cudarc::driver::CudaDevice;

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        println!("This example requires the 'gpu' feature.");
        println!("Run with: cargo run --example gpu_matmul --features gpu");
    }

    #[cfg(feature = "gpu")]
    {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        println!("CUDA device initialized");

        let n = 256i32;
        let a: Vec<f32> = (0..(n * n) as usize).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..(n * n) as usize).map(|i| (i as f32) * 0.002).collect();

        let a_gpu = device.htod_copy(a).unwrap();
        let b_gpu = device.htod_copy(b).unwrap();
        let mut c_gpu = device.alloc_zeros::<f32>((n * n) as usize).unwrap();

        let blas = CudaBlas::new(device.clone()).unwrap();

        let cfg = GemmConfig {
            transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            m: n,
            n,
            k: n,
            alpha: 1.0f32,
            lda: n,
            ldb: n,
            beta: 0.0f32,
            ldc: n,
        };

        unsafe {
            blas.gemm(cfg, &a_gpu, &b_gpu, &mut c_gpu).unwrap();
        }

        let c_cpu = device.dtoh_sync_copy(&c_gpu).unwrap();
        println!("GEMM completed: {}x{} matrix multiplication", n, n);
        println!("Result[0] = {}", c_cpu[0]);
        println!(
            "Result[{}] = {}",
            (n * n / 2) as usize,
            c_cpu[(n * n / 2) as usize]
        );

        println!("\nTo use this in DMRG/TEBD:");
        println!("  1. Replace faer eigensolver with cuSOLVER cusolverDnSsyevd");
        println!("  2. Replace tensor contractions with cuBLAS gemm (above)");
        println!("  3. Replace SVD truncation with cuSOLVER cusolverDnSgesvd");
    }
}
