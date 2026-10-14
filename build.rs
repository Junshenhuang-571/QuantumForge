use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let root = env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .expect("the cuda feature requires CUDA_PATH to identify the CUDA Toolkit");
    let lib = root.join("lib").join("x64");
    for required in ["cuda.lib", "nvrtc.lib"] {
        if !lib.join(required).exists() {
            panic!(
                "the cuda feature requires {} in {}",
                required,
                lib.display()
            );
        }
    }
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
}
