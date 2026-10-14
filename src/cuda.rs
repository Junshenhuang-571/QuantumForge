//! Optional CUDA state-vector backend. Requires the CUDA Toolkit and an
//! NVIDIA driver when the `cuda` Cargo feature is enabled.

use crate::{
    Circuit, CircuitOp, Complex64, DensityMatrix, Gate, NoiseChannel, QuantumError, RotationAxis,
    ShotResult, SmallRng, StateVector,
};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr;
use std::rc::Rc;

type CuDevice = c_int;
type CuDevicePtr = u64;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type NvrtcProgram = *mut c_void;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: c_uint) -> c_int;
    fn cuDeviceGet(device: *mut CuDevice, ordinal: c_int) -> c_int;
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: CuDevice) -> c_int;
    fn cuDevicePrimaryCtxRetain(context: *mut CuContext, device: CuDevice) -> c_int;
    fn cuDevicePrimaryCtxRelease_v2(device: CuDevice) -> c_int;
    fn cuCtxSetCurrent(context: CuContext) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuModuleLoadData(module: *mut CuModule, image: *const c_void) -> c_int;
    fn cuModuleGetFunction(
        function: *mut CuFunction,
        module: CuModule,
        name: *const c_char,
    ) -> c_int;
    fn cuModuleUnload(module: CuModule) -> c_int;
    fn cuMemAlloc_v2(pointer: *mut CuDevicePtr, bytes: usize) -> c_int;
    fn cuMemFree_v2(pointer: CuDevicePtr) -> c_int;
    fn cuMemcpyHtoD_v2(destination: CuDevicePtr, source: *const c_void, bytes: usize) -> c_int;
    fn cuMemcpyDtoH_v2(destination: *mut c_void, source: CuDevicePtr, bytes: usize) -> c_int;
    fn cuMemcpyDtoD_v2(destination: CuDevicePtr, source: CuDevicePtr, bytes: usize) -> c_int;
    fn cuLaunchKernel(
        function: CuFunction,
        gx: c_uint,
        gy: c_uint,
        gz: c_uint,
        bx: c_uint,
        by: c_uint,
        bz: c_uint,
        shared: c_uint,
        stream: *mut c_void,
        args: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;
    fn cuGetErrorString(error: c_int, text: *mut *const c_char) -> c_int;

    fn nvrtcCreateProgram(
        program: *mut NvrtcProgram,
        source: *const c_char,
        name: *const c_char,
        headers: c_int,
        include_names: *const *const c_char,
        include_contents: *const *const c_char,
    ) -> c_int;
    fn nvrtcCompileProgram(
        program: NvrtcProgram,
        count: c_int,
        options: *const *const c_char,
    ) -> c_int;
    fn nvrtcGetPTXSize(program: NvrtcProgram, size: *mut usize) -> c_int;
    fn nvrtcGetPTX(program: NvrtcProgram, ptx: *mut c_char) -> c_int;
    fn nvrtcGetProgramLogSize(program: NvrtcProgram, size: *mut usize) -> c_int;
    fn nvrtcGetProgramLog(program: NvrtcProgram, log: *mut c_char) -> c_int;
    fn nvrtcDestroyProgram(program: *mut NvrtcProgram) -> c_int;
    fn nvrtcGetErrorString(error: c_int) -> *const c_char;
}

const KERNEL: &str = r#"
extern "C" __global__ void apply_rotation_kernel(double2* state, int n, int target, int axis, double theta) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int mask = 1 << target;
    if (i >= n || (i & mask) != 0) return;
    int j = i | mask;
    double c = cos(theta * 0.5);
    double s = sin(theta * 0.5);
    double2 a = state[i];
    double2 b = state[j];
    if (axis == 0) {
        state[i] = make_double2(c*a.x + s*b.y, c*a.y - s*b.x);
        state[j] = make_double2(c*b.x + s*a.y, c*b.y - s*a.x);
    } else if (axis == 1) {
        state[i] = make_double2(c*a.x - s*b.x, c*a.y - s*b.y);
        state[j] = make_double2(s*a.x + c*b.x, s*a.y + c*b.y);
    } else {
        state[i] = make_double2(c*a.x + s*a.y, c*a.y - s*a.x);
        state[j] = make_double2(c*b.x - s*b.y, c*b.y + s*b.x);
    }
}
extern "C" __global__ void apply_two_qubit_kernel(double2* state, int n, int first, int second, int op) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int first_mask = 1 << first;
    int second_mask = 1 << second;
    if (op == 0 && (i & first_mask) != 0 && (i & second_mask) == 0) {
        int j = i | second_mask;
        double2 value = state[i]; state[i] = state[j]; state[j] = value;
    } else if (op == 1 && (i & first_mask) != 0 && (i & second_mask) != 0) {
        state[i] = make_double2(-state[i].x, -state[i].y);
    } else if (op == 2 && (i & first_mask) == 0 && (i & second_mask) != 0) {
        int j = i ^ first_mask ^ second_mask;
        double2 value = state[i]; state[i] = state[j]; state[j] = value;
    }
}
extern "C" __global__ void apply_matrix_kernel(
    double2* state, int n, int target,
    double2 m00, double2 m01, double2 m10, double2 m11
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int mask = 1 << target;
    if (i >= n || (i & mask) != 0) return;
    int j = i | mask;
    double2 a = state[i];
    double2 b = state[j];
    state[i] = make_double2(m00.x*a.x - m00.y*a.y + m01.x*b.x - m01.y*b.y,
                            m00.x*a.y + m00.y*a.x + m01.x*b.y + m01.y*b.x);
    state[j] = make_double2(m10.x*a.x - m10.y*a.y + m11.x*b.x - m11.y*b.y,
                            m10.x*a.y + m10.y*a.x + m11.x*b.y + m11.y*b.x);
}
extern "C" __global__ void apply_batch_matrix_kernel(
    double2* states, int n, int batch_size, int target,
    double2 m00, double2 m01, double2 m10, double2 m11
) {
    int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int pairs_per_state = n / 2;
    int total_pairs = pairs_per_state * batch_size;
    if (pair >= total_pairs) return;
    int batch = pair / pairs_per_state;
    int local_pair = pair % pairs_per_state;
    int mask = 1 << target;
    int i = (local_pair & (mask - 1)) | ((local_pair & ~(mask - 1)) << 1);
    int j = i | mask;
    int offset = batch * n;
    double2 a = states[offset + i];
    double2 b = states[offset + j];
    states[offset + i] = make_double2(m00.x*a.x - m00.y*a.y + m01.x*b.x - m01.y*b.y,
                                      m00.x*a.y + m00.y*a.x + m01.x*b.y + m01.y*b.x);
    states[offset + j] = make_double2(m10.x*a.x - m10.y*a.y + m11.x*b.x - m11.y*b.y,
                                      m10.x*a.y + m10.y*a.x + m11.x*b.y + m11.y*b.x);
}
extern "C" __global__ void apply_batch_masked_matrix_kernel(
    double2* states, int n, int batch_size, int target,
    const unsigned char* enabled,
    double2 m00, double2 m01, double2 m10, double2 m11
) {
    int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int pairs_per_state = n / 2;
    int total_pairs = pairs_per_state * batch_size;
    if (pair >= total_pairs) return;
    int batch = pair / pairs_per_state;
    if (enabled[batch] == 0) return;
    int local_pair = pair % pairs_per_state;
    int mask = 1 << target;
    int i = (local_pair & (mask - 1)) | ((local_pair & ~(mask - 1)) << 1);
    int j = i | mask;
    int offset = batch * n;
    double2 a = states[offset + i];
    double2 b = states[offset + j];
    states[offset + i] = make_double2(m00.x*a.x - m00.y*a.y + m01.x*b.x - m01.y*b.y,
                                      m00.x*a.y + m00.y*a.x + m01.x*b.y + m01.y*b.x);
    states[offset + j] = make_double2(m10.x*a.x - m10.y*a.y + m11.x*b.x - m11.y*b.y,
                                      m10.x*a.y + m10.y*a.x + m11.x*b.y + m11.y*b.x);
}
extern "C" __global__ void apply_batch_two_qubit_kernel(
    double2* states, int n, int batch_size, int first, int second, int op
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = n * batch_size;
    if (index >= total) return;
    int batch = index / n;
    int i = index % n;
    int offset = batch * n;
    int first_mask = 1 << first;
    int second_mask = 1 << second;
    if (op == 0 && (i & first_mask) != 0 && (i & second_mask) == 0) {
        int j = i | second_mask;
        double2 value = states[offset + i];
        states[offset + i] = states[offset + j];
        states[offset + j] = value;
    } else if (op == 1 && (i & first_mask) != 0 && (i & second_mask) != 0) {
        states[index] = make_double2(-states[index].x, -states[index].y);
    } else if (op == 2 && (i & first_mask) == 0 && (i & second_mask) != 0) {
        int j = i ^ first_mask ^ second_mask;
        double2 value = states[offset + i];
        states[offset + i] = states[offset + j];
        states[offset + j] = value;
    }
}
extern "C" __global__ void measure_batch_probability_kernel(
    const double2* states, int n, int batch_size, int target,
    int blocks_per_state, double* partial_sums
) {
    __shared__ double values[256];
    int lane = (int)threadIdx.x;
    int block = (int)blockIdx.x;
    int total_blocks = blocks_per_state * batch_size;
    if (block >= total_blocks) return;
    int state_index = block / blocks_per_state;
    int local_block = block % blocks_per_state;
    int index = local_block * (int)blockDim.x + lane;
    double probability = 0.0;
    if (index < n && (index & (1 << target)) != 0) {
        double2 amplitude = states[state_index * n + index];
        probability = amplitude.x * amplitude.x + amplitude.y * amplitude.y;
    }
    values[lane] = probability;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }
    if (lane == 0) {
        partial_sums[state_index * blocks_per_state + local_block] = values[0];
    }
}
extern "C" __global__ void measure_batch_collapse_kernel(
    double2* states, int n, int batch_size, int target,
    const unsigned char* outcomes, const double* inverse_norms
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = n * batch_size;
    if (index >= total) return;
    int state_index = index / n;
    int local_index = index % n;
    int outcome = outcomes[state_index];
    int measured = (local_index & (1 << target)) != 0;
    if (measured != outcome) {
        states[index] = make_double2(0.0, 0.0);
    } else {
        states[index].x *= inverse_norms[state_index];
        states[index].y *= inverse_norms[state_index];
    }
}
extern "C" __global__ void apply_batch_damping_trajectory_kernel(
    double2* states, int n, int batch_size, int target, int channel,
    double gamma, const unsigned char* jumps, const double* inverse_norms
) {
    int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int pairs_per_state = n / 2;
    int total_pairs = pairs_per_state * batch_size;
    if (pair >= total_pairs) return;
    int state_index = pair / pairs_per_state;
    int local_pair = pair % pairs_per_state;
    int mask = 1 << target;
    int index_zero = (local_pair & (mask - 1)) | ((local_pair & ~(mask - 1)) << 1);
    int index_one = index_zero | mask;
    int offset = state_index * n;
    double2 zero = states[offset + index_zero];
    double2 one = states[offset + index_one];
    double inverse = inverse_norms[state_index];
    if (jumps[state_index] != 0) {
        double scale = sqrt(gamma) * inverse;
        if (channel == 0) {
            states[offset + index_zero] = make_double2(one.x * scale, one.y * scale);
            states[offset + index_one] = make_double2(0.0, 0.0);
        } else {
            states[offset + index_zero] = make_double2(0.0, 0.0);
            states[offset + index_one] = make_double2(one.x * scale, one.y * scale);
        }
    } else {
        states[offset + index_zero] = make_double2(zero.x * inverse, zero.y * inverse);
        double scale = sqrt(1.0 - gamma) * inverse;
        states[offset + index_one] = make_double2(one.x * scale, one.y * scale);
    }
}
extern "C" __global__ void apply_batch_pauli_trajectory_kernel(
    double2* states, int n, int batch_size, int target,
    const unsigned char* branches
) {
    int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int pairs_per_state = n / 2;
    int total_pairs = pairs_per_state * batch_size;
    if (pair >= total_pairs) return;
    int state_index = pair / pairs_per_state;
    int local_pair = pair % pairs_per_state;
    int branch = branches[state_index];
    if (branch == 0) return;
    int mask = 1 << target;
    int index_zero = (local_pair & (mask - 1)) | ((local_pair & ~(mask - 1)) << 1);
    int index_one = index_zero | mask;
    int offset = state_index * n;
    double2 zero = states[offset + index_zero];
    double2 one = states[offset + index_one];
    if (branch == 1) {
        states[offset + index_zero] = one;
        states[offset + index_one] = zero;
    } else if (branch == 2) {
        states[offset + index_zero] = make_double2(one.y, -one.x);
        states[offset + index_one] = make_double2(-zero.y, zero.x);
    } else if (branch == 3) {
        states[offset + index_one] = make_double2(-one.x, -one.y);
    }
}
extern "C" __global__ void measure_probability_kernel(
    const double2* state, int n, int target, double* partial_sums
) {
    __shared__ double values[256];
    int lane = (int)threadIdx.x;
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    double probability = 0.0;
    if (index < n && (index & (1 << target)) != 0) {
        double2 amplitude = state[index];
        probability = amplitude.x * amplitude.x + amplitude.y * amplitude.y;
    }
    values[lane] = probability;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }
    if (lane == 0) partial_sums[blockIdx.x] = values[0];
}
extern "C" __global__ void measure_collapse_kernel(
    double2* state, int n, int target, int outcome, double inverse_norm
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n) return;
    int measured = (index & (1 << target)) != 0;
    if (measured != outcome) {
        state[index] = make_double2(0.0, 0.0);
    } else {
        state[index].x *= inverse_norm;
        state[index].y *= inverse_norm;
    }
}
extern "C" __global__ void density_measure_probability_kernel(
    const double2* density, int dimension, int target, double* partial_sums
) {
    __shared__ double values[256];
    int lane = (int)threadIdx.x;
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    double probability = 0.0;
    if (index < dimension && (index & (1 << target)) != 0) {
        probability = density[index * dimension + index].x;
    }
    values[lane] = probability;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }
    if (lane == 0) partial_sums[blockIdx.x] = values[0];
}
extern "C" __global__ void density_measure_collapse_kernel(
    double2* density, int dimension, int target, int outcome, double inverse_probability
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int elements = dimension * dimension;
    if (index >= elements) return;
    int row = index / dimension;
    int col = index % dimension;
    int row_matches = ((row >> target) & 1) == outcome;
    int col_matches = ((col >> target) & 1) == outcome;
    if (!row_matches || !col_matches) {
        density[index] = make_double2(0.0, 0.0);
    } else {
        density[index].x *= inverse_probability;
        density[index].y *= inverse_probability;
    }
}
__device__ double2 complex_add(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ double2 complex_scale(double2 a, double value) {
    return make_double2(a.x * value, a.y * value);
}
extern "C" __global__ void density_local_kernel(
    const double2* input, double2* output, int dimension, int target, int direction,
    double2 m00, double2 m01, double2 m10, double2 m11
) {
    int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int pairs = (dimension * dimension) / 2;
    if (pair >= pairs) return;
    int mask = 1 << target;
    if (direction == 0) {
        int row_pair = pair / dimension;
        int col = pair % dimension;
        int row0 = (row_pair & (mask - 1)) | ((row_pair & ~(mask - 1)) << 1);
        int row1 = row0 | mask;
        double2 a = input[row0 * dimension + col];
        double2 b = input[row1 * dimension + col];
        output[row0 * dimension + col] = complex_add(
            make_double2(m00.x*a.x - m00.y*a.y, m00.x*a.y + m00.y*a.x),
            make_double2(m01.x*b.x - m01.y*b.y, m01.x*b.y + m01.y*b.x));
        output[row1 * dimension + col] = complex_add(
            make_double2(m10.x*a.x - m10.y*a.y, m10.x*a.y + m10.y*a.x),
            make_double2(m11.x*b.x - m11.y*b.y, m11.x*b.y + m11.y*b.x));
    } else {
        int row = pair / (dimension / 2);
        int col_pair = pair % (dimension / 2);
        int col0 = (col_pair & (mask - 1)) | ((col_pair & ~(mask - 1)) << 1);
        int col1 = col0 | mask;
        double2 a = input[row * dimension + col0];
        double2 b = input[row * dimension + col1];
        output[row * dimension + col0] = complex_add(
            make_double2(a.x*m00.x + a.y*m00.y, a.y*m00.x - a.x*m00.y),
            make_double2(b.x*m01.x + b.y*m01.y, b.y*m01.x - b.x*m01.y));
        output[row * dimension + col1] = complex_add(
            make_double2(a.x*m10.x + a.y*m10.y, a.y*m10.x - a.x*m10.y),
            make_double2(b.x*m11.x + b.y*m11.y, b.y*m11.x - b.x*m11.y));
    }
}
extern "C" __global__ void density_permute_kernel(
    const double2* input, double2* output, int dimension, int first, int second, int operation
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int elements = dimension * dimension;
    if (index >= elements) return;
    int row = index / dimension;
    int col = index % dimension;
    int first_mask = 1 << first;
    int second_mask = 1 << second;
    int mapped_row = row;
    int mapped_col = col;
    if (operation == 0) {
        if ((row & first_mask) != 0) mapped_row ^= second_mask;
        if ((col & first_mask) != 0) mapped_col ^= second_mask;
    } else {
        if (((row & first_mask) == 0) != ((row & second_mask) == 0)) mapped_row ^= first_mask | second_mask;
        if (((col & first_mask) == 0) != ((col & second_mask) == 0)) mapped_col ^= first_mask | second_mask;
    }
    output[index] = input[mapped_row * dimension + mapped_col];
}
extern "C" __global__ void density_cz_kernel(
    double2* state, int dimension, int first, int second
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int elements = dimension * dimension;
    if (index >= elements) return;
    int row = index / dimension;
    int col = index % dimension;
    int a = 1 << first;
    int b = 1 << second;
    int row_phase = (row & a) != 0 && (row & b) != 0;
    int col_phase = (col & a) != 0 && (col & b) != 0;
    if (row_phase != col_phase) state[index] = complex_scale(state[index], -1.0);
}
extern "C" __global__ void density_channel_kernel(
    const double2* input, double2* output, int dimension, int target,
    int channel, double parameter
) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int elements = dimension * dimension;
    if (index >= elements) return;
    int row = index / dimension;
    int col = index % dimension;
    int mask = 1 << target;
    int row_one = (row & mask) != 0;
    int col_one = (col & mask) != 0;
    double2 current = input[index];
    if (channel == 0 || channel == 1) {
        if (row_one != col_one) {
            output[index] = complex_scale(current, sqrt(1.0 - parameter));
        } else if (row_one && channel == 0) {
            output[index] = complex_scale(current, 1.0 - parameter);
        } else if (!row_one && channel == 0) {
            double2 excited = input[(row | mask) * dimension + (col | mask)];
            output[index] = complex_add(current, complex_scale(excited, parameter));
        } else {
            output[index] = current;
        }
    } else {
        double2 flipped = input[(row ^ mask) * dimension + (col ^ mask)];
        double sign = row_one == col_one ? 1.0 : -1.0;
        double scale = parameter / 3.0;
        double2 pauli_sum = complex_add(
            complex_add(flipped, complex_scale(flipped, sign)),
            complex_scale(current, sign));
        output[index] = complex_add(complex_scale(current, 1.0 - parameter),
                                    complex_scale(pauli_sum, scale));
    }
}
"#;

struct CudaBackendInner {
    device: CuDevice,
    context: CuContext,
    module: CuModule,
    rotation: CuFunction,
    two_qubit: CuFunction,
    matrix: CuFunction,
    batch_matrix: CuFunction,
    batch_masked_matrix: CuFunction,
    batch_two_qubit: CuFunction,
    batch_measure_probability: CuFunction,
    batch_measure_collapse: CuFunction,
    batch_damping_trajectory: CuFunction,
    batch_pauli_trajectory: CuFunction,
    measure_probability: CuFunction,
    measure_collapse: CuFunction,
    density_local: CuFunction,
    density_permute: CuFunction,
    density_cz: CuFunction,
    density_channel: CuFunction,
    density_measure_probability: CuFunction,
    density_measure_collapse: CuFunction,
}

pub struct CudaBackend {
    inner: Rc<CudaBackendInner>,
}

/// State vector allocated on the CUDA device. It keeps the backend context and
/// compiled kernels alive, and repeated operations avoid host/device transfers.
pub struct CudaDeviceState {
    backend: Rc<CudaBackendInner>,
    pointer: CuDevicePtr,
    qubits: usize,
    amplitudes: usize,
    reduction_scratch: Option<CuDevicePtr>,
}

/// A contiguous batch of equal-width state vectors kept resident on the GPU.
pub struct CudaDeviceStateBatch {
    backend: Rc<CudaBackendInner>,
    pointer: CuDevicePtr,
    qubits: usize,
    amplitudes_per_state: usize,
    batch_size: usize,
    reduction_scratch: CuDevicePtr,
    outcomes_scratch: CuDevicePtr,
    inverse_norms_scratch: CuDevicePtr,
    reduction_blocks_per_state: usize,
}

/// Device-resident row-major density matrix with reusable transformation scratch.
pub struct CudaDeviceDensityMatrix {
    backend: Rc<CudaBackendInner>,
    pointer: CuDevicePtr,
    scratch: CuDevicePtr,
    reduction_scratch: CuDevicePtr,
    qubits: usize,
    dimension: usize,
    elements: usize,
}

impl CudaBackend {
    pub fn new(ordinal: i32) -> Result<Self, CudaError> {
        check_driver(unsafe { cuInit(0) })?;
        let mut device = 0;
        check_driver(unsafe { cuDeviceGet(&mut device, ordinal) })?;
        let mut context = ptr::null_mut();
        check_driver(unsafe { cuDevicePrimaryCtxRetain(&mut context, device) })?;
        let initialized = (|| {
            check_driver(unsafe { cuCtxSetCurrent(context) })?;
            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR
            let mut major = 0;
            let mut minor = 0;
            check_driver(unsafe { cuDeviceGetAttribute(&mut major, 75, device) })?;
            check_driver(unsafe { cuDeviceGetAttribute(&mut minor, 76, device) })?;
            let ptx = compile_ptx(major, minor)?;
            let mut module = ptr::null_mut();
            check_driver(unsafe { cuModuleLoadData(&mut module, ptx.as_ptr().cast()) })?;
            let name = CString::new("apply_rotation_kernel").unwrap();
            let mut rotation = ptr::null_mut();
            if let Err(error) =
                check_driver(unsafe { cuModuleGetFunction(&mut rotation, module, name.as_ptr()) })
            {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("apply_two_qubit_kernel").unwrap();
            let mut two_qubit = ptr::null_mut();
            if let Err(error) =
                check_driver(unsafe { cuModuleGetFunction(&mut two_qubit, module, name.as_ptr()) })
            {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("apply_matrix_kernel").unwrap();
            let mut matrix = ptr::null_mut();
            if let Err(error) =
                check_driver(unsafe { cuModuleGetFunction(&mut matrix, module, name.as_ptr()) })
            {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("apply_batch_matrix_kernel").unwrap();
            let mut batch_matrix = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut batch_matrix, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("apply_batch_two_qubit_kernel").unwrap();
            let mut batch_two_qubit = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut batch_two_qubit, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("apply_batch_masked_matrix_kernel").unwrap();
            let mut batch_masked_matrix = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut batch_masked_matrix, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("measure_batch_probability_kernel").unwrap();
            let mut batch_measure_probability = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut batch_measure_probability, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("measure_batch_collapse_kernel").unwrap();
            let mut batch_measure_collapse = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut batch_measure_collapse, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("apply_batch_damping_trajectory_kernel").unwrap();
            let mut batch_damping_trajectory = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut batch_damping_trajectory, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("apply_batch_pauli_trajectory_kernel").unwrap();
            let mut batch_pauli_trajectory = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut batch_pauli_trajectory, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("measure_probability_kernel").unwrap();
            let mut measure_probability = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut measure_probability, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("measure_collapse_kernel").unwrap();
            let mut measure_collapse = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut measure_collapse, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("density_local_kernel").unwrap();
            let mut density_local = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut density_local, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("density_permute_kernel").unwrap();
            let mut density_permute = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut density_permute, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("density_cz_kernel").unwrap();
            let mut density_cz = ptr::null_mut();
            if let Err(error) =
                check_driver(unsafe { cuModuleGetFunction(&mut density_cz, module, name.as_ptr()) })
            {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("density_channel_kernel").unwrap();
            let mut density_channel = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut density_channel, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("density_measure_probability_kernel").unwrap();
            let mut density_measure_probability = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut density_measure_probability, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            let name = CString::new("density_measure_collapse_kernel").unwrap();
            let mut density_measure_collapse = ptr::null_mut();
            if let Err(error) = check_driver(unsafe {
                cuModuleGetFunction(&mut density_measure_collapse, module, name.as_ptr())
            }) {
                unsafe { cuModuleUnload(module) };
                return Err(error);
            }
            Ok((
                module,
                rotation,
                two_qubit,
                matrix,
                batch_matrix,
                batch_masked_matrix,
                batch_two_qubit,
                batch_measure_probability,
                batch_measure_collapse,
                batch_damping_trajectory,
                batch_pauli_trajectory,
                measure_probability,
                measure_collapse,
                density_local,
                density_permute,
                density_cz,
                density_channel,
                density_measure_probability,
                density_measure_collapse,
            ))
        })();
        match initialized {
            Ok((
                module,
                rotation,
                two_qubit,
                matrix,
                batch_matrix,
                batch_masked_matrix,
                batch_two_qubit,
                batch_measure_probability,
                batch_measure_collapse,
                batch_damping_trajectory,
                batch_pauli_trajectory,
                measure_probability,
                measure_collapse,
                density_local,
                density_permute,
                density_cz,
                density_channel,
                density_measure_probability,
                density_measure_collapse,
            )) => Ok(Self {
                inner: Rc::new(CudaBackendInner {
                    device,
                    context,
                    module,
                    rotation,
                    two_qubit,
                    matrix,
                    batch_matrix,
                    batch_masked_matrix,
                    batch_two_qubit,
                    batch_measure_probability,
                    batch_measure_collapse,
                    batch_damping_trajectory,
                    batch_pauli_trajectory,
                    measure_probability,
                    measure_collapse,
                    density_local,
                    density_permute,
                    density_cz,
                    density_channel,
                    density_measure_probability,
                    density_measure_collapse,
                }),
            }),
            Err(error) => {
                unsafe { cuDevicePrimaryCtxRelease_v2(device) };
                Err(error)
            }
        }
    }

    /// Apply Ry on the device, returning amplitudes to host memory on completion.
    pub fn apply_ry(
        &self,
        state: &mut StateVector,
        qubit: usize,
        theta: f64,
    ) -> Result<(), CudaError> {
        self.apply_rotation(state, RotationAxis::Y, qubit, theta)
    }

    pub fn apply_rotation(
        &self,
        state: &mut StateVector,
        axis: RotationAxis,
        qubit: usize,
        theta: f64,
    ) -> Result<(), CudaError> {
        self.apply_rotation_chain(state, &[(axis, qubit, theta)])
    }

    /// Upload once, apply a sequence of Ry gates on the device, then download
    /// once. Gate dispatches share one device allocation and one stream.
    pub fn apply_ry_chain(
        &self,
        state: &mut StateVector,
        gates: &[(usize, f64)],
    ) -> Result<(), CudaError> {
        let gates: Vec<_> = gates
            .iter()
            .map(|(q, t)| (RotationAxis::Y, *q, *t))
            .collect();
        self.apply_rotation_chain(state, &gates)
    }

    pub fn apply_rotation_chain(
        &self,
        state: &mut StateVector,
        gates: &[(RotationAxis, usize, f64)],
    ) -> Result<(), CudaError> {
        let mut device_state = self.upload(state)?;
        device_state.apply_rotation_chain(gates)?;
        *state = device_state.download()?;
        Ok(())
    }

    /// Upload a host state once. The returned state can remain on the GPU
    /// across many gate dispatches and is released automatically on drop.
    pub fn upload(&self, state: &StateVector) -> Result<CudaDeviceState, CudaError> {
        validate_gates(state.qubits(), &[])?;
        let bytes = state
            .amplitudes()
            .len()
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe { cuCtxSetCurrent(self.inner.context) })?;
        let mut pointer = 0;
        check_driver(unsafe { cuMemAlloc_v2(&mut pointer, bytes) })?;
        if let Err(error) = check_driver(unsafe {
            cuMemcpyHtoD_v2(pointer, state.amplitudes().as_ptr().cast(), bytes)
        }) {
            unsafe { cuMemFree_v2(pointer) };
            return Err(error);
        }
        Ok(CudaDeviceState {
            backend: Rc::clone(&self.inner),
            pointer,
            qubits: state.qubits(),
            amplitudes: state.amplitudes().len(),
            reduction_scratch: None,
        })
    }

    /// Upload equal-width state vectors as one contiguous allocation for
    /// batched gate execution.
    pub fn upload_batch(&self, states: &[StateVector]) -> Result<CudaDeviceStateBatch, CudaError> {
        let first = states.first().ok_or(CudaError::EmptyBatch)?;
        let qubits = first.qubits();
        validate_gates(qubits, &[])?;
        if states.iter().any(|state| state.qubits() != qubits) {
            return Err(CudaError::IncompatibleBatch);
        }
        let amplitudes_per_state = first.amplitudes().len();
        let total_amplitudes = amplitudes_per_state
            .checked_mul(states.len())
            .ok_or(CudaError::StateTooLarge)?;
        i32::try_from(total_amplitudes).map_err(|_| CudaError::StateTooLarge)?;
        let bytes = total_amplitudes
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(CudaError::StateTooLarge)?;
        let reduction_blocks_per_state = amplitudes_per_state.div_ceil(256);
        let reduction_entries = reduction_blocks_per_state
            .checked_mul(states.len())
            .ok_or(CudaError::StateTooLarge)?;
        let reduction_bytes = reduction_entries
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        let inverse_norms_bytes = states
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        let mut amplitudes = Vec::with_capacity(total_amplitudes);
        for state in states {
            amplitudes.extend_from_slice(state.amplitudes());
        }
        check_driver(unsafe { cuCtxSetCurrent(self.inner.context) })?;
        let mut pointer = 0;
        check_driver(unsafe { cuMemAlloc_v2(&mut pointer, bytes) })?;
        let mut reduction_scratch = 0;
        if let Err(error) =
            check_driver(unsafe { cuMemAlloc_v2(&mut reduction_scratch, reduction_bytes) })
        {
            unsafe { cuMemFree_v2(pointer) };
            return Err(error);
        }
        let mut outcomes_scratch = 0;
        if let Err(error) =
            check_driver(unsafe { cuMemAlloc_v2(&mut outcomes_scratch, states.len()) })
        {
            unsafe {
                cuMemFree_v2(pointer);
                cuMemFree_v2(reduction_scratch);
            }
            return Err(error);
        }
        let mut inverse_norms_scratch = 0;
        if let Err(error) =
            check_driver(unsafe { cuMemAlloc_v2(&mut inverse_norms_scratch, inverse_norms_bytes) })
        {
            unsafe {
                cuMemFree_v2(pointer);
                cuMemFree_v2(reduction_scratch);
                cuMemFree_v2(outcomes_scratch);
            }
            return Err(error);
        }
        if let Err(error) =
            check_driver(unsafe { cuMemcpyHtoD_v2(pointer, amplitudes.as_ptr().cast(), bytes) })
        {
            unsafe {
                cuMemFree_v2(pointer);
                cuMemFree_v2(reduction_scratch);
                cuMemFree_v2(outcomes_scratch);
                cuMemFree_v2(inverse_norms_scratch);
            }
            return Err(error);
        }
        Ok(CudaDeviceStateBatch {
            backend: Rc::clone(&self.inner),
            pointer,
            qubits,
            amplitudes_per_state,
            batch_size: states.len(),
            reduction_scratch,
            outcomes_scratch,
            inverse_norms_scratch,
            reduction_blocks_per_state,
        })
    }

    /// Run one unitary circuit over many independent initial states with a
    /// single upload and download around device-side batched gate kernels.
    pub fn run_circuit_batch(
        &self,
        states: &[StateVector],
        circuit: &Circuit,
    ) -> Result<Vec<StateVector>, CudaError> {
        let mut batch = self.upload_batch(states)?;
        batch.run_circuit(circuit)?;
        batch.download()
    }

    /// Upload a dense density matrix to a resident CUDA allocation.
    pub fn upload_density(
        &self,
        density: &DensityMatrix,
    ) -> Result<CudaDeviceDensityMatrix, CudaError> {
        let (dimension, elements) = validate_density_size(density.qubits())?;
        if density.entries().len() != elements {
            return Err(CudaError::StateTooLarge);
        }
        let bytes = elements
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe { cuCtxSetCurrent(self.inner.context) })?;
        let mut pointer = 0;
        check_driver(unsafe { cuMemAlloc_v2(&mut pointer, bytes) })?;
        let mut scratch = 0;
        if let Err(error) = check_driver(unsafe { cuMemAlloc_v2(&mut scratch, bytes) }) {
            unsafe { cuMemFree_v2(pointer) };
            return Err(error);
        }
        let reduction_blocks = dimension.div_ceil(256);
        let reduction_bytes = reduction_blocks
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        let mut reduction_scratch = 0;
        if let Err(error) =
            check_driver(unsafe { cuMemAlloc_v2(&mut reduction_scratch, reduction_bytes) })
        {
            unsafe {
                cuMemFree_v2(pointer);
                cuMemFree_v2(scratch);
            }
            return Err(error);
        }
        if let Err(error) = check_driver(unsafe {
            cuMemcpyHtoD_v2(pointer, density.entries().as_ptr().cast(), bytes)
        }) {
            unsafe {
                cuMemFree_v2(pointer);
                cuMemFree_v2(scratch);
                cuMemFree_v2(reduction_scratch);
            }
            return Err(error);
        }
        Ok(CudaDeviceDensityMatrix {
            backend: Rc::clone(&self.inner),
            pointer,
            scratch,
            reduction_scratch,
            qubits: density.qubits(),
            dimension,
            elements,
        })
    }

    /// Run a circuit over a dense density matrix, including local noise
    /// channels, while retaining the matrix and scratch buffers on the GPU.
    pub fn run_density_circuit(
        &self,
        initial: &DensityMatrix,
        circuit: &Circuit,
    ) -> Result<DensityMatrix, CudaError> {
        let mut density = self.upload_density(initial)?;
        for operation in circuit.operations() {
            match *operation {
                CircuitOp::SingleQubit { gate, target } => density.apply_gate(gate, target)?,
                CircuitOp::Cnot { control, target } => density.apply_cnot(control, target)?,
                CircuitOp::Cz { first, second } => density.apply_cz(first, second)?,
                CircuitOp::Swap { first, second } => density.apply_swap(first, second)?,
                CircuitOp::Channel { channel, target } => density.apply_channel(channel, target)?,
                CircuitOp::Measure { .. } | CircuitOp::ConditionalSingleQubit { .. } => {
                    return Err(CudaError::Quantum(
                        QuantumError::OperationRequiresShotExecution,
                    ));
                }
            }
        }
        density.download()
    }

    /// Run a unitary circuit without intermediate host transfers. Channels
    /// require density-matrix simulation and are rejected with a typed error.
    pub fn run_circuit(
        &self,
        state: StateVector,
        circuit: &Circuit,
    ) -> Result<StateVector, CudaError> {
        let mut device_state = self.upload(&state)?;
        for operation in circuit.operations() {
            match *operation {
                CircuitOp::SingleQubit { gate, target } => {
                    device_state.apply_gate(gate, target)?;
                }
                CircuitOp::Cnot { control, target } => {
                    device_state.apply_cnot(control, target)?;
                }
                CircuitOp::Cz { first, second } => {
                    device_state.apply_cz(first, second)?;
                }
                CircuitOp::Swap { first, second } => {
                    device_state.apply_swap(first, second)?;
                }
                CircuitOp::Channel { .. } => {
                    return Err(CudaError::Quantum(
                        QuantumError::OperationRequiresDensityMatrix,
                    ));
                }
                CircuitOp::Measure { .. } | CircuitOp::ConditionalSingleQubit { .. } => {
                    return Err(CudaError::Quantum(
                        QuantumError::OperationRequiresShotExecution,
                    ));
                }
            }
        }
        device_state.download()
    }

    /// Execute seeded circuit shots with gates, projective measurement, and
    /// classical single-qubit feed-forward on a resident CUDA state. Between
    /// shots the initial state is restored with a device-to-device copy.
    pub fn run_circuit_shots(
        &self,
        initial_state: &StateVector,
        circuit: &Circuit,
        shots: usize,
        seed: u64,
    ) -> Result<ShotResult, CudaError> {
        if shots == 0 {
            return Err(CudaError::Quantum(QuantumError::InvalidShotCount));
        }
        let mut classical_width = 0;
        for operation in circuit.operations() {
            match operation {
                CircuitOp::Measure { classical_bit, .. }
                | CircuitOp::ConditionalSingleQubit { classical_bit, .. } => {
                    if *classical_bit >= 4096 {
                        return Err(CudaError::Quantum(QuantumError::ClassicalBitOutOfRange(
                            *classical_bit,
                        )));
                    }
                    classical_width = classical_width.max(classical_bit + 1);
                }
                CircuitOp::Channel { .. } => {
                    return Err(CudaError::Quantum(
                        QuantumError::OperationRequiresDensityMatrix,
                    ));
                }
                _ => {}
            }
        }
        let original = self.upload(initial_state)?;
        let mut state = self.upload(initial_state)?;
        let mut rng = SmallRng::seeded(seed);
        let mut counts = BTreeMap::new();
        for _ in 0..shots {
            state.copy_from_device(&original)?;
            let mut classical = vec![0; classical_width];
            for operation in circuit.operations() {
                match *operation {
                    CircuitOp::SingleQubit { gate, target } => state.apply_gate(gate, target)?,
                    CircuitOp::Cnot { control, target } => state.apply_cnot(control, target)?,
                    CircuitOp::Cz { first, second } => state.apply_cz(first, second)?,
                    CircuitOp::Swap { first, second } => state.apply_swap(first, second)?,
                    CircuitOp::Measure {
                        target,
                        classical_bit,
                    } => classical[classical_bit] = state.measure(target, &mut rng)?,
                    CircuitOp::ConditionalSingleQubit {
                        classical_bit,
                        value,
                        gate,
                        target,
                    } => {
                        if classical[classical_bit] == value {
                            state.apply_gate(gate, target)?;
                        }
                    }
                    CircuitOp::Channel { .. } => unreachable!("channels were rejected above"),
                }
            }
            *counts.entry(classical).or_insert(0) += 1;
        }
        Ok(ShotResult { shots, counts })
    }

    /// Execute independent shots as one state-vector batch. Measurement
    /// probabilities and collapse are batched on the GPU; sampled uniforms are
    /// assigned in the CPU simulator's shot-major order to preserve seeded
    /// reference results. Noise channels still need trajectory execution.
    pub fn run_circuit_shots_batched(
        &self,
        initial_state: &StateVector,
        circuit: &Circuit,
        shots: usize,
        seed: u64,
    ) -> Result<ShotResult, CudaError> {
        if shots == 0 {
            return Err(CudaError::Quantum(QuantumError::InvalidShotCount));
        }
        let total_amplitudes = initial_state
            .amplitudes()
            .len()
            .checked_mul(shots)
            .ok_or(CudaError::StateTooLarge)?;
        i32::try_from(total_amplitudes).map_err(|_| CudaError::StateTooLarge)?;
        let mut classical_width = 0;
        let mut random_operation_count = 0usize;
        let mut has_noise = false;
        for operation in circuit.operations() {
            match operation {
                CircuitOp::Measure { classical_bit, .. } => {
                    if *classical_bit >= 4096 {
                        return Err(CudaError::Quantum(QuantumError::ClassicalBitOutOfRange(
                            *classical_bit,
                        )));
                    }
                    classical_width = classical_width.max(classical_bit + 1);
                    random_operation_count += 1;
                }
                CircuitOp::ConditionalSingleQubit { classical_bit, .. } => {
                    if *classical_bit >= 4096 {
                        return Err(CudaError::Quantum(QuantumError::ClassicalBitOutOfRange(
                            *classical_bit,
                        )));
                    }
                    classical_width = classical_width.max(classical_bit + 1);
                }
                CircuitOp::Channel { .. } => {
                    random_operation_count += 1;
                    has_noise = true;
                }
                _ => {}
            }
        }

        let draw_count = shots
            .checked_mul(random_operation_count)
            .ok_or(CudaError::StateTooLarge)?;
        let mut rng = SmallRng::seeded(seed);
        let mut draws = vec![0.0; draw_count];
        for shot in 0..shots {
            for operation in 0..random_operation_count {
                draws[shot * random_operation_count + operation] = rng.next_f64();
            }
        }

        let mut initial_states = vec![initial_state.clone(); shots];
        if has_noise {
            for state in &mut initial_states {
                state.normalize().map_err(CudaError::Quantum)?;
            }
        }
        let mut batch = self.upload_batch(&initial_states)?;
        let mut classical = vec![vec![0u8; classical_width]; shots];
        let mut random_operation_index = 0;
        for operation in circuit.operations() {
            match *operation {
                CircuitOp::SingleQubit { gate, target } => batch.apply_gate(gate, target)?,
                CircuitOp::Cnot { control, target } => batch.apply_cnot(control, target)?,
                CircuitOp::Cz { first, second } => batch.apply_cz(first, second)?,
                CircuitOp::Swap { first, second } => batch.apply_swap(first, second)?,
                CircuitOp::Measure {
                    target,
                    classical_bit,
                } => {
                    let uniforms: Vec<_> = (0..shots)
                        .map(|shot| draws[shot * random_operation_count + random_operation_index])
                        .collect();
                    let outcomes = batch.measure_with_uniforms(target, &uniforms)?;
                    for (register, outcome) in classical.iter_mut().zip(outcomes) {
                        register[classical_bit] = outcome;
                    }
                    random_operation_index += 1;
                }
                CircuitOp::ConditionalSingleQubit {
                    classical_bit,
                    value,
                    gate,
                    target,
                } => {
                    let enabled: Vec<_> = classical
                        .iter()
                        .map(|register| u8::from(register[classical_bit] == value))
                        .collect();
                    batch.apply_gate_masked(gate, target, &enabled)?;
                }
                CircuitOp::Channel { channel, target } => {
                    let uniforms: Vec<_> = (0..shots)
                        .map(|shot| draws[shot * random_operation_count + random_operation_index])
                        .collect();
                    batch.apply_channel_trajectory_with_uniforms(channel, target, &uniforms)?;
                    random_operation_index += 1;
                }
            }
        }
        batch.synchronize()?;
        let mut counts = BTreeMap::new();
        for register in classical {
            *counts.entry(register).or_insert(0) += 1;
        }
        Ok(ShotResult { shots, counts })
    }

    pub fn device_ordinal(&self) -> i32 {
        self.inner.device
    }
}

impl CudaDeviceState {
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    pub fn apply_rotation(
        &mut self,
        axis: RotationAxis,
        qubit: usize,
        theta: f64,
    ) -> Result<(), CudaError> {
        self.apply_rotation_chain(&[(axis, qubit, theta)])
    }

    pub fn apply_gate(&mut self, gate: Gate, qubit: usize) -> Result<(), CudaError> {
        validate_gates(
            self.qubits,
            &match gate {
                Gate::Rx(theta) => vec![(RotationAxis::X, qubit, theta)],
                Gate::Ry(theta) => vec![(RotationAxis::Y, qubit, theta)],
                Gate::Rz(theta) => vec![(RotationAxis::Z, qubit, theta)],
                _ => vec![],
            },
        )?;
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let n = i32::try_from(self.amplitudes).map_err(|_| CudaError::StateTooLarge)?;
        let (m00, m01, m10, m11) = gate.matrix();
        launch_matrix(
            self.backend.matrix,
            self.pointer,
            n,
            qubit as i32,
            [m00, m01, m10, m11],
        )?;
        check_driver(unsafe { cuCtxSynchronize() })
    }

    pub fn apply_rotation_chain(
        &mut self,
        gates: &[(RotationAxis, usize, f64)],
    ) -> Result<(), CudaError> {
        validate_gates(self.qubits, gates)?;
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let n = i32::try_from(self.amplitudes).map_err(|_| CudaError::StateTooLarge)?;
        for &(axis, qubit, theta) in gates {
            let axis_arg = match axis {
                RotationAxis::X => 0,
                RotationAxis::Y => 1,
                RotationAxis::Z => 2,
            };
            launch_rotation(
                self.backend.rotation,
                self.pointer,
                n,
                qubit as i32,
                axis_arg,
                theta,
            )?;
        }
        check_driver(unsafe { cuCtxSynchronize() })
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), CudaError> {
        self.apply_two_qubit(0, control, target)
    }

    pub fn apply_cz(&mut self, first: usize, second: usize) -> Result<(), CudaError> {
        self.apply_two_qubit(1, first, second)
    }

    pub fn apply_swap(&mut self, first: usize, second: usize) -> Result<(), CudaError> {
        self.apply_two_qubit(2, first, second)
    }

    fn apply_two_qubit(&mut self, op: i32, first: usize, second: usize) -> Result<(), CudaError> {
        validate_pair(self.qubits, first, second)?;
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let n = i32::try_from(self.amplitudes).map_err(|_| CudaError::StateTooLarge)?;
        launch_two_qubit(
            self.backend.two_qubit,
            self.pointer,
            n,
            first as i32,
            second as i32,
            op,
        )?;
        check_driver(unsafe { cuCtxSynchronize() })
    }

    /// Projectively measure one qubit, sampling the outcome on the host and
    /// reducing probabilities plus collapsing amplitudes on the device.
    pub fn measure(&mut self, qubit: usize, rng: &mut SmallRng) -> Result<u8, CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let n = i32::try_from(self.amplitudes).map_err(|_| CudaError::StateTooLarge)?;
        let blocks = self.amplitudes.div_ceil(256);
        let partial_bytes = blocks
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        if self.reduction_scratch.is_none() {
            let mut pointer = 0;
            check_driver(unsafe { cuMemAlloc_v2(&mut pointer, partial_bytes) })?;
            self.reduction_scratch = Some(pointer);
        }
        let scratch = self.reduction_scratch.expect("scratch was allocated");
        launch_measure_probability(
            self.backend.measure_probability,
            self.pointer,
            n,
            qubit as i32,
            scratch,
            blocks as u32,
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;

        let mut partial_sums = vec![0.0_f64; blocks];
        check_driver(unsafe {
            cuMemcpyDtoH_v2(partial_sums.as_mut_ptr().cast(), scratch, partial_bytes)
        })?;
        let probability_one = partial_sums.iter().sum::<f64>().clamp(0.0, 1.0);
        let outcome = u8::from(rng.next_f64() < probability_one);
        let probability = if outcome == 1 {
            probability_one
        } else {
            1.0 - probability_one
        };
        if probability <= f64::EPSILON {
            return Err(CudaError::Quantum(QuantumError::InvalidProbability(
                probability,
            )));
        }
        launch_measure_collapse(
            self.backend.measure_collapse,
            self.pointer,
            n,
            qubit as i32,
            outcome as i32,
            probability.sqrt().recip(),
            blocks as u32,
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        Ok(outcome)
    }

    fn copy_from_device(&mut self, source: &Self) -> Result<(), CudaError> {
        if self.amplitudes != source.amplitudes || !Rc::ptr_eq(&self.backend, &source.backend) {
            return Err(CudaError::IncompatibleDeviceState);
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let bytes = self
            .amplitudes
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe { cuMemcpyDtoD_v2(self.pointer, source.pointer, bytes) })?;
        check_driver(unsafe { cuCtxSynchronize() })
    }

    pub fn download(&self) -> Result<StateVector, CudaError> {
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let mut amplitudes = vec![Complex64::ZERO; self.amplitudes];
        let bytes = amplitudes
            .len()
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe {
            cuMemcpyDtoH_v2(amplitudes.as_mut_ptr().cast(), self.pointer, bytes)
        })?;
        StateVector::from_amplitudes(amplitudes).map_err(CudaError::Quantum)
    }
}

impl Drop for CudaDeviceState {
    fn drop(&mut self) {
        unsafe {
            if cuCtxSetCurrent(self.backend.context) == 0 {
                cuMemFree_v2(self.pointer);
                if let Some(scratch) = self.reduction_scratch {
                    cuMemFree_v2(scratch);
                }
            }
        }
    }
}

impl CudaDeviceStateBatch {
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Measure the same qubit independently in every batch entry. Only the
    /// per-state probability reductions and sampled outcomes cross the bus;
    /// state amplitudes remain resident and are collapsed in one GPU kernel.
    pub fn measure(&mut self, qubit: usize, rng: &mut SmallRng) -> Result<Vec<u8>, CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        let uniforms: Vec<_> = (0..self.batch_size).map(|_| rng.next_f64()).collect();
        self.measure_with_uniforms(qubit, &uniforms)
    }

    fn measure_with_uniforms(
        &mut self,
        qubit: usize,
        uniforms: &[f64],
    ) -> Result<Vec<u8>, CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        if uniforms.len() != self.batch_size
            || uniforms
                .iter()
                .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
        {
            return Err(CudaError::InvalidBatchSamples);
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let total_blocks = self
            .reduction_blocks_per_state
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        let grid_blocks = u32::try_from(total_blocks).map_err(|_| CudaError::StateTooLarge)?;
        launch_batch_measure_probability(
            self.backend.batch_measure_probability,
            BatchMeasureProbabilityArgs {
                states: self.pointer,
                n: i32::try_from(self.amplitudes_per_state)
                    .map_err(|_| CudaError::StateTooLarge)?,
                batch_size: i32::try_from(self.batch_size).map_err(|_| CudaError::StateTooLarge)?,
                target: qubit as i32,
                blocks_per_state: i32::try_from(self.reduction_blocks_per_state)
                    .map_err(|_| CudaError::StateTooLarge)?,
                partial_sums: self.reduction_scratch,
                grid_blocks,
            },
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;

        let reduction_entries = total_blocks;
        let reduction_bytes = reduction_entries
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        let mut partial_sums = vec![0.0; reduction_entries];
        check_driver(unsafe {
            cuMemcpyDtoH_v2(
                partial_sums.as_mut_ptr().cast(),
                self.reduction_scratch,
                reduction_bytes,
            )
        })?;

        let mut outcomes = Vec::with_capacity(self.batch_size);
        let mut inverse_norms = Vec::with_capacity(self.batch_size);
        for (state_index, state_sums) in partial_sums
            .chunks_exact(self.reduction_blocks_per_state)
            .enumerate()
        {
            let probability_one = state_sums.iter().sum::<f64>().clamp(0.0, 1.0);
            let outcome = u8::from(uniforms[state_index] < probability_one);
            let probability = if outcome == 1 {
                probability_one
            } else {
                1.0 - probability_one
            };
            if probability <= f64::EPSILON {
                return Err(CudaError::Quantum(QuantumError::InvalidProbability(
                    probability,
                )));
            }
            outcomes.push(outcome);
            inverse_norms.push(probability.sqrt().recip());
        }

        check_driver(unsafe {
            cuMemcpyHtoD_v2(
                self.outcomes_scratch,
                outcomes.as_ptr().cast(),
                outcomes.len(),
            )
        })?;
        let inverse_bytes = inverse_norms
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe {
            cuMemcpyHtoD_v2(
                self.inverse_norms_scratch,
                inverse_norms.as_ptr().cast(),
                inverse_bytes,
            )
        })?;
        let total_amplitudes = self
            .amplitudes_per_state
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        launch_batch_measure_collapse(
            self.backend.batch_measure_collapse,
            BatchMeasureCollapseArgs {
                states: self.pointer,
                n: i32::try_from(self.amplitudes_per_state)
                    .map_err(|_| CudaError::StateTooLarge)?,
                batch_size: i32::try_from(self.batch_size).map_err(|_| CudaError::StateTooLarge)?,
                target: qubit as i32,
                outcomes: self.outcomes_scratch,
                inverse_norms: self.inverse_norms_scratch,
                blocks: u32::try_from(total_amplitudes.div_ceil(256))
                    .map_err(|_| CudaError::StateTooLarge)?,
            },
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        Ok(outcomes)
    }

    fn probabilities_one(&mut self, qubit: usize) -> Result<Vec<f64>, CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let total_blocks = self
            .reduction_blocks_per_state
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        launch_batch_measure_probability(
            self.backend.batch_measure_probability,
            BatchMeasureProbabilityArgs {
                states: self.pointer,
                n: i32::try_from(self.amplitudes_per_state)
                    .map_err(|_| CudaError::StateTooLarge)?,
                batch_size: i32::try_from(self.batch_size).map_err(|_| CudaError::StateTooLarge)?,
                target: qubit as i32,
                blocks_per_state: i32::try_from(self.reduction_blocks_per_state)
                    .map_err(|_| CudaError::StateTooLarge)?,
                partial_sums: self.reduction_scratch,
                grid_blocks: u32::try_from(total_blocks).map_err(|_| CudaError::StateTooLarge)?,
            },
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        let bytes = total_blocks
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        let mut partial_sums = vec![0.0; total_blocks];
        check_driver(unsafe {
            cuMemcpyDtoH_v2(
                partial_sums.as_mut_ptr().cast(),
                self.reduction_scratch,
                bytes,
            )
        })?;
        Ok(partial_sums
            .chunks_exact(self.reduction_blocks_per_state)
            .map(|state_sums| state_sums.iter().sum::<f64>().clamp(0.0, 1.0))
            .collect())
    }

    fn apply_channel_trajectory_with_uniforms(
        &mut self,
        channel: NoiseChannel,
        qubit: usize,
        uniforms: &[f64],
    ) -> Result<(), CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        if uniforms.len() != self.batch_size
            || uniforms
                .iter()
                .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
        {
            return Err(CudaError::InvalidBatchSamples);
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let n = i32::try_from(self.amplitudes_per_state).map_err(|_| CudaError::StateTooLarge)?;
        let batch_size = i32::try_from(self.batch_size).map_err(|_| CudaError::StateTooLarge)?;
        let pairs = (self.amplitudes_per_state / 2)
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        let blocks = u32::try_from(pairs.div_ceil(256)).map_err(|_| CudaError::StateTooLarge)?;

        match channel {
            NoiseChannel::Depolarizing { probability } => {
                crate::validate_probability(probability).map_err(CudaError::Quantum)?;
                let branches: Vec<u8> = uniforms
                    .iter()
                    .map(|sample| {
                        if *sample < 1.0 - probability || probability == 0.0 {
                            0
                        } else {
                            (((*sample - (1.0 - probability)) * (3.0 / probability)).floor() as u8)
                                .saturating_add(1)
                                .min(3)
                        }
                    })
                    .collect();
                check_driver(unsafe {
                    cuMemcpyHtoD_v2(
                        self.outcomes_scratch,
                        branches.as_ptr().cast(),
                        branches.len(),
                    )
                })?;
                launch_batch_pauli_trajectory(
                    self.backend.batch_pauli_trajectory,
                    BatchPauliTrajectoryArgs {
                        states: self.pointer,
                        n,
                        batch_size,
                        target: qubit as i32,
                        branches: self.outcomes_scratch,
                        blocks,
                    },
                )?;
            }
            NoiseChannel::AmplitudeDamping { gamma } | NoiseChannel::PhaseDamping { gamma } => {
                crate::validate_probability(gamma).map_err(CudaError::Quantum)?;
                let probabilities_one = self.probabilities_one(qubit)?;
                let mut jumps = Vec::with_capacity(self.batch_size);
                let mut inverse_norms = Vec::with_capacity(self.batch_size);
                for (sample, probability_one) in uniforms.iter().zip(probabilities_one) {
                    let jump_probability = gamma * probability_one;
                    let jump = *sample < jump_probability;
                    let branch_probability = if jump {
                        jump_probability
                    } else {
                        1.0 - jump_probability
                    };
                    if branch_probability <= f64::EPSILON {
                        return Err(CudaError::Quantum(QuantumError::InvalidProbability(
                            branch_probability,
                        )));
                    }
                    jumps.push(u8::from(jump));
                    inverse_norms.push(branch_probability.sqrt().recip());
                }
                check_driver(unsafe {
                    cuMemcpyHtoD_v2(self.outcomes_scratch, jumps.as_ptr().cast(), jumps.len())
                })?;
                let inverse_bytes = inverse_norms
                    .len()
                    .checked_mul(std::mem::size_of::<f64>())
                    .ok_or(CudaError::StateTooLarge)?;
                check_driver(unsafe {
                    cuMemcpyHtoD_v2(
                        self.inverse_norms_scratch,
                        inverse_norms.as_ptr().cast(),
                        inverse_bytes,
                    )
                })?;
                let channel_kind = i32::from(matches!(channel, NoiseChannel::PhaseDamping { .. }));
                launch_batch_damping_trajectory(
                    self.backend.batch_damping_trajectory,
                    BatchDampingTrajectoryArgs {
                        states: self.pointer,
                        n,
                        batch_size,
                        target: qubit as i32,
                        channel: channel_kind,
                        gamma,
                        jumps: self.outcomes_scratch,
                        inverse_norms: self.inverse_norms_scratch,
                        blocks,
                    },
                )?;
            }
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        check_driver(unsafe { cuCtxSynchronize() })
    }

    pub fn copy_from_device(&mut self, source: &Self) -> Result<(), CudaError> {
        if self.amplitudes_per_state != source.amplitudes_per_state
            || self.batch_size != source.batch_size
            || !Rc::ptr_eq(&self.backend, &source.backend)
        {
            return Err(CudaError::IncompatibleDeviceState);
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let bytes = self
            .amplitudes_per_state
            .checked_mul(self.batch_size)
            .and_then(|amplitudes| amplitudes.checked_mul(std::mem::size_of::<Complex64>()))
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe { cuMemcpyDtoD_v2(self.pointer, source.pointer, bytes) })?;
        self.synchronize()
    }

    /// Enqueue the same single-qubit gate for every state. Device execution is
    /// asynchronous; `download` synchronizes and reports deferred failures.
    pub fn apply_gate(&mut self, gate: Gate, qubit: usize) -> Result<(), CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        if let Gate::Rx(theta) | Gate::Ry(theta) | Gate::Rz(theta) = gate
            && !theta.is_finite()
        {
            return Err(CudaError::NonFiniteAngle);
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let n = i32::try_from(self.amplitudes_per_state).map_err(|_| CudaError::StateTooLarge)?;
        let batch_size = i32::try_from(self.batch_size).map_err(|_| CudaError::StateTooLarge)?;
        let pairs = (self.amplitudes_per_state / 2)
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        let blocks = u32::try_from(pairs.div_ceil(256)).map_err(|_| CudaError::StateTooLarge)?;
        let (m00, m01, m10, m11) = gate.matrix();
        launch_batch_matrix(
            self.backend.batch_matrix,
            BatchMatrixArgs {
                state: self.pointer,
                n,
                batch_size,
                target: qubit as i32,
                matrix: [m00, m01, m10, m11],
                blocks,
            },
        )
    }

    fn apply_gate_masked(
        &mut self,
        gate: Gate,
        qubit: usize,
        enabled: &[u8],
    ) -> Result<(), CudaError> {
        if enabled.len() != self.batch_size || enabled.iter().any(|value| *value > 1) {
            return Err(CudaError::InvalidBatchSamples);
        }
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        if let Gate::Rx(theta) | Gate::Ry(theta) | Gate::Rz(theta) = gate
            && !theta.is_finite()
        {
            return Err(CudaError::NonFiniteAngle);
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        check_driver(unsafe {
            cuMemcpyHtoD_v2(
                self.outcomes_scratch,
                enabled.as_ptr().cast(),
                enabled.len(),
            )
        })?;
        let n = i32::try_from(self.amplitudes_per_state).map_err(|_| CudaError::StateTooLarge)?;
        let batch_size = i32::try_from(self.batch_size).map_err(|_| CudaError::StateTooLarge)?;
        let pairs = (self.amplitudes_per_state / 2)
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        let blocks = u32::try_from(pairs.div_ceil(256)).map_err(|_| CudaError::StateTooLarge)?;
        let (m00, m01, m10, m11) = gate.matrix();
        launch_batch_masked_matrix(
            self.backend.batch_masked_matrix,
            BatchMaskedMatrixArgs {
                states: self.pointer,
                n,
                batch_size,
                target: qubit as i32,
                enabled: self.outcomes_scratch,
                matrix: [m00, m01, m10, m11],
                blocks,
            },
        )
    }

    /// Enqueue the same two-qubit operation for every state in the batch.
    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), CudaError> {
        self.apply_two_qubit(0, control, target)
    }

    pub fn apply_cz(&mut self, first: usize, second: usize) -> Result<(), CudaError> {
        self.apply_two_qubit(1, first, second)
    }

    pub fn apply_swap(&mut self, first: usize, second: usize) -> Result<(), CudaError> {
        self.apply_two_qubit(2, first, second)
    }

    fn apply_two_qubit(
        &mut self,
        operation: i32,
        first: usize,
        second: usize,
    ) -> Result<(), CudaError> {
        validate_pair(self.qubits, first, second)?;
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let n = i32::try_from(self.amplitudes_per_state).map_err(|_| CudaError::StateTooLarge)?;
        let batch_size = i32::try_from(self.batch_size).map_err(|_| CudaError::StateTooLarge)?;
        let total = self
            .amplitudes_per_state
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        let blocks = u32::try_from(total.div_ceil(256)).map_err(|_| CudaError::StateTooLarge)?;
        launch_batch_two_qubit(
            self.backend.batch_two_qubit,
            BatchTwoQubitArgs {
                states: self.pointer,
                n,
                batch_size,
                first: first as i32,
                second: second as i32,
                operation,
                blocks,
            },
        )
    }

    pub fn run_circuit(&mut self, circuit: &Circuit) -> Result<(), CudaError> {
        for operation in circuit.operations() {
            match *operation {
                CircuitOp::SingleQubit { gate, target } => self.apply_gate(gate, target)?,
                CircuitOp::Cnot { control, target } => self.apply_cnot(control, target)?,
                CircuitOp::Cz { first, second } => self.apply_cz(first, second)?,
                CircuitOp::Swap { first, second } => self.apply_swap(first, second)?,
                CircuitOp::Channel { .. } => {
                    return Err(CudaError::Quantum(
                        QuantumError::OperationRequiresDensityMatrix,
                    ));
                }
                CircuitOp::Measure { .. } | CircuitOp::ConditionalSingleQubit { .. } => {
                    return Err(CudaError::Quantum(
                        QuantumError::OperationRequiresShotExecution,
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn download(&self) -> Result<Vec<StateVector>, CudaError> {
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let total = self
            .amplitudes_per_state
            .checked_mul(self.batch_size)
            .ok_or(CudaError::StateTooLarge)?;
        let mut amplitudes = vec![Complex64::ZERO; total];
        let bytes = total
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        check_driver(unsafe {
            cuMemcpyDtoH_v2(amplitudes.as_mut_ptr().cast(), self.pointer, bytes)
        })?;
        amplitudes
            .chunks_exact(self.amplitudes_per_state)
            .map(|state| StateVector::from_amplitudes(state.to_vec()).map_err(CudaError::Quantum))
            .collect()
    }
}

impl Drop for CudaDeviceStateBatch {
    fn drop(&mut self) {
        unsafe {
            if cuCtxSetCurrent(self.backend.context) == 0 {
                cuMemFree_v2(self.pointer);
                cuMemFree_v2(self.reduction_scratch);
                cuMemFree_v2(self.outcomes_scratch);
                cuMemFree_v2(self.inverse_norms_scratch);
            }
        }
    }
}

impl CudaDeviceDensityMatrix {
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    pub fn apply_gate(&mut self, gate: Gate, qubit: usize) -> Result<(), CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        if let Gate::Rx(theta) | Gate::Ry(theta) | Gate::Rz(theta) = gate
            && !theta.is_finite()
        {
            return Err(CudaError::NonFiniteAngle);
        }
        let (m00, m01, m10, m11) = gate.matrix();
        self.apply_local_matrix(qubit, [m00, m01, m10, m11])
    }

    fn apply_local_matrix(
        &mut self,
        qubit: usize,
        matrix: [Complex64; 4],
    ) -> Result<(), CudaError> {
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let dimension = i32::try_from(self.dimension).map_err(|_| CudaError::StateTooLarge)?;
        let pairs = self.elements / 2;
        let blocks = u32::try_from(pairs.div_ceil(256)).map_err(|_| CudaError::StateTooLarge)?;
        launch_density_local(
            self.backend.density_local,
            DensityLocalArgs {
                input: self.pointer,
                output: self.scratch,
                dimension,
                target: qubit as i32,
                direction: 0,
                matrix,
                blocks,
            },
        )?;
        launch_density_local(
            self.backend.density_local,
            DensityLocalArgs {
                input: self.scratch,
                output: self.pointer,
                dimension,
                target: qubit as i32,
                direction: 1,
                matrix,
                blocks,
            },
        )?;
        check_driver(unsafe { cuCtxSynchronize() })
    }

    pub fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), CudaError> {
        validate_pair(self.qubits, control, target)?;
        self.apply_permutation(control, target, 0)
    }

    pub fn apply_cz(&mut self, first: usize, second: usize) -> Result<(), CudaError> {
        validate_pair(self.qubits, first, second)?;
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        launch_density_cz(
            self.backend.density_cz,
            self.pointer,
            self.dimension as i32,
            first as i32,
            second as i32,
            (self.elements as u32).div_ceil(256),
        )?;
        check_driver(unsafe { cuCtxSynchronize() })
    }

    pub fn apply_swap(&mut self, first: usize, second: usize) -> Result<(), CudaError> {
        validate_pair(self.qubits, first, second)?;
        self.apply_permutation(first, second, 1)
    }

    fn apply_permutation(
        &mut self,
        first: usize,
        second: usize,
        operation: i32,
    ) -> Result<(), CudaError> {
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        launch_density_permute(
            self.backend.density_permute,
            DensityPermutationArgs {
                input: self.pointer,
                output: self.scratch,
                dimension: self.dimension as i32,
                first: first as i32,
                second: second as i32,
                operation,
                blocks: (self.elements as u32).div_ceil(256),
            },
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        std::mem::swap(&mut self.pointer, &mut self.scratch);
        Ok(())
    }

    pub fn apply_channel(&mut self, channel: NoiseChannel, qubit: usize) -> Result<(), CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        let (kind, parameter) = match channel {
            NoiseChannel::AmplitudeDamping { gamma } => (0, gamma),
            NoiseChannel::PhaseDamping { gamma } => (1, gamma),
            NoiseChannel::Depolarizing { probability } => (2, probability),
        };
        crate::validate_probability(parameter).map_err(CudaError::Quantum)?;
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        launch_density_channel(
            self.backend.density_channel,
            DensityChannelArgs {
                input: self.pointer,
                output: self.scratch,
                dimension: self.dimension as i32,
                target: qubit as i32,
                channel: kind,
                parameter,
                blocks: (self.elements as u32).div_ceil(256),
            },
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        std::mem::swap(&mut self.pointer, &mut self.scratch);
        Ok(())
    }

    /// Measure and collapse a qubit while keeping the density matrix on the GPU.
    /// Only the per-block probability reduction is copied to the host for seeded sampling.
    pub fn measure(&mut self, qubit: usize, rng: &mut SmallRng) -> Result<u8, CudaError> {
        if qubit >= self.qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits: self.qubits,
            }));
        }
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let blocks =
            u32::try_from(self.dimension.div_ceil(256)).map_err(|_| CudaError::StateTooLarge)?;
        launch_density_measure_probability(
            self.backend.density_measure_probability,
            self.pointer,
            self.reduction_scratch,
            self.dimension as i32,
            qubit as i32,
            blocks,
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        let mut partial_sums = vec![0.0f64; blocks as usize];
        let bytes = partial_sums
            .len()
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe {
            cuMemcpyDtoH_v2(
                partial_sums.as_mut_ptr().cast(),
                self.reduction_scratch,
                bytes,
            )
        })?;
        let p_one = partial_sums.iter().sum::<f64>().clamp(0.0, 1.0);
        let outcome = u8::from(rng.next_f64() < p_one);
        let probability = if outcome == 1 { p_one } else { 1.0 - p_one };
        if probability <= f64::EPSILON {
            return Err(CudaError::Quantum(QuantumError::InvalidProbability(
                probability,
            )));
        }
        launch_density_measure_collapse(
            self.backend.density_measure_collapse,
            self.pointer,
            self.dimension as i32,
            qubit as i32,
            outcome as i32,
            1.0 / probability,
            (self.elements as u32).div_ceil(256),
        )?;
        check_driver(unsafe { cuCtxSynchronize() })?;
        Ok(outcome)
    }

    pub fn download(&self) -> Result<DensityMatrix, CudaError> {
        check_driver(unsafe { cuCtxSetCurrent(self.backend.context) })?;
        let mut entries = vec![Complex64::ZERO; self.elements];
        let bytes = self
            .elements
            .checked_mul(std::mem::size_of::<Complex64>())
            .ok_or(CudaError::StateTooLarge)?;
        check_driver(unsafe { cuMemcpyDtoH_v2(entries.as_mut_ptr().cast(), self.pointer, bytes) })?;
        Ok(DensityMatrix::from_backend_entries(entries, self.qubits))
    }
}

impl Drop for CudaDeviceDensityMatrix {
    fn drop(&mut self) {
        unsafe {
            if cuCtxSetCurrent(self.backend.context) == 0 {
                cuMemFree_v2(self.pointer);
                cuMemFree_v2(self.scratch);
                cuMemFree_v2(self.reduction_scratch);
            }
        }
    }
}

fn validate_density_size(qubits: usize) -> Result<(usize, usize), CudaError> {
    if qubits == 0 {
        return Err(CudaError::Quantum(QuantumError::ZeroQubits));
    }
    if qubits >= 16 {
        return Err(CudaError::StateTooLarge);
    }
    let dimension = 1usize << qubits;
    let elements = dimension
        .checked_mul(dimension)
        .ok_or(CudaError::StateTooLarge)?;
    if elements > i32::MAX as usize {
        return Err(CudaError::StateTooLarge);
    }
    Ok((dimension, elements))
}

fn validate_gates(qubits: usize, gates: &[(RotationAxis, usize, f64)]) -> Result<(), CudaError> {
    if qubits >= 31 {
        return Err(CudaError::StateTooLarge);
    }
    for &(_axis, qubit, theta) in gates {
        if qubit >= qubits {
            return Err(CudaError::Quantum(QuantumError::QubitOutOfRange {
                qubit,
                qubits,
            }));
        }
        if !theta.is_finite() {
            return Err(CudaError::NonFiniteAngle);
        }
    }
    Ok(())
}

fn validate_pair(qubits: usize, first: usize, second: usize) -> Result<(), CudaError> {
    if first >= qubits || second >= qubits || first == second {
        return Err(CudaError::Quantum(QuantumError::InvalidQubitPair {
            first,
            second,
        }));
    }
    if first >= 31 || second >= 31 {
        return Err(CudaError::StateTooLarge);
    }
    Ok(())
}

fn launch_two_qubit(
    function: CuFunction,
    device_state: CuDevicePtr,
    n: i32,
    first: i32,
    second: i32,
    op: i32,
) -> Result<(), CudaError> {
    let mut device_arg = device_state;
    let mut n_arg = n;
    let mut first_arg = first;
    let mut second_arg = second;
    let mut op_arg = op;
    let mut args = [
        (&mut device_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut first_arg as *mut i32).cast(),
        (&mut second_arg as *mut i32).cast(),
        (&mut op_arg as *mut i32).cast(),
    ];
    let blocks = (n as usize).div_ceil(256) as u32;
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

fn launch_matrix(
    function: CuFunction,
    device_state: CuDevicePtr,
    n: i32,
    target: i32,
    matrix: [Complex64; 4],
) -> Result<(), CudaError> {
    let mut device_arg = device_state;
    let mut n_arg = n;
    let mut target_arg = target;
    let mut values = matrix;
    let mut args = [
        (&mut device_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut values[0] as *mut Complex64).cast(),
        (&mut values[1] as *mut Complex64).cast(),
        (&mut values[2] as *mut Complex64).cast(),
        (&mut values[3] as *mut Complex64).cast(),
    ];
    let blocks = (n as usize).div_ceil(256) as u32;
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct BatchMatrixArgs {
    state: CuDevicePtr,
    n: i32,
    batch_size: i32,
    target: i32,
    matrix: [Complex64; 4],
    blocks: u32,
}

fn launch_batch_matrix(function: CuFunction, dispatch: BatchMatrixArgs) -> Result<(), CudaError> {
    let mut state_arg = dispatch.state;
    let mut n_arg = dispatch.n;
    let mut batch_arg = dispatch.batch_size;
    let mut target_arg = dispatch.target;
    let mut matrix_arg = dispatch.matrix;
    let mut args = [
        (&mut state_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut batch_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut matrix_arg[0] as *mut Complex64).cast(),
        (&mut matrix_arg[1] as *mut Complex64).cast(),
        (&mut matrix_arg[2] as *mut Complex64).cast(),
        (&mut matrix_arg[3] as *mut Complex64).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct BatchMaskedMatrixArgs {
    states: CuDevicePtr,
    n: i32,
    batch_size: i32,
    target: i32,
    enabled: CuDevicePtr,
    matrix: [Complex64; 4],
    blocks: u32,
}

fn launch_batch_masked_matrix(
    function: CuFunction,
    dispatch: BatchMaskedMatrixArgs,
) -> Result<(), CudaError> {
    let mut states_arg = dispatch.states;
    let mut n_arg = dispatch.n;
    let mut batch_arg = dispatch.batch_size;
    let mut target_arg = dispatch.target;
    let mut enabled_arg = dispatch.enabled;
    let mut matrix_arg = dispatch.matrix;
    let mut args = [
        (&mut states_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut batch_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut enabled_arg as *mut CuDevicePtr).cast(),
        (&mut matrix_arg[0] as *mut Complex64).cast(),
        (&mut matrix_arg[1] as *mut Complex64).cast(),
        (&mut matrix_arg[2] as *mut Complex64).cast(),
        (&mut matrix_arg[3] as *mut Complex64).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct BatchTwoQubitArgs {
    states: CuDevicePtr,
    n: i32,
    batch_size: i32,
    first: i32,
    second: i32,
    operation: i32,
    blocks: u32,
}

fn launch_batch_two_qubit(
    function: CuFunction,
    dispatch: BatchTwoQubitArgs,
) -> Result<(), CudaError> {
    let mut states_arg = dispatch.states;
    let mut n_arg = dispatch.n;
    let mut batch_arg = dispatch.batch_size;
    let mut first_arg = dispatch.first;
    let mut second_arg = dispatch.second;
    let mut operation_arg = dispatch.operation;
    let mut args = [
        (&mut states_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut batch_arg as *mut i32).cast(),
        (&mut first_arg as *mut i32).cast(),
        (&mut second_arg as *mut i32).cast(),
        (&mut operation_arg as *mut i32).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct BatchMeasureProbabilityArgs {
    states: CuDevicePtr,
    n: i32,
    batch_size: i32,
    target: i32,
    blocks_per_state: i32,
    partial_sums: CuDevicePtr,
    grid_blocks: u32,
}

fn launch_batch_measure_probability(
    function: CuFunction,
    dispatch: BatchMeasureProbabilityArgs,
) -> Result<(), CudaError> {
    let mut states_arg = dispatch.states;
    let mut n_arg = dispatch.n;
    let mut batch_size_arg = dispatch.batch_size;
    let mut target_arg = dispatch.target;
    let mut blocks_per_state_arg = dispatch.blocks_per_state;
    let mut partial_sums_arg = dispatch.partial_sums;
    let mut args = [
        (&mut states_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut batch_size_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut blocks_per_state_arg as *mut i32).cast(),
        (&mut partial_sums_arg as *mut CuDevicePtr).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.grid_blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct BatchMeasureCollapseArgs {
    states: CuDevicePtr,
    n: i32,
    batch_size: i32,
    target: i32,
    outcomes: CuDevicePtr,
    inverse_norms: CuDevicePtr,
    blocks: u32,
}

fn launch_batch_measure_collapse(
    function: CuFunction,
    dispatch: BatchMeasureCollapseArgs,
) -> Result<(), CudaError> {
    let mut states_arg = dispatch.states;
    let mut n_arg = dispatch.n;
    let mut batch_size_arg = dispatch.batch_size;
    let mut target_arg = dispatch.target;
    let mut outcomes_arg = dispatch.outcomes;
    let mut inverse_norms_arg = dispatch.inverse_norms;
    let mut args = [
        (&mut states_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut batch_size_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut outcomes_arg as *mut CuDevicePtr).cast(),
        (&mut inverse_norms_arg as *mut CuDevicePtr).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct BatchDampingTrajectoryArgs {
    states: CuDevicePtr,
    n: i32,
    batch_size: i32,
    target: i32,
    channel: i32,
    gamma: f64,
    jumps: CuDevicePtr,
    inverse_norms: CuDevicePtr,
    blocks: u32,
}

fn launch_batch_damping_trajectory(
    function: CuFunction,
    dispatch: BatchDampingTrajectoryArgs,
) -> Result<(), CudaError> {
    let mut states_arg = dispatch.states;
    let mut n_arg = dispatch.n;
    let mut batch_arg = dispatch.batch_size;
    let mut target_arg = dispatch.target;
    let mut channel_arg = dispatch.channel;
    let mut gamma_arg = dispatch.gamma;
    let mut jumps_arg = dispatch.jumps;
    let mut inverse_norms_arg = dispatch.inverse_norms;
    let mut args = [
        (&mut states_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut batch_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut channel_arg as *mut i32).cast(),
        (&mut gamma_arg as *mut f64).cast(),
        (&mut jumps_arg as *mut CuDevicePtr).cast(),
        (&mut inverse_norms_arg as *mut CuDevicePtr).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct BatchPauliTrajectoryArgs {
    states: CuDevicePtr,
    n: i32,
    batch_size: i32,
    target: i32,
    branches: CuDevicePtr,
    blocks: u32,
}

fn launch_batch_pauli_trajectory(
    function: CuFunction,
    dispatch: BatchPauliTrajectoryArgs,
) -> Result<(), CudaError> {
    let mut states_arg = dispatch.states;
    let mut n_arg = dispatch.n;
    let mut batch_arg = dispatch.batch_size;
    let mut target_arg = dispatch.target;
    let mut branches_arg = dispatch.branches;
    let mut args = [
        (&mut states_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut batch_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut branches_arg as *mut CuDevicePtr).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct DensityLocalArgs {
    input: CuDevicePtr,
    output: CuDevicePtr,
    dimension: i32,
    target: i32,
    direction: i32,
    matrix: [Complex64; 4],
    blocks: u32,
}

fn launch_density_local(function: CuFunction, dispatch: DensityLocalArgs) -> Result<(), CudaError> {
    let mut input_arg = dispatch.input;
    let mut output_arg = dispatch.output;
    let mut dimension_arg = dispatch.dimension;
    let mut target_arg = dispatch.target;
    let mut direction_arg = dispatch.direction;
    let mut matrix_arg = dispatch.matrix;
    let mut args = [
        (&mut input_arg as *mut CuDevicePtr).cast(),
        (&mut output_arg as *mut CuDevicePtr).cast(),
        (&mut dimension_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut direction_arg as *mut i32).cast(),
        (&mut matrix_arg[0] as *mut Complex64).cast(),
        (&mut matrix_arg[1] as *mut Complex64).cast(),
        (&mut matrix_arg[2] as *mut Complex64).cast(),
        (&mut matrix_arg[3] as *mut Complex64).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct DensityPermutationArgs {
    input: CuDevicePtr,
    output: CuDevicePtr,
    dimension: i32,
    first: i32,
    second: i32,
    operation: i32,
    blocks: u32,
}

fn launch_density_permute(
    function: CuFunction,
    dispatch: DensityPermutationArgs,
) -> Result<(), CudaError> {
    let mut input_arg = dispatch.input;
    let mut output_arg = dispatch.output;
    let mut dimension_arg = dispatch.dimension;
    let mut first_arg = dispatch.first;
    let mut second_arg = dispatch.second;
    let mut operation_arg = dispatch.operation;
    let mut args = [
        (&mut input_arg as *mut CuDevicePtr).cast(),
        (&mut output_arg as *mut CuDevicePtr).cast(),
        (&mut dimension_arg as *mut i32).cast(),
        (&mut first_arg as *mut i32).cast(),
        (&mut second_arg as *mut i32).cast(),
        (&mut operation_arg as *mut i32).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

fn launch_density_cz(
    function: CuFunction,
    state: CuDevicePtr,
    dimension: i32,
    first: i32,
    second: i32,
    blocks: u32,
) -> Result<(), CudaError> {
    let mut state_arg = state;
    let mut dimension_arg = dimension;
    let mut first_arg = first;
    let mut second_arg = second;
    let mut args = [
        (&mut state_arg as *mut CuDevicePtr).cast(),
        (&mut dimension_arg as *mut i32).cast(),
        (&mut first_arg as *mut i32).cast(),
        (&mut second_arg as *mut i32).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

struct DensityChannelArgs {
    input: CuDevicePtr,
    output: CuDevicePtr,
    dimension: i32,
    target: i32,
    channel: i32,
    parameter: f64,
    blocks: u32,
}

fn launch_density_channel(
    function: CuFunction,
    dispatch: DensityChannelArgs,
) -> Result<(), CudaError> {
    let mut input_arg = dispatch.input;
    let mut output_arg = dispatch.output;
    let mut dimension_arg = dispatch.dimension;
    let mut target_arg = dispatch.target;
    let mut channel_arg = dispatch.channel;
    let mut parameter_arg = dispatch.parameter;
    let mut args = [
        (&mut input_arg as *mut CuDevicePtr).cast(),
        (&mut output_arg as *mut CuDevicePtr).cast(),
        (&mut dimension_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut channel_arg as *mut i32).cast(),
        (&mut parameter_arg as *mut f64).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            dispatch.blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

fn launch_density_measure_probability(
    function: CuFunction,
    density: CuDevicePtr,
    sums: CuDevicePtr,
    dimension: i32,
    target: i32,
    blocks: u32,
) -> Result<(), CudaError> {
    let mut density_arg = density;
    let mut dimension_arg = dimension;
    let mut target_arg = target;
    let mut sums_arg = sums;
    let mut args = [
        (&mut density_arg as *mut CuDevicePtr).cast(),
        (&mut dimension_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut sums_arg as *mut CuDevicePtr).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

fn launch_density_measure_collapse(
    function: CuFunction,
    density: CuDevicePtr,
    dimension: i32,
    target: i32,
    outcome: i32,
    inverse_probability: f64,
    blocks: u32,
) -> Result<(), CudaError> {
    let mut density_arg = density;
    let mut dimension_arg = dimension;
    let mut target_arg = target;
    let mut outcome_arg = outcome;
    let mut inverse_probability_arg = inverse_probability;
    let mut args = [
        (&mut density_arg as *mut CuDevicePtr).cast(),
        (&mut dimension_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut outcome_arg as *mut i32).cast(),
        (&mut inverse_probability_arg as *mut f64).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

fn launch_measure_probability(
    function: CuFunction,
    device_state: CuDevicePtr,
    n: i32,
    target: i32,
    partial_sums: CuDevicePtr,
    blocks: u32,
) -> Result<(), CudaError> {
    let mut state_arg = device_state;
    let mut n_arg = n;
    let mut target_arg = target;
    let mut sums_arg = partial_sums;
    let mut args = [
        (&mut state_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut sums_arg as *mut CuDevicePtr).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

fn launch_measure_collapse(
    function: CuFunction,
    device_state: CuDevicePtr,
    n: i32,
    target: i32,
    outcome: i32,
    inverse_norm: f64,
    blocks: u32,
) -> Result<(), CudaError> {
    let mut state_arg = device_state;
    let mut n_arg = n;
    let mut target_arg = target;
    let mut outcome_arg = outcome;
    let mut inverse_norm_arg = inverse_norm;
    let mut args = [
        (&mut state_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut outcome_arg as *mut i32).cast(),
        (&mut inverse_norm_arg as *mut f64).cast(),
    ];
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

fn launch_rotation(
    function: CuFunction,
    device_state: CuDevicePtr,
    n: i32,
    target: i32,
    axis: i32,
    theta: f64,
) -> Result<(), CudaError> {
    let mut device_arg = device_state;
    let mut n_arg = n;
    let mut target_arg = target;
    let mut axis_arg = axis;
    let mut theta_arg = theta;
    let mut args = [
        (&mut device_arg as *mut CuDevicePtr).cast(),
        (&mut n_arg as *mut i32).cast(),
        (&mut target_arg as *mut i32).cast(),
        (&mut axis_arg as *mut i32).cast(),
        (&mut theta_arg as *mut f64).cast(),
    ];
    let blocks = (n as usize).div_ceil(256) as u32;
    check_driver(unsafe {
        cuLaunchKernel(
            function,
            blocks,
            1,
            1,
            256,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        )
    })
}

impl Drop for CudaBackendInner {
    fn drop(&mut self) {
        unsafe {
            cuCtxSetCurrent(self.context);
            cuModuleUnload(self.module);
            cuDevicePrimaryCtxRelease_v2(self.device);
        }
    }
}

#[derive(Debug)]
pub enum CudaError {
    Driver { code: i32, message: String },
    Nvrtc { code: i32, message: String },
    Quantum(QuantumError),
    NonFiniteAngle,
    StateTooLarge,
    EmptyBatch,
    IncompatibleBatch,
    InvalidBatchSamples,
    IncompatibleDeviceState,
}
impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver { code, message } => write!(f, "CUDA driver error {code}: {message}"),
            Self::Nvrtc { code, message } => write!(f, "NVRTC error {code}: {message}"),
            Self::Quantum(error) => write!(f, "{error}"),
            Self::NonFiniteAngle => write!(f, "rotation angle must be finite"),
            Self::StateTooLarge => write!(f, "state vector is too large for the CUDA kernel"),
            Self::EmptyBatch => write!(f, "a CUDA state batch cannot be empty"),
            Self::IncompatibleBatch => {
                write!(f, "all states in a CUDA batch must have equal dimensions")
            }
            Self::InvalidBatchSamples => {
                write!(f, "batch measurement needs one uniform sample per state")
            }
            Self::IncompatibleDeviceState => {
                write!(
                    f,
                    "device states must share a backend and have equal dimensions"
                )
            }
        }
    }
}
impl std::error::Error for CudaError {}

fn compile_ptx(major: i32, minor: i32) -> Result<Vec<u8>, CudaError> {
    let source = CString::new(KERNEL).unwrap();
    let name = CString::new("quantumforge.cu").unwrap();
    let mut program = ptr::null_mut();
    check_nvrtc(unsafe {
        nvrtcCreateProgram(
            &mut program,
            source.as_ptr(),
            name.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        )
    })?;
    let architecture = CString::new(format!("--gpu-architecture=compute_{major}{minor}")).unwrap();
    let options = [architecture.as_ptr()];
    let compile_status = unsafe { nvrtcCompileProgram(program, 1, options.as_ptr()) };
    if compile_status != 0 {
        let mut log_size = 0usize;
        unsafe { nvrtcGetProgramLogSize(program, &mut log_size) };
        let mut log = vec![0u8; log_size.max(1)];
        unsafe {
            nvrtcGetProgramLog(program, log.as_mut_ptr().cast());
            nvrtcDestroyProgram(&mut program);
        }
        return Err(CudaError::Nvrtc {
            code: compile_status,
            message: String::from_utf8_lossy(&log)
                .trim_end_matches('\0')
                .to_string(),
        });
    }
    let mut size = 0usize;
    if let Err(error) = check_nvrtc(unsafe { nvrtcGetPTXSize(program, &mut size) }) {
        unsafe { nvrtcDestroyProgram(&mut program) };
        return Err(error);
    }
    let mut ptx = vec![0u8; size];
    let result = check_nvrtc(unsafe { nvrtcGetPTX(program, ptx.as_mut_ptr().cast()) });
    unsafe { nvrtcDestroyProgram(&mut program) };
    result?;
    Ok(ptx)
}

fn check_driver(code: i32) -> Result<(), CudaError> {
    if code == 0 {
        return Ok(());
    }
    let mut text = ptr::null();
    unsafe { cuGetErrorString(code, &mut text) };
    let message = if text.is_null() {
        "unknown driver error".to_string()
    } else {
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    };
    Err(CudaError::Driver { code, message })
}
fn check_nvrtc(code: i32) -> Result<(), CudaError> {
    if code == 0 {
        return Ok(());
    }
    let text = unsafe { nvrtcGetErrorString(code) };
    let message = if text.is_null() {
        "unknown NVRTC error".to_string()
    } else {
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    };
    Err(CudaError::Nvrtc { code, message })
}
