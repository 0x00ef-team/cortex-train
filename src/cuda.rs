//! Minimal CUDA driver + NVRTC backend for GPU training.
//!
//! No Rust dependencies: the driver and NVRTC are loaded with `dlopen`.
//! This is the first device-tensor layer for CUDA backward passes. It exposes
//! matmul forward/backward, MSE forward/backward and an AdamW update kernel.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::rc::Rc;

const CUDA_SUCCESS: i32 = 0;
type Result<T> = std::result::Result<T, Error>;

type CuInit = unsafe extern "C" fn(u32) -> i32;
type CuDeviceGet = unsafe extern "C" fn(*mut i32, i32) -> i32;
type CuCtxCreate = unsafe extern "C" fn(*mut u64, u32, i32) -> i32;
type CuCtxSync = unsafe extern "C" fn() -> i32;
type CuMemAlloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFree = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyH2D = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyD2H = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuModuleLoad = unsafe extern "C" fn(*mut u64, *const c_void) -> i32;
type CuModuleFn = unsafe extern "C" fn(*mut u64, u64, *const c_char) -> i32;
type CuLaunch = unsafe extern "C" fn(
    u64, u32, u32, u32, u32, u32, u32, u32, u64,
    *mut *mut c_void, *mut c_void,
) -> i32;
type NvrtcCreate = unsafe extern "C" fn(
    *mut u64, *const c_char, *const c_char, c_int,
    *const *const c_char, *const *const c_char,
) -> i32;
type NvrtcCompile = unsafe extern "C" fn(u64, c_int, *const *const c_char) -> i32;
type NvrtcPtxSize = unsafe extern "C" fn(u64, *mut usize) -> i32;
type NvrtcPtx = unsafe extern "C" fn(u64, *mut c_char) -> i32;
type NvrtcLogSize = unsafe extern "C" fn(u64, *mut usize) -> i32;
type NvrtcLog = unsafe extern "C" fn(u64, *mut c_char) -> i32;
type NvrtcDestroy = unsafe extern "C" fn(*mut u64) -> i32;

extern "C" {
    fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

unsafe fn sym<T: Copy>(h: *mut c_void, name: &str) -> Result<T> {
    let n = CString::new(name).unwrap();
    let p = dlsym(h, n.as_ptr());
    if p.is_null() {
        return Err(Error { msg: format!("missing symbol {name}") });
    }
    Ok(std::mem::transmute_copy(&p))
}

unsafe fn load_lib(name: &str) -> Result<*mut c_void> {
    let c = CString::new(name).unwrap();
    let h = dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
    if h.is_null() {
        let e = dlerror();
        let msg = if e.is_null() { "unknown".into() }
            else { CStr::from_ptr(e).to_string_lossy().into_owned() };
        return Err(Error { msg: format!("dlopen {name}: {msg}") });
    }
    Ok(h)
}

const KERNELS: &str = r#"
extern "C" __global__ void matmul_fwd(
    const float* __restrict__ W, const float* __restrict__ x,
    float* __restrict__ y, int m, int k)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    float s = 0.f;
    for (int i = 0; i < k; ++i) s += W[r*k + i] * x[i];
    y[r] = s;
}

extern "C" __global__ void matmul_bwd_w(
    const float* __restrict__ x, const float* __restrict__ dy,
    float* __restrict__ dW, int m, int k)
{
    int r = blockIdx.x;
    if (r >= m) return;
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        dW[r*k + i] = x[i] * dy[r];
}

extern "C" __global__ void matmul_bwd_x(
    const float* __restrict__ W, const float* __restrict__ dy,
    float* __restrict__ dx, int m, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    float s = 0.f;
    for (int r = 0; r < m; ++r) s += W[r*k + i] * dy[r];
    dx[i] = s;
}

extern "C" __global__ void mse_fwd(
    const float* __restrict__ y, const float* __restrict__ t,
    float* __restrict__ out, int n)
{
    __shared__ float acc[256];
    int tid = threadIdx.x;
    float s = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = y[i] - t[i];
        s += d * d;
    }
    acc[tid] = s;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = acc[0] / (float)n;
}

extern "C" __global__ void mse_bwd(
    const float* __restrict__ y, const float* __restrict__ t,
    float* __restrict__ dy, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dy[i] = 2.f * (y[i] - t[i]) / (float)n;
}


extern "C" __global__ void matmul_batch_fwd(
    const float* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, int b, int m, int k)
{
    int br = blockIdx.x * blockDim.x + threadIdx.x;
    int total = b * m;
    if (br >= total) return;
    int bi = br / m, r = br % m;
    float s = 0.f;
    for (int i = 0; i < k; ++i) s += W[r*k + i] * X[bi*k + i];
    Y[br] = s;
}

extern "C" __global__ void matmul_batch_bwd_w(
    const float* __restrict__ X, const float* __restrict__ dY,
    float* __restrict__ dW, int b, int m, int k)
{
    int r = blockIdx.x;
    if (r >= m) return;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float s = 0.f;
        for (int n = 0; n < b; ++n) s += X[n*k + i] * dY[n*m + r];
        dW[r*k + i] = s;
    }
}

extern "C" __global__ void matmul_batch_bwd_x(
    const float* __restrict__ W, const float* __restrict__ dY,
    float* __restrict__ dX, int b, int m, int k)
{
    int bi = blockIdx.x;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float s = 0.f;
        for (int r = 0; r < m; ++r) s += W[r*k + i] * dY[bi*m + r];
        dX[bi*k + i] = s;
    }
}

extern "C" __global__ void bias_add_fwd(
    const float* __restrict__ X, const float* __restrict__ bias,
    float* __restrict__ Y, int b, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < b*d) Y[i] = X[i] + bias[i % d];
}

extern "C" __global__ void bias_bwd(
    const float* __restrict__ dY, float* __restrict__ dB, int b, int d)
{
    for (int i = blockIdx.x; i < d; i += gridDim.x) {
        float s = 0.f;
        for (int n = 0; n < b; ++n) s += dY[n*d + i];
        dB[i] = s / (float)b;
    }
}

extern "C" __global__ void silu_fwd(
    const float* __restrict__ x, float* __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.f + expf(-x[i]));
}

extern "C" __global__ void silu_bwd(
    const float* __restrict__ x, const float* __restrict__ dy,
    float* __restrict__ dx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sig = 1.f / (1.f + expf(-x[i]));
    dx[i] = dy[i] * sig * (1.f + x[i] * (1.f - sig));
}

extern "C" __global__ void ce_fwd(
    const float* __restrict__ logits, const int* __restrict__ target,
    float* __restrict__ out, int n, int c)
{
    __shared__ float acc[256];
    int tid = threadIdx.x;
    float loss = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        float mx = logits[i*c];
        for (int j = 1; j < c; ++j) mx = fmaxf(mx, logits[i*c+j]);
        float sum = 0.f;
        for (int j = 0; j < c; ++j) sum += expf(logits[i*c+j] - mx);
        loss += mx + logf(sum) - logits[i*c + target[i]];
    }
    acc[tid] = loss;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) acc[tid] += acc[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[0] = acc[0] / (float)n;
}

extern "C" __global__ void ce_bwd(
    const float* __restrict__ logits, const int* __restrict__ target,
    float* __restrict__ dlogits, int n, int c)
{
    int i = blockIdx.x;
    if (i >= n) return;
    const float* row = logits + i*c;
    float* drow = dlogits + i*c;
    float mx = row[0];
    for (int j = 1; j < c; ++j) mx = fmaxf(mx, row[j]);
    float sum = 0.f;
    for (int j = 0; j < c; ++j) { drow[j] = expf(row[j] - mx); sum += drow[j]; }
    for (int j = 0; j < c; ++j) drow[j] = drow[j] / sum / (float)n;
    drow[target[i]] -= 1.f / (float)n;
}

extern "C" __global__ void adamw_step(
    float* __restrict__ p, float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v,
    int n, float lr, float wd, float b1, float b2, float eps,
    float c1, float c2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gr = g[i];
    m[i] = b1 * m[i] + (1.f - b1) * gr;
    v[i] = b2 * v[i] + (1.f - b2) * gr * gr;
    p[i] -= lr * (m[i] / c1) / (sqrtf(v[i] / c2) + eps) + lr * wd * p[i];
    g[i] = 0.f;
}
"#;

pub struct Error {
    msg: String,
}
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.msg) }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.msg) }
}
impl std::error::Error for Error {}

struct Api {
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    h2d: CuMemcpyH2D,
    d2h: CuMemcpyD2H,
    launch: CuLaunch,
    sync: CuCtxSync,
    fwd: u64,
    bw_w: u64,
    bw_x: u64,
    mse: u64,
    mse_bwd: u64,
    adam: u64,
    mm_fwd: u64,
    mm_bw_w: u64,
    mm_bw_x: u64,
    bias_add: u64,
    bias_bwd: u64,
    silu_fwd: u64,
    silu_bwd: u64,
    ce_fwd: u64,
    ce_bwd: u64,
}

pub struct Cuda {
    api: Rc<Api>,
}

impl Cuda {
    pub fn new() -> std::result::Result<Self, Error> {
        unsafe {
            let cu = load_lib("libcuda.so.1")?;
            let nv = load_lib("libnvrtc.so.12")?;
            let init: CuInit = sym(cu, "cuInit")?;
            let dev_get: CuDeviceGet = sym(cu, "cuDeviceGet")?;
            let ctx: CuCtxCreate = sym(cu, "cuCtxCreate_v2")?;
            let sync: CuCtxSync = sym(cu, "cuCtxSynchronize")?;
            let mem_alloc: CuMemAlloc = sym(cu, "cuMemAlloc_v2")?;
            let mem_free: CuMemFree = sym(cu, "cuMemFree_v2")?;
            let h2d: CuMemcpyH2D = sym(cu, "cuMemcpyHtoD_v2")?;
            let d2h: CuMemcpyD2H = sym(cu, "cuMemcpyDtoH_v2")?;
            let module_load: CuModuleLoad = sym(cu, "cuModuleLoadData")?;
            let module_fn: CuModuleFn = sym(cu, "cuModuleGetFunction")?;
            let launch: CuLaunch = sym(cu, "cuLaunchKernel")?;

            let create: NvrtcCreate = sym(nv, "nvrtcCreateProgram")?;
            let compile: NvrtcCompile = sym(nv, "nvrtcCompileProgram")?;
            let ptx_size: NvrtcPtxSize = sym(nv, "nvrtcGetPTXSize")?;
            let ptx_get: NvrtcPtx = sym(nv, "nvrtcGetPTX")?;
            let log_size: NvrtcLogSize = sym(nv, "nvrtcGetProgramLogSize")?;
            let log_get: NvrtcLog = sym(nv, "nvrtcGetProgramLog")?;
            let destroy: NvrtcDestroy = sym(nv, "nvrtcDestroyProgram")?;

            check(init(0), "cuInit")?;
            let mut device = 0i32;
            check(dev_get(&mut device, 0), "cuDeviceGet")?;
            let mut ctx_handle = 0u64;
            check(ctx(&mut ctx_handle, 0, device), "cuCtxCreate")?;

            let src = CString::new(KERNELS).unwrap();
            let name = CString::new("train.cu").unwrap();
            let mut prog = 0u64;
            check(create(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()), "nvrtcCreate")?;
            if compile(prog, 0, std::ptr::null()) != CUDA_SUCCESS {
                let mut n = 0usize;
                log_size(prog, &mut n);
                let mut log = vec![0i8; n + 1];
                log_get(prog, log.as_mut_ptr());
                let msg = CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();
                return Err(Error { msg: format!("NVRTC: {msg}") });
            }
            let mut n = 0usize;
            check(ptx_size(prog, &mut n), "ptx size")?;
            let mut ptx = vec![0i8; n + 1];
            check(ptx_get(prog, ptx.as_mut_ptr()), "ptx get")?;
            destroy(&mut prog);
            let mut module = 0u64;
            check(module_load(&mut module, ptx.as_ptr() as *const c_void), "module load")?;
            let fn_handle = |n: &str| -> std::result::Result<u64, Error> {
                let c = CString::new(n).unwrap();
                let mut h = 0u64;
                check(module_fn(&mut h, module, c.as_ptr()), n)?;
                Ok(h)
            };
            Ok(Self { api: Rc::new(Api {
                mem_alloc, mem_free, h2d, d2h, launch, sync,
                fwd: fn_handle("matmul_fwd")?,
                bw_w: fn_handle("matmul_bwd_w")?,
                bw_x: fn_handle("matmul_bwd_x")?,
                mse: fn_handle("mse_fwd")?,
                mse_bwd: fn_handle("mse_bwd")?,
                adam: fn_handle("adamw_step")?,
                mm_fwd: fn_handle("matmul_batch_fwd")?,
                mm_bw_w: fn_handle("matmul_batch_bwd_w")?,
                mm_bw_x: fn_handle("matmul_batch_bwd_x")?,
                bias_add: fn_handle("bias_add_fwd")?,
                bias_bwd: fn_handle("bias_bwd")?,
                silu_fwd: fn_handle("silu_fwd")?,
                silu_bwd: fn_handle("silu_bwd")?,
                ce_fwd: fn_handle("ce_fwd")?,
                ce_bwd: fn_handle("ce_bwd")?,
            }) })
        }
    }

    pub fn sync(&self) { unsafe { (self.api.sync)(); } }
}

fn check(code: i32, what: &str) -> std::result::Result<(), Error> {
    if code == CUDA_SUCCESS { Ok(()) }
    else { Err(Error { msg: format!("{what}: CUDA error {code}") }) }
}

pub struct DeviceTensor {
    ptr: u64,
    pub len: usize,
    api: Rc<Api>,
}

impl DeviceTensor {
    pub fn from_i32(dev: &Cuda, data: &[i32]) -> std::result::Result<Self, Error> {
        let t = Self::zeros(dev, data.len())?;
        unsafe { check((dev.api.h2d)(t.ptr, data.as_ptr() as *const c_void, data.len() * 4), "HtoD i32")?; }
        Ok(t)
    }

    pub fn from_slice(dev: &Cuda, data: &[f32]) -> std::result::Result<Self, Error> {
        let t = Self::zeros(dev, data.len())?;
        t.copy_from(data)?;
        Ok(t)
    }

    pub fn zeros(dev: &Cuda, len: usize) -> std::result::Result<Self, Error> {
        let mut ptr = 0u64;
        unsafe { check((dev.api.mem_alloc)(&mut ptr, len * 4), "cuMemAlloc")?; }
        Ok(Self { ptr, len, api: dev.api.clone() })
    }

    pub fn copy_from(&self, data: &[f32]) -> std::result::Result<(), Error> {
        assert_eq!(data.len(), self.len);
        unsafe { check((self.api.h2d)(self.ptr, data.as_ptr() as *const c_void, data.len() * 4), "HtoD") }
    }

    pub fn to_vec(&self) -> std::result::Result<Vec<f32>, Error> {
        let mut out = vec![0f32; self.len];
        unsafe { check((self.api.d2h)(out.as_mut_ptr() as *mut c_void, self.ptr, self.len * 4), "DtoH")?; }
        Ok(out)
    }
}

impl Drop for DeviceTensor {
    fn drop(&mut self) { unsafe { (self.api.mem_free)(self.ptr); } }
}

fn launch(api: &Api, func: u64, grid: u32, block: u32, params: &mut [*mut c_void]) -> std::result::Result<(), Error> {
    unsafe {
        check((api.launch)(func, grid, 1, 1, block, 1, 1, 0, 0, params.as_mut_ptr(), std::ptr::null_mut()), "launch")
    }
}

impl Cuda {
    pub fn matmul_fwd(&self, w: &DeviceTensor, x: &DeviceTensor, y: &DeviceTensor, m: usize, k: usize) -> std::result::Result<(), Error> {
        let (mm, kk) = (m as i32, k as i32);
        let mut p: [*mut c_void; 5] = [
            &w.ptr as *const u64 as *mut c_void, &x.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void, &mm as *const i32 as *mut c_void, &kk as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.fwd, m.div_ceil(128) as u32, 128, &mut p)
    }

    pub fn matmul_bwd(&self, w: &DeviceTensor, x: &DeviceTensor, dy: &DeviceTensor,
                      dw: &DeviceTensor, dx: &DeviceTensor, m: usize, k: usize) -> std::result::Result<(), Error> {
        let (mm, kk) = (m as i32, k as i32);
        let mut p: [*mut c_void; 5] = [
            &x.ptr as *const u64 as *mut c_void, &dy.ptr as *const u64 as *mut c_void,
            &dw.ptr as *const u64 as *mut c_void, &mm as *const i32 as *mut c_void, &kk as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.bw_w, m as u32, 128, &mut p)?;
        let mut p2: [*mut c_void; 5] = [
            &w.ptr as *const u64 as *mut c_void, &dy.ptr as *const u64 as *mut c_void,
            &dx.ptr as *const u64 as *mut c_void, &mm as *const i32 as *mut c_void, &kk as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.bw_x, k.div_ceil(128) as u32, 128, &mut p2)
    }

    pub fn mse_fwd(&self, y: &DeviceTensor, t: &DeviceTensor, out: &DeviceTensor, n: usize) -> std::result::Result<(), Error> {
        let nn = n as i32;
        let mut p: [*mut c_void; 4] = [
            &y.ptr as *const u64 as *mut c_void, &t.ptr as *const u64 as *mut c_void,
            &out.ptr as *const u64 as *mut c_void, &nn as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.mse, 1, 256, &mut p)
    }

    pub fn mse_bwd(&self, y: &DeviceTensor, t: &DeviceTensor, dy: &DeviceTensor, n: usize) -> std::result::Result<(), Error> {
        let nn = n as i32;
        let mut p: [*mut c_void; 4] = [
            &y.ptr as *const u64 as *mut c_void, &t.ptr as *const u64 as *mut c_void,
            &dy.ptr as *const u64 as *mut c_void, &nn as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.mse_bwd, n.div_ceil(256) as u32, 256, &mut p)
    }

    pub fn adamw(&self, p: &DeviceTensor, g: &DeviceTensor, m: &DeviceTensor, v: &DeviceTensor,
                 lr: f32, wd: f32, b1: f32, b2: f32, eps: f32, c1: f32, c2: f32) -> std::result::Result<(), Error> {
        let n = p.len as i32;
        let mut params: [*mut c_void; 12] = [
            &p.ptr as *const u64 as *mut c_void, &g.ptr as *const u64 as *mut c_void,
            &m.ptr as *const u64 as *mut c_void, &v.ptr as *const u64 as *mut c_void,
            &n as *const i32 as *mut c_void, &lr as *const f32 as *mut c_void,
            &wd as *const f32 as *mut c_void, &b1 as *const f32 as *mut c_void,
            &b2 as *const f32 as *mut c_void, &eps as *const f32 as *mut c_void,
            &c1 as *const f32 as *mut c_void, &c2 as *const f32 as *mut c_void,
        ];
        launch(&self.api, self.api.adam, (p.len as u32).div_ceil(256), 256, &mut params)
    }
    pub fn matmul_batch_fwd(&self, w: &DeviceTensor, x: &DeviceTensor, y: &DeviceTensor,
                            b: usize, m: usize, k: usize) -> std::result::Result<(), Error> {
        let (bb, mm, kk) = (b as i32, m as i32, k as i32);
        let mut p: [*mut c_void; 6] = [
            &w.ptr as *const u64 as *mut c_void, &x.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void, &bb as *const i32 as *mut c_void,
            &mm as *const i32 as *mut c_void, &kk as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.mm_fwd, (b * m).div_ceil(128) as u32, 128, &mut p)
    }

    pub fn matmul_batch_bwd(&self, w: &DeviceTensor, x: &DeviceTensor, dy: &DeviceTensor,
                            dw: &DeviceTensor, dx: &DeviceTensor,
                            b: usize, m: usize, k: usize) -> std::result::Result<(), Error> {
        let (bb, mm, kk) = (b as i32, m as i32, k as i32);
        let mut p: [*mut c_void; 6] = [
            &x.ptr as *const u64 as *mut c_void, &dy.ptr as *const u64 as *mut c_void,
            &dw.ptr as *const u64 as *mut c_void, &bb as *const i32 as *mut c_void,
            &mm as *const i32 as *mut c_void, &kk as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.mm_bw_w, m as u32, 128, &mut p)?;
        let mut p2: [*mut c_void; 6] = [
            &w.ptr as *const u64 as *mut c_void, &dy.ptr as *const u64 as *mut c_void,
            &dx.ptr as *const u64 as *mut c_void, &bb as *const i32 as *mut c_void,
            &mm as *const i32 as *mut c_void, &kk as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.mm_bw_x, b as u32, 128, &mut p2)
    }

    pub fn bias_add(&self, x: &DeviceTensor, bias: &DeviceTensor, y: &DeviceTensor,
                    b: usize, d: usize) -> std::result::Result<(), Error> {
        let (bb, dd) = (b as i32, d as i32);
        let mut p: [*mut c_void; 5] = [
            &x.ptr as *const u64 as *mut c_void, &bias.ptr as *const u64 as *mut c_void,
            &y.ptr as *const u64 as *mut c_void, &bb as *const i32 as *mut c_void, &dd as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.bias_add, (b*d).div_ceil(256) as u32, 256, &mut p)
    }

    pub fn bias_backward(&self, dy: &DeviceTensor, db: &DeviceTensor, b: usize, d: usize) -> std::result::Result<(), Error> {
        let (bb, dd) = (b as i32, d as i32);
        let mut p: [*mut c_void; 4] = [
            &dy.ptr as *const u64 as *mut c_void, &db.ptr as *const u64 as *mut c_void,
            &bb as *const i32 as *mut c_void, &dd as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.bias_bwd, d.min(256) as u32, 128, &mut p)
    }

    pub fn silu(&self, x: &DeviceTensor, y: &DeviceTensor, backward: bool,
                dy: Option<&DeviceTensor>, dx: Option<&DeviceTensor>) -> std::result::Result<(), Error> {
        let n = x.len as i32;
        if !backward {
            let mut p: [*mut c_void; 3] = [
                &x.ptr as *const u64 as *mut c_void, &y.ptr as *const u64 as *mut c_void, &n as *const i32 as *mut c_void,
            ];
            launch(&self.api, self.api.silu_fwd, (x.len as u32).div_ceil(256), 256, &mut p)
        } else {
            let mut p: [*mut c_void; 4] = [
                &x.ptr as *const u64 as *mut c_void, &dy.unwrap().ptr as *const u64 as *mut c_void,
                &dx.unwrap().ptr as *const u64 as *mut c_void, &n as *const i32 as *mut c_void,
            ];
            launch(&self.api, self.api.silu_bwd, (x.len as u32).div_ceil(256), 256, &mut p)
        }
    }

    pub fn cross_entropy(&self, logits: &DeviceTensor, target: &DeviceTensor, loss: &DeviceTensor,
                         n: usize, c: usize) -> std::result::Result<(), Error> {
        let (nn, cc) = (n as i32, c as i32);
        let mut p: [*mut c_void; 5] = [
            &logits.ptr as *const u64 as *mut c_void, &target.ptr as *const u64 as *mut c_void,
            &loss.ptr as *const u64 as *mut c_void, &nn as *const i32 as *mut c_void, &cc as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.ce_fwd, 1, 256, &mut p)
    }

    pub fn cross_entropy_bwd(&self, logits: &DeviceTensor, target: &DeviceTensor,
                             dlogits: &DeviceTensor, n: usize, c: usize) -> std::result::Result<(), Error> {
        let (nn, cc) = (n as i32, c as i32);
        let mut p: [*mut c_void; 5] = [
            &logits.ptr as *const u64 as *mut c_void, &target.ptr as *const u64 as *mut c_void,
            &dlogits.ptr as *const u64 as *mut c_void, &nn as *const i32 as *mut c_void, &cc as *const i32 as *mut c_void,
        ];
        launch(&self.api, self.api.ce_bwd, n as u32, 128, &mut p)
    }
}

