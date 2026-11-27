//! GPU training with NVRTC-compiled backward kernels.
//!
//! Trains a linear layer `y = W x` on synthetic data using CUDA kernels for
//! forward, MSE loss, backward (dW/dx) and AdamW. No CPU autograd is involved.

use cortex_train::cuda::{Cuda, DeviceTensor};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dev = Cuda::new()?;
    let (k, m) = (256usize, 128usize);

    // Deterministic synthetic training pair.
    let mut x = vec![0.0f32; k];
    let mut w0 = vec![0.0f32; m * k];
    let mut target = vec![0.0f32; m];
    let mut rng = 12345u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    for i in 0..k { x[i] = ((next() >> 11) as f32 / ((1u64 << 53) as f32)) * 2.0 - 1.0; }
    for i in 0..m * k { w0[i] = (((next() >> 11) as f32 / ((1u64 << 53) as f32)) * 2.0 - 1.0) * 0.02; }
    for r in 0..m { for c in 0..k { target[r] += w0[r*k+c] * x[c]; } }

    let d_w = DeviceTensor::from_slice(&dev, &w0)?;
    let d_x = DeviceTensor::from_slice(&dev, &x)?;
    let d_t = DeviceTensor::from_slice(&dev, &target)?;
    let d_y = DeviceTensor::zeros(&dev, m)?;
    let d_loss = DeviceTensor::zeros(&dev, 1)?;
    let d_dy = DeviceTensor::zeros(&dev, m)?;
    let d_dw = DeviceTensor::zeros(&dev, m * k)?;
    let d_dx = DeviceTensor::zeros(&dev, k)?;
    let d_m = DeviceTensor::zeros(&dev, m * k)?;
    let d_v = DeviceTensor::zeros(&dev, m * k)?;

    let (lr, wd, b1, b2, eps) = (0.05f32, 0.0f32, 0.9f32, 0.999f32, 1e-8f32);
    for step in 1..=200 {
        dev.matmul_fwd(&d_w, &d_x, &d_y, m, k)?;
        dev.mse_fwd(&d_y, &d_t, &d_loss, m)?;
        dev.mse_bwd(&d_y, &d_t, &d_dy, m)?;
        dev.matmul_bwd(&d_w, &d_x, &d_dy, &d_dw, &d_dx, m, k)?;
        let c1 = 1.0 - b1.powi(step);
        let c2 = 1.0 - b2.powi(step);
        dev.adamw(&d_w, &d_dw, &d_m, &d_v, lr, wd, b1, b2, eps, c1, c2)?;
        if step % 40 == 0 {
            dev.sync();
            println!("step {step:>3} loss {:.6}", d_loss.to_vec()?[0]);
        }
    }
    dev.sync();
    println!("final loss {:.6}", d_loss.to_vec()?[0]);
    Ok(())
}
