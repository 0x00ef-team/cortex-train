//! Two-layer MLP trained entirely on GPU with NVRTC kernels.
//!
//! Linear -> SiLU -> Linear -> cross-entropy, with explicit backward kernels
//! and AdamW for every parameter tensor.

use cortex_train::cuda::{Cuda, DeviceTensor};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dev = Cuda::new()?;
    let (b, k, hidden, classes) = (64usize, 32usize, 64usize, 4usize);

    let mut rng = 4242u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let mut rand = || ((next() >> 11) as f32 / ((1u64 << 53) as f32)) * 2.0 - 1.0;

    let mut x = vec![0f32; b * k];
    let mut target = vec![0i32; b];
    for n in 0..b {
        let cls = n % classes;
        for i in 0..k { x[n*k + i] = rand() + (cls as f32) * 0.3; }
        target[n] = cls as i32;
    }
    let w1: Vec<f32> = (0..hidden*k).map(|_| rand() * 0.05).collect();
    let w2: Vec<f32> = (0..classes*hidden).map(|_| rand() * 0.05).collect();

    let (d_x, d_t) = (DeviceTensor::from_slice(&dev, &x)?, DeviceTensor::from_i32(&dev, &target)?);
    let mk = |v: &[f32]| DeviceTensor::from_slice(&dev, v);
    let (w1, b1, w2, b2) = (mk(&w1)?, mk(&vec![0f32; hidden])?, mk(&w2)?, mk(&vec![0f32; classes])?);
    let zeros = |n: usize| DeviceTensor::zeros(&dev, n);
    let (z1, a1, h, logits, loss) = (zeros(b*hidden)?, zeros(b*hidden)?, zeros(b*hidden)?, zeros(b*classes)?, zeros(1)?);
    let (dlogits, db2, dh, dW2, da1, db1, dW1, dx1) = (
        zeros(b*classes)?, zeros(classes)?, zeros(b*hidden)?, zeros(classes*hidden)?,
        zeros(b*hidden)?, zeros(hidden)?, zeros(hidden*k)?, zeros(b*k)?,
    );
    let (m1, v1) = (zeros(hidden*k)?, zeros(hidden*k)?);
    let (mb1, vb1) = (zeros(hidden)?, zeros(hidden)?);
    let (m2, v2) = (zeros(classes*hidden)?, zeros(classes*hidden)?);
    let (mb2, vb2) = (zeros(classes)?, zeros(classes)?);

    let (lr, wd, b1t, b2t, eps) = (0.003f32, 0.0f32, 0.9f32, 0.999f32, 1e-8f32);
    for step in 1..=300 {
        // forward
        dev.matmul_batch_fwd(&w1, &d_x, &z1, b, hidden, k)?;
        dev.bias_add(&z1, &b1, &a1, b, hidden)?;
        dev.silu(&a1, &h, false, None, None)?;
        dev.matmul_batch_fwd(&w2, &h, &logits, b, classes, hidden)?;
        dev.bias_add(&logits, &b2, &logits, b, classes)?;
        dev.cross_entropy(&logits, &d_t, &loss, b, classes)?;

        // backward
        dev.cross_entropy_bwd(&logits, &d_t, &dlogits, b, classes)?;
        dev.bias_backward(&dlogits, &db2, b, classes)?;
        dev.matmul_batch_bwd(&w2, &h, &dlogits, &dW2, &dh, b, classes, hidden)?;
        dev.silu(&a1, &dh, true, Some(&dh), Some(&da1))?;
        dev.bias_backward(&da1, &db1, b, hidden)?;
        dev.matmul_batch_bwd(&w1, &d_x, &da1, &dW1, &dx1, b, hidden, k)?;

        let c1 = 1.0 - b1t.powi(step);
        let c2 = 1.0 - b2t.powi(step);
        dev.adamw(&w1, &dW1, &m1, &v1, lr, wd, b1t, b2t, eps, c1, c2)?;
        dev.adamw(&b1, &db1, &mb1, &vb1, lr, wd, b1t, b2t, eps, c1, c2)?;
        dev.adamw(&w2, &dW2, &m2, &v2, lr, wd, b1t, b2t, eps, c1, c2)?;
        dev.adamw(&b2, &db2, &mb2, &vb2, lr, wd, b1t, b2t, eps, c1, c2)?;

        if step % 50 == 0 {
            dev.sync();
            println!("step {step:>3} loss {:.4}", loss.to_vec()?[0]);
        }
    }
    dev.sync();
    let final_loss = loss.to_vec()?[0];
    println!("final loss {final_loss:.4}");
    assert!(final_loss < 1.4, "MLP did not learn: {final_loss}");
    Ok(())
}
