//! One decoder transformer block trained end-to-end on GPU.
//!
//! Embedding -> RMSNorm -> causal attention -> residual -> RMSNorm -> SwiGLU
//! FFN -> residual -> output head -> cross-entropy. Every backward pass is a
//! compiled CUDA kernel; no CPU autograd is used.

use cortex_train::cuda::{Cuda, DeviceTensor};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let dev = Cuda::new()?;
    let (b, s, d, h, ff, vocab) = (8usize, 8usize, 32usize, 2usize, 64usize, 64usize);
    let dh = d / h;
    let n = b * s;
    let bh = b * h;

    let mut rng = 777u64;
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let mut rand = || ((next() >> 11) as f32 / ((1u64 << 53) as f32)) * 2.0 - 1.0;
    let mut weights = |rows: usize, cols: usize, scale: f32| -> Vec<f32> {
        (0..rows * cols).map(|_| rand() * scale).collect()
    };
    let devt = |v: &[f32]| DeviceTensor::from_slice(&dev, v);
    let zeros = |len: usize| DeviceTensor::zeros(&dev, len);

    // Data: predict the next byte token from a repeating pattern.
    let ids: Vec<i32> = (0..n).map(|i| ((i % 17) + 3) as i32).collect();
    let targets: Vec<i32> = (0..n).map(|i| ((i + 1) % 17 + 3) as i32).collect();
    let d_ids = DeviceTensor::from_i32(&dev, &ids)?;
    let d_targets = DeviceTensor::from_i32(&dev, &targets)?;

    // Parameters.
    let w_embed = devt(&weights(vocab, d, 0.05))?;
    let d_embed = zeros(vocab * d)?;
    let n1 = devt(&vec![1f32; d])?; let d_n1 = zeros(d)?;
    let n2 = devt(&vec![1f32; d])?; let d_n2 = zeros(d)?;
    let nf = devt(&vec![1f32; d])?; let d_nf = zeros(d)?;
    let wq = devt(&weights(d, d, 0.05))?; let d_wq = zeros(d * d)?;
    let wk = devt(&weights(d, d, 0.05))?; let d_wk = zeros(d * d)?;
    let wv = devt(&weights(d, d, 0.05))?; let d_wv = zeros(d * d)?;
    let wo = devt(&weights(d, d, 0.05))?; let d_wo = zeros(d * d)?;
    let wg = devt(&weights(ff, d, 0.05))?; let d_wg = zeros(ff * d)?;
    let wu = devt(&weights(ff, d, 0.05))?; let d_wu = zeros(ff * d)?;
    let wd = devt(&weights(d, ff, 0.05))?; let d_wd = zeros(ff * d)?;
    let wh = devt(&weights(vocab, d, 0.05))?; let d_wh = zeros(vocab * d)?;

    let params: Vec<(&DeviceTensor, &DeviceTensor, DeviceTensor, DeviceTensor)> = Vec::new();
    let _ = params;
    // Optimizer state for all matrices (flattened pairs).
    let mut states: Vec<(&DeviceTensor, &DeviceTensor, DeviceTensor, DeviceTensor)> = Vec::new();
    for (p, g) in [(&w_embed,&d_embed),(&n1,&d_n1),(&n2,&d_n2),(&nf,&d_nf),
                   (&wq,&d_wq),(&wk,&d_wk),(&wv,&d_wv),(&wo,&d_wo),
                   (&wg,&d_wg),(&wu,&d_wu),(&wd,&d_wd),(&wh,&d_wh)] {
        let len = p.len;
        states.push((p, g, zeros(len)?, zeros(len)?));
    }

    // Activations.
    let (x, x1, x2, xn1, xn2, xf) = (zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?);
    let (inv1, inv2, invf) = (zeros(n)?,zeros(n)?,zeros(n)?);
    let (q, k, v, qh, kh, vh, kt) = (zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(bh*s*dh)?,zeros(bh*s*dh)?,zeros(bh*s*dh)?,zeros(bh*dh*s)?);
    let (scores, probs, o, oh) = (zeros(bh*s*s)?,zeros(bh*s*s)?,zeros(bh*s*dh)?,zeros(n*d)?);
    let (gate, up, act, down, logits, loss) = (zeros(n*ff)?,zeros(n*ff)?,zeros(n*ff)?,zeros(n*d)?,zeros(n*vocab)?,zeros(1)?);
    // Gradients.
    let (dx, dxn1, dxn1k, dxn1v, dxn2, dxn2u, dx1, dx2, dxf, dq, dk, dv, dqh, dkh, dvh, dkt) = (
        zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,
        zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,zeros(n*d)?,
        zeros(bh*s*dh)?,zeros(bh*s*dh)?,zeros(bh*s*dh)?,zeros(bh*dh*s)?,
    );
    let (dscores, dprobs, do_h, do2, dgate, dup, dact, ddown, dlogits, _dinv) = (
        zeros(bh*s*s)?,zeros(bh*s*s)?,zeros(bh*s*dh)?,zeros(n*d)?,zeros(n*ff)?,zeros(n*ff)?,
        zeros(n*ff)?,zeros(n*d)?,zeros(n*vocab)?,zeros(n)?,
    );
    let scale = (dh as f32).sqrt().recip();
    let (lr, decay, b1, b2, eps) = (0.003f32, 0.0f32, 0.9f32, 0.999f32, 1e-8f32);

    for step in 1..=100 {
        // ---- forward ----
        dev.embedding(&w_embed, &d_ids, &x, n, d, None)?;
        dev.rms_norm(&x, &n1, &xn1, &inv1, n, d, 1e-5)?;
        dev.matmul_batch_fwd(&wq, &xn1, &q, n, d, d)?;
        dev.matmul_batch_fwd(&wk, &xn1, &k, n, d, d)?;
        dev.matmul_batch_fwd(&wv, &xn1, &v, n, d, d)?;
        dev.split_heads(&q, &qh, b, h, s, dh)?;
        dev.split_heads(&k, &kh, b, h, s, dh)?;
        dev.split_heads(&v, &vh, b, h, s, dh)?;
        dev.transpose3(&kh, &kt, bh, s, dh)?;
        dev.attn_scores(&qh, &kt, &scores, bh, s, dh, scale)?;
        dev.causal_softmax(&scores, &probs, bh*s, s)?;
        dev.matmul3_fwd(&probs, &vh, &o, bh, s, s, dh)?;
        dev.split_heads(&o, &oh, b, h, s, dh)?;
        dev.matmul_batch_fwd(&wo, &oh, &x1, n, d, d)?;
        dev.add(&x, &x1, &x1, n*d)?;

        dev.rms_norm(&x1, &n2, &xn2, &inv2, n, d, 1e-5)?;
        dev.matmul_batch_fwd(&wg, &xn2, &gate, n, ff, d)?;
        dev.matmul_batch_fwd(&wu, &xn2, &up, n, ff, d)?;
        dev.silu(&gate, &gate, false, None, None)?;
        dev.mul(&gate, &up, &act, n*ff, false, None, None, None)?;
        dev.matmul_batch_fwd(&wd, &act, &down, n, d, ff)?;
        dev.add(&x1, &down, &x2, n*d)?;

        dev.rms_norm(&x2, &nf, &xf, &invf, n, d, 1e-5)?;
        dev.matmul_batch_fwd(&wh, &xf, &logits, n, vocab, d)?;
        dev.cross_entropy(&logits, &d_targets, &loss, n, vocab)?;

        // ---- backward ----
        dev.cross_entropy_bwd(&logits, &d_targets, &dlogits, n, vocab)?;
        dev.matmul_batch_bwd(&wh, &xf, &dlogits, &d_wh, &dxf, n, vocab, d)?;
        dev.rms_norm_bwd(&x2, &nf, &invf, &dxf, &dx2, &d_nf, n, d)?;
        // Residual: the same `dxf` also flows directly to x1 through x2 = x1 + down.
        dev.matmul_batch_bwd(&wd, &act, &dxf, &d_wd, &ddown, n, d, ff)?;
        dev.mul(&gate, &up, &dact, n*ff, true, Some(&ddown), Some(&dgate), Some(&dup))?;
        dev.silu(&gate, &dgate, true, Some(&dgate), Some(&dgate))?;
        dev.matmul_batch_bwd(&wg, &xn2, &dgate, &d_wg, &dxn2, n, ff, d)?;
        dev.matmul_batch_bwd(&wu, &xn2, &dup, &d_wu, &dxn2u, n, ff, d)?;
        dev.add_into(&dxn2, &dxn2u, n*d)?;
        dev.add_into(&dx2, &dxf, n*d)?;
        dev.rms_norm_bwd(&x1, &n2, &inv2, &dx2, &dx1, &d_n2, n, d)?;
        dev.add_into(&dx1, &dx2, n*d)?;

        dev.matmul_batch_bwd(&wo, &oh, &dx1, &d_wo, &do2, n, d, d)?;
        dev.merge_heads_rev(&do2, &do_h, b, h, s, dh)?;
        dev.matmul3_bwd(&probs, &vh, &do_h, &dprobs, &dvh, bh, s, s, dh)?;
        dev.softmax_rows(&probs, &dprobs, bh*s, s, true, Some(&dprobs), Some(&dscores))?;
        dev.matmul3_bwd(&qh, &kt, &dscores, &dqh, &dkt, bh, s, dh, s)?;
        dev.transpose3(&dkt, &dkh, bh, dh, s)?;
        dev.split_heads(&dqh, &dq, b, h, s, dh)?;
        dev.split_heads(&dkh, &dk, b, h, s, dh)?;
        dev.split_heads(&dvh, &dv, b, h, s, dh)?;
        dev.matmul_batch_bwd(&wq, &xn1, &dq, &d_wq, &dxn1, n, d, d)?;
        dev.matmul_batch_bwd(&wk, &xn1, &dk, &d_wk, &dxn1k, n, d, d)?;
        dev.matmul_batch_bwd(&wv, &xn1, &dv, &d_wv, &dxn1v, n, d, d)?;
        dev.add_into(&dxn1, &dxn1k, n*d)?;
        dev.add_into(&dxn1, &dxn1v, n*d)?;
        dev.rms_norm_bwd(&x, &n1, &inv1, &dxn1, &dx, &d_n1, n, d)?;
        dev.embedding(&x, &d_ids, &dx, n, d, Some(&d_embed))?;

        let c1 = 1.0 - b1.powi(step);
        let c2 = 1.0 - b2.powi(step);
        for (p, g, m, v) in states.iter_mut() {
            dev.adamw(p, g, m, v, lr, decay, b1, b2, eps, c1, c2)?;
        }
        if step % 10 == 0 {
            dev.sync();
            println!("step {step:>3} loss {:.4}", loss.to_vec()?[0]);
        }
    }
    dev.sync();
    println!("final loss {:.4}", loss.to_vec()?[0]);
    Ok(())
}
