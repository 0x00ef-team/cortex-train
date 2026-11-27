//! Multi-layer decoder-only transformer trained entirely on GPU.
//!
//! This is a reusable counterpart to the tape-based CPU `Transformer`:
//! forward/backward/step are explicit CUDA kernel sequences, so activations
//! stay in VRAM and no CPU autograd is involved.

use crate::cuda::{Cuda, DeviceTensor};
use std::rc::Rc;

pub struct GpuConfig {
    pub vocab: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub hidden: usize,
    pub seq: usize,
    pub batch: usize,
    pub seed: u64,
    pub lr: f32,
}

struct Layer {
    n1: DeviceTensor, dn1: DeviceTensor, inv1: DeviceTensor,
    wq: DeviceTensor, dwq: DeviceTensor,
    wk: DeviceTensor, dwk: DeviceTensor,
    wv: DeviceTensor, dwv: DeviceTensor,
    wo: DeviceTensor, dwo: DeviceTensor,
    n2: DeviceTensor, dn2: DeviceTensor, inv2: DeviceTensor,
    wg: DeviceTensor, dwg: DeviceTensor,
    wu: DeviceTensor, dwu: DeviceTensor,
    wd: DeviceTensor, dwd: DeviceTensor,
    // forward activations
    x: DeviceTensor, xn1: DeviceTensor, qh: DeviceTensor, kh: DeviceTensor,
    vh: DeviceTensor, kt: DeviceTensor, scores: DeviceTensor, probs: DeviceTensor,
    o: DeviceTensor, oh: DeviceTensor, a1: DeviceTensor, gate_raw: DeviceTensor, gate: DeviceTensor,
    up: DeviceTensor, act: DeviceTensor, xn2: DeviceTensor,
    // gradients
    dx: DeviceTensor, dxn1: DeviceTensor, dxn1k: DeviceTensor, dxn1v: DeviceTensor,
    dqh: DeviceTensor, dkh: DeviceTensor, dvh: DeviceTensor, dkt: DeviceTensor,
    dscores: DeviceTensor, dprobs: DeviceTensor, do_h: DeviceTensor, do2: DeviceTensor,
    dgate: DeviceTensor, dup: DeviceTensor, dact: DeviceTensor, ddown: DeviceTensor,
    dxn2: DeviceTensor, dxn2u: DeviceTensor, dx2: DeviceTensor, dx1: DeviceTensor,
    dq: DeviceTensor, dk: DeviceTensor, dv: DeviceTensor,
}

struct Adam {
    p: DeviceTensor, g: DeviceTensor, m: DeviceTensor, v: DeviceTensor,
}

pub struct GpuTransformer {
    dev: Rc<Cuda>,
    cfg: GpuConfig,
    n: usize,
    bh: usize,
    dh: usize,
    embed: DeviceTensor,
    dembed: DeviceTensor,
    layers: Vec<Layer>,
    nf: DeviceTensor,
    dnf: DeviceTensor,
    invf: DeviceTensor,
    xf: DeviceTensor,
    dxf: DeviceTensor,
    dxf_out: DeviceTensor,
    wh: DeviceTensor,
    dwh: DeviceTensor,
    logits: DeviceTensor,
    dlogits: DeviceTensor,
    loss: DeviceTensor,
    opt: Vec<Adam>,
    step: usize,
    max_temp: Option<u32>,
}

fn rand_init(rng: &mut u64, rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    (0..rows * cols).map(|_| {
        *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
        (((*rng >> 11) as f32 / ((1u64 << 53) as f32)) * 2.0 - 1.0) * scale
    }).collect()
}

impl GpuTransformer {
    pub fn new(dev: Cuda, cfg: GpuConfig) -> Result<Self, crate::cuda::Error> {
        let dev = Rc::new(dev);
        let n = cfg.batch * cfg.seq;
        let bh = cfg.batch * cfg.heads;
        let dh = cfg.dim / cfg.heads;
        assert_eq!(dh * cfg.heads, cfg.dim);
        let z = |len: usize| DeviceTensor::zeros(&dev, len);
        let w = |rows: usize, cols: usize, scale: f32, rng: &mut u64| -> Result<DeviceTensor, crate::cuda::Error> {
            DeviceTensor::from_slice(&dev, &rand_init(rng, rows, cols, scale))
        };
        let mut rng = cfg.seed | 1;
        let embed = w(cfg.vocab, cfg.dim, 0.05, &mut rng)?;
        let dembed = z(cfg.vocab * cfg.dim)?;
        let nf = DeviceTensor::from_slice(&dev, &vec![1f32; cfg.dim])?;
        let dnf = z(cfg.dim)?;
        let invf = z(n)?;
        let xf = z(n * cfg.dim)?;
        let dxf = z(n * cfg.dim)?;
        let dxf_out = z(n * cfg.dim)?;
        let wh = w(cfg.vocab, cfg.dim, 0.05, &mut rng)?;
        let dwh = z(cfg.vocab * cfg.dim)?;
        let logits = z(n * cfg.vocab)?;
        let dlogits = z(n * cfg.vocab)?;
        let loss = z(1)?;

        let mut layers = Vec::with_capacity(cfg.layers);
        for _ in 0..cfg.layers {
            let mut mk = |rr: usize, cc: usize| w(rr, cc, 0.05, &mut rng);
            let n1 = DeviceTensor::from_slice(&dev, &vec![1f32; cfg.dim])?;
            let n2 = DeviceTensor::from_slice(&dev, &vec![1f32; cfg.dim])?;
            layers.push(Layer {
                n1, dn1: z(cfg.dim)?, inv1: z(n)?,
                wq: mk(cfg.dim, cfg.dim)?, dwq: z(cfg.dim*cfg.dim)?,
                wk: mk(cfg.dim, cfg.dim)?, dwk: z(cfg.dim*cfg.dim)?,
                wv: mk(cfg.dim, cfg.dim)?, dwv: z(cfg.dim*cfg.dim)?,
                wo: mk(cfg.dim, cfg.dim)?, dwo: z(cfg.dim*cfg.dim)?,
                n2, dn2: z(cfg.dim)?, inv2: z(n)?,
                wg: mk(cfg.hidden, cfg.dim)?, dwg: z(cfg.hidden*cfg.dim)?,
                wu: mk(cfg.hidden, cfg.dim)?, dwu: z(cfg.hidden*cfg.dim)?,
                wd: mk(cfg.dim, cfg.hidden)?, dwd: z(cfg.dim*cfg.hidden)?,
                x: z(n*cfg.dim)?, xn1: z(n*cfg.dim)?,
                qh: z(bh*cfg.seq*dh)?, kh: z(bh*cfg.seq*dh)?, vh: z(bh*cfg.seq*dh)?,
                kt: z(bh*dh*cfg.seq)?, scores: z(bh*cfg.seq*cfg.seq)?, probs: z(bh*cfg.seq*cfg.seq)?,
                o: z(bh*cfg.seq*dh)?, oh: z(n*cfg.dim)?, a1: z(n*cfg.dim)?,
                gate_raw: z(n*cfg.hidden)?, gate: z(n*cfg.hidden)?, up: z(n*cfg.hidden)?, act: z(n*cfg.hidden)?, xn2: z(n*cfg.dim)?,
                dx: z(n*cfg.dim)?, dxn1: z(n*cfg.dim)?, dxn1k: z(n*cfg.dim)?, dxn1v: z(n*cfg.dim)?,
                dqh: z(bh*cfg.seq*dh)?, dkh: z(bh*cfg.seq*dh)?, dvh: z(bh*cfg.seq*dh)?, dkt: z(bh*dh*cfg.seq)?,
                dscores: z(bh*cfg.seq*cfg.seq)?, dprobs: z(bh*cfg.seq*cfg.seq)?,
                do_h: z(bh*cfg.seq*dh)?, do2: z(n*cfg.dim)?,
                dgate: z(n*cfg.hidden)?, dup: z(n*cfg.hidden)?, dact: z(n*cfg.hidden)?, ddown: z(n*cfg.dim)?,
                dxn2: z(n*cfg.dim)?, dxn2u: z(n*cfg.dim)?, dx2: z(n*cfg.dim)?, dx1: z(n*cfg.dim)?,
                dq: z(n*cfg.dim)?, dk: z(n*cfg.dim)?, dv: z(n*cfg.dim)?,
            });
        }

        let mut opt = Vec::new();
        let mut add = |p: &DeviceTensor, g: &DeviceTensor, opt: &mut Vec<Adam>| -> Result<(), crate::cuda::Error> {
            opt.push(Adam { p: p.share(), g: g.share(), m: z(p.len)?, v: z(p.len)? });
            Ok(())
        };
        add(&embed, &dembed, &mut opt)?;
        for l in &layers {
            for (p, g) in [(&l.n1,&l.dn1),(&l.wq,&l.dwq),(&l.wk,&l.dwk),(&l.wv,&l.dwv),
                           (&l.wo,&l.dwo),(&l.n2,&l.dn2),(&l.wg,&l.dwg),(&l.wu,&l.dwu),(&l.wd,&l.dwd)] {
                add(p, g, &mut opt)?;
            }
        }
        add(&nf, &dnf, &mut opt)?;
        add(&wh, &dwh, &mut opt)?;

        Ok(Self { dev, cfg, n, bh, dh, embed, dembed, layers, nf, dnf, invf, xf, dxf, dxf_out,
                  wh, dwh, logits, dlogits, loss, opt, step: 0, max_temp: None })
    }

    pub fn forward(&mut self, ids: &[i32], targets: &[i32]) -> Result<f32, crate::cuda::Error> {
        assert_eq!(ids.len(), self.n);
        assert_eq!(targets.len(), self.n);
        let d_ids = DeviceTensor::from_i32(&self.dev, ids)?;
        let d_t = DeviceTensor::from_i32(&self.dev, targets)?;
        let (d, s, h, ff, v, n, bh, dh) =
            (self.cfg.dim, self.cfg.seq, self.cfg.heads, self.cfg.hidden, self.cfg.vocab,
             self.n, self.bh, self.dh);
        let scale = (dh as f32).sqrt().recip();
        self.dev.embedding(&self.embed, &d_ids, &self.layers[0].x, n, d, None)?;

        for (li, l) in self.layers.iter().enumerate() {
            let prev = if li == 0 { &self.layers[0].x } else { &self.layers[li-1].x };
            self.dev.rms_norm(prev, &l.n1, &l.xn1, &l.inv1, n, d, 1e-5)?;
            self.dev.matmul_batch_fwd(&l.wq, &l.xn1, &l.dq, n, d, d)?;
            self.dev.matmul_batch_fwd(&l.wk, &l.xn1, &l.dk, n, d, d)?;
            self.dev.matmul_batch_fwd(&l.wv, &l.xn1, &l.dv, n, d, d)?;
            self.dev.split_heads(&l.dq, &l.qh, self.cfg.batch, h, s, dh)?;
            self.dev.split_heads(&l.dk, &l.kh, self.cfg.batch, h, s, dh)?;
            self.dev.split_heads(&l.dv, &l.vh, self.cfg.batch, h, s, dh)?;
            self.dev.transpose3(&l.kh, &l.kt, bh, s, dh)?;
            self.dev.attn_scores(&l.qh, &l.kt, &l.scores, bh, s, dh, scale)?;
            self.dev.causal_softmax(&l.scores, &l.probs, bh*s, s)?;
            self.dev.matmul3_fwd(&l.probs, &l.vh, &l.o, bh, s, s, dh)?;
            self.dev.split_heads(&l.o, &l.oh, self.cfg.batch, h, s, dh)?;
            self.dev.matmul_batch_fwd(&l.wo, &l.oh, &l.dx1, n, d, d)?;
            self.dev.add(prev, &l.dx1, &l.a1, n*d)?;

            self.dev.rms_norm(&l.a1, &l.n2, &l.xn2, &l.inv2, n, d, 1e-5)?;
            self.dev.matmul_batch_fwd(&l.wg, &l.xn2, &l.gate_raw, n, ff, d)?;
            self.dev.matmul_batch_fwd(&l.wu, &l.xn2, &l.up, n, ff, d)?;
            self.dev.silu(&l.gate_raw, &l.gate, false, None, None)?;
            self.dev.mul(&l.gate, &l.up, &l.act, n*ff, false, None, None, None)?;
            self.dev.matmul_batch_fwd(&l.wd, &l.act, &l.dx2, n, d, ff)?;
            self.dev.add(&l.a1, &l.dx2, &l.x, n*d)?;
        }
        let last = &self.layers[self.layers.len()-1].x;
        self.dev.rms_norm(last, &self.nf, &self.xf, &self.invf, n, d, 1e-5)?;
        self.dev.matmul_batch_fwd(&self.wh, &self.xf, &self.logits, n, v, d)?;
        self.dev.cross_entropy(&self.logits, &d_t, &self.loss, n, v)?;
        self.dev.sync();
        self.loss.to_vec().map(|v| v[0])
    }

    pub fn train_batch(&mut self, ids: &[i32], targets: &[i32]) -> Result<f32, crate::cuda::Error> {
        let loss = self.forward(ids, targets)?;
        let d_ids = DeviceTensor::from_i32(&self.dev, ids)?;
        let d_t = DeviceTensor::from_i32(&self.dev, targets)?;
        let (d, s, h, ff, v, n, bh, dh) =
            (self.cfg.dim, self.cfg.seq, self.cfg.heads, self.cfg.hidden, self.cfg.vocab,
             self.n, self.bh, self.dh);
        let scale = (dh as f32).sqrt().recip();
        self.dev.cross_entropy_bwd(&self.logits, &d_t, &self.dlogits, n, v)?;
        self.dev.matmul_batch_bwd(&self.wh, &self.xf, &self.dlogits, &self.dwh, &self.dxf, n, v, d)?;
        self.dev.rms_norm_bwd(&self.layers[self.layers.len()-1].x, &self.nf, &self.invf,
                              &self.dxf, &self.dxf_out, &self.dnf, n, d)?;
        self.dxf.copy_from(&self.dxf_out.values()?)?;

        for li in (0..self.layers.len()).rev() {
            let l = &self.layers[li];
            let prev = if li == 0 { &self.layers[0].x } else { &self.layers[li-1].x };
            // `dxf` currently carries grad wrt this layer's output `x`.
            self.dev.matmul_batch_bwd(&l.wd, &l.act, &self.dxf, &l.dwd, &l.ddown, n, d, ff)?;
            self.dev.mul(&l.gate, &l.up, &l.dact, n*ff, true, Some(&l.ddown), Some(&l.dgate), Some(&l.dup))?;
            self.dev.silu(&l.gate_raw, &l.dgate, true, Some(&l.dgate), Some(&l.dgate))?;
            self.dev.matmul_batch_bwd(&l.wg, &l.xn2, &l.dgate, &l.dwg, &l.dxn2, n, ff, d)?;
            self.dev.matmul_batch_bwd(&l.wu, &l.xn2, &l.dup, &l.dwu, &l.dxn2u, n, ff, d)?;
            self.dev.add_into(&l.dxn2, &l.dxn2u, n*d)?;
            self.dev.rms_norm_bwd(&l.a1, &l.n2, &l.inv2, &l.dxn2, &l.dx1, &l.dn2, n, d)?;
            self.dev.add_into(&l.dx1, &self.dxf, n*d)?;

            self.dev.matmul_batch_bwd(&l.wo, &l.oh, &l.dx1, &l.dwo, &l.do2, n, d, d)?;
            self.dev.merge_heads_rev(&l.do2, &l.do_h, self.cfg.batch, h, s, dh)?;
            self.dev.matmul3_bwd(&l.probs, &l.vh, &l.do_h, &l.dprobs, &l.dvh, bh, s, s, dh)?;
            self.dev.softmax_rows(&l.probs, &l.dprobs, bh*s, s, true, Some(&l.dprobs), Some(&l.dscores))?;
            self.dev.matmul3_bwd(&l.qh, &l.kt, &l.dscores, &l.dqh, &l.dkt, bh, s, dh, s)?;
            self.dev.transpose3(&l.dkt, &l.dkh, bh, dh, s)?;
            self.dev.split_heads(&l.dqh, &l.dq, self.cfg.batch, h, s, dh)?;
            self.dev.split_heads(&l.dkh, &l.dk, self.cfg.batch, h, s, dh)?;
            self.dev.split_heads(&l.dvh, &l.dv, self.cfg.batch, h, s, dh)?;
            self.dev.matmul_batch_bwd(&l.wq, &l.xn1, &l.dq, &l.dwq, &l.dxn1, n, d, d)?;
            self.dev.matmul_batch_bwd(&l.wk, &l.xn1, &l.dk, &l.dwk, &l.dxn1k, n, d, d)?;
            self.dev.matmul_batch_bwd(&l.wv, &l.xn1, &l.dv, &l.dwv, &l.dxn1v, n, d, d)?;
            self.dev.add_into(&l.dxn1, &l.dxn1k, n*d)?;
            self.dev.add_into(&l.dxn1, &l.dxn1v, n*d)?;
            self.dev.rms_norm_bwd(prev, &l.n1, &l.inv1, &l.dxn1, &l.dx, &l.dn1, n, d)?;
            // pass gradient to the previous layer via dxf
            self.dxf.copy_from(&l.dx.values()?)?;
        }
        self.dev.embedding(&self.embed, &d_ids, &self.dxf, n, d, Some(&self.dembed))?;

        self.step += 1;
        let c1 = 1.0 - 0.9f32.powi(self.step as i32);
        let c2 = 1.0 - 0.999f32.powi(self.step as i32);
        for a in &self.opt {
            self.dev.adamw(&a.p, &a.g, &a.m, &a.v, self.cfg.lr, 0.0, 0.9, 0.999, 1e-8, c1, c2)?;
        }
        if let Some(max_c) = self.max_temp {
            crate::thermal::throttle_if_hot(max_c, self.step, 10);
        }
        let _ = (scale, v);
        Ok(loss)
    }
    /// Copy all parameters to host with stable names and write a `CTRK1`
    /// checkpoint that the CPU autograd model can also load.
    pub fn set_lr(&mut self, lr: f32) { self.cfg.lr = lr; }

    /// Clear Adam moments (used when rolling back to a best checkpoint).
    pub fn reset_optimizer(&mut self) -> Result<(), crate::cuda::Error> {
        for a in &self.opt {
            a.m.copy_from(&vec![0f32; a.m.len])?;
            a.v.copy_from(&vec![0f32; a.v.len])?;
        }
        self.step = 0;
        Ok(())
    }

    /// Keep the GPU at or below this temperature by duty-cycling the loop.
    pub fn set_max_temp(&mut self, max_c: u32) { self.max_temp = Some(max_c); }

    pub fn save_checkpoint(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        use crate::checkpoint;
        use crate::Tensor;
        let mut named: Vec<(String, Tensor)> = Vec::new();
        let add = |name: String, t: &DeviceTensor, out: &mut Vec<(String, Tensor)>| -> std::io::Result<()> {
            let v = t.to_vec().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            out.push((name, Tensor::parameter(v, &[t.len])));
            Ok(())
        };
        add("embed.weight".into(), &self.embed, &mut named)?;
        for (i, l) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blocks.{i}.{s}");
            add(n("attn_norm"), &l.n1, &mut named)?;
            add(n("wq"), &l.wq, &mut named)?;
            add(n("wk"), &l.wk, &mut named)?;
            add(n("wv"), &l.wv, &mut named)?;
            add(n("wo"), &l.wo, &mut named)?;
            add(n("ffn_norm"), &l.n2, &mut named)?;
            add(n("gate"), &l.wg, &mut named)?;
            add(n("up"), &l.wu, &mut named)?;
            add(n("down"), &l.wd, &mut named)?;
        }
        add("final_norm".into(), &self.nf, &mut named)?;
        add("head.weight".into(), &self.wh, &mut named)?;
        checkpoint::save(path, &named)
    }


    /// Restore weights from a local `CTRK1` checkpoint, matching the names
    /// written by `save_checkpoint`. Optimizer state starts fresh.
    pub fn load_checkpoint(
        &mut self, path: impl AsRef<std::path::Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::checkpoint;
        let map = checkpoint::load(path)?;
        let copy = |name: &str, t: &DeviceTensor, map: &std::collections::HashMap<String, crate::Tensor>| -> Result<(), Box<dyn std::error::Error>> {
            let values = map.get(name).ok_or_else(|| format!("missing checkpoint tensor {name}"))?.values();
            if values.len() != t.len {
                return Err(format!("shape mismatch for {name}: {} vs {}", values.len(), t.len).into());
            }
            t.copy_from(&values)?;
            Ok(())
        };
        copy("embed.weight", &self.embed, &map)?;
        for (i, l) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blocks.{i}.{s}");
            copy(&n("attn_norm"), &l.n1, &map)?;
            copy(&n("wq"), &l.wq, &map)?;
            copy(&n("wk"), &l.wk, &map)?;
            copy(&n("wv"), &l.wv, &map)?;
            copy(&n("wo"), &l.wo, &map)?;
            copy(&n("ffn_norm"), &l.n2, &map)?;
            copy(&n("gate"), &l.wg, &map)?;
            copy(&n("up"), &l.wu, &map)?;
            copy(&n("down"), &l.wd, &map)?;
        }
        copy("final_norm", &self.nf, &map)?;
        copy("head.weight", &self.wh, &map)?;
        Ok(())
    }

    /// Greedy generation on device. Logits are copied to host only for argmax.
    pub fn generate(&mut self, prompt: &[i32], steps: usize) -> Result<Vec<i32>, crate::cuda::Error> {
        let mut ids = prompt.to_vec();
        let dummy_targets = vec![0i32; self.n];
        for _ in 0..steps {
            let mut window = ids.clone();
            window.resize(self.n, 0);
            self.forward(&window, &dummy_targets)?;
            let logits = self.logits.to_vec()?;
            let c = self.cfg.vocab;
            let last = &logits[(self.n - 1) * c..self.n * c];
            let mut best = 0usize;
            for i in 1..c { if last[i] > last[best] { best = i; } }
            ids.push(best as i32);
        }
        Ok(ids)
    }
}

