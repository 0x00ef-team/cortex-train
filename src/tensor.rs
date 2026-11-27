//! Minimal reverse-mode automatic differentiation on dense f32 tensors.
//!
//! No dependencies. Tensors are reference-counted; each differentiable op
//! pushes a tape entry holding its inputs and a backward closure. Calling
//! `backward()` on a scalar walks the tape in reverse and accumulates grads.

use std::cell::RefCell;
use std::rc::Rc;

pub type Storage = Rc<RefCell<Data>>;

#[derive(Debug)]
pub struct Data {
    pub v: Vec<f32>,
    pub shape: Vec<usize>,
    pub grad: Vec<f32>,
}

impl Data {
    fn new(v: Vec<f32>, shape: Vec<usize>) -> Self {
        let n = v.len();
        Self { v, shape, grad: vec![0.0; n] }
    }
    pub fn len(&self) -> usize { self.v.len() }
    pub fn ndim(&self) -> usize { self.shape.len() }
    pub fn dim(&self, i: usize) -> usize { self.shape[i] }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Storage,
    pub id: usize,
    pub requires_grad: bool,
}

thread_local! {
    static NEXT_ID: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
    static TAPE: RefCell<Vec<Edge>> = const { RefCell::new(Vec::new()) };
}

struct Edge {
    output: Tensor,
    inputs: Vec<Tensor>,
    backward: Box<dyn Fn(&[f32], &mut [Vec<f32>])>,
}

fn next_id() -> usize {
    NEXT_ID.with(|c| {
        let id = c.get();
        c.set(id + 1);
        id
    })
}

impl Tensor {
    pub fn new(v: Vec<f32>, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(v.len(), n, "data length does not match shape");
        Self {
            data: Rc::new(RefCell::new(Data::new(v, shape.to_vec()))),
            id: next_id(),
            requires_grad: false,
        }
    }

    pub fn scalar(v: f32) -> Self { Self::new(vec![v], &[]) }
    pub fn zeros(shape: &[usize]) -> Self { Self::new(vec![0.0; shape.iter().product()], shape) }

    /// Deterministic pseudo-random tensor (xorshift), scaled by `scale`.
    pub fn randn(shape: &[usize], scale: f32, seed: u64) -> Self {
        let n: usize = shape.iter().product();
        let mut s = seed | 1;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 11) as f32) / ((1u64 << 53) as f32);
            v.push((u * 2.0 - 1.0) * scale);
        }
        Self::new(v, shape)
    }

    pub fn parameter(v: Vec<f32>, shape: &[usize]) -> Self {
        let mut t = Self::new(v, shape);
        t.requires_grad = true;
        t
    }

    pub fn shape(&self) -> Vec<usize> { self.data.borrow().shape.clone() }
    pub fn values(&self) -> Vec<f32> { self.data.borrow().v.clone() }
    pub fn grad(&self) -> Vec<f32> { self.data.borrow().grad.clone() }
    pub fn zero_grad(&self) {
        let mut d = self.data.borrow_mut();
        for g in &mut d.grad { *g = 0.0; }
    }

    pub fn view(&self, f: impl FnOnce(&Data)) { f(&*self.data.borrow()) }

    fn record(&self, inputs: Vec<Tensor>, backward: impl Fn(&[f32], &mut [Vec<f32>]) + 'static) {
        if self.requires_grad {
            TAPE.with(|t| t.borrow_mut().push(Edge {
                output: self.clone(),
                inputs,
                backward: Box::new(backward),
            }));
        }
    }
}

pub fn zero_grad_all(params: &[Tensor]) {
    for p in params { p.zero_grad(); }
    TAPE.with(|t| t.borrow_mut().clear());
}

pub fn reset_tape() { TAPE.with(|t| t.borrow_mut().clear()); }

impl Tensor {
    /// Reverse-mode backward from this tensor (usually a scalar loss).
    pub fn backward(&self) {
        // Seed the output gradient with ones.
        {
            let mut d = self.data.borrow_mut();
            for g in &mut d.grad { *g = 1.0; }
        }
        let edges = TAPE.with(|t| std::mem::take(&mut *t.borrow_mut()));
        for edge in edges.into_iter().rev() {
            let upstream = edge.output.grad();
            let mut grads: Vec<Vec<f32>> = edge.inputs.iter().map(|i| vec![0.0; i.data.borrow().len()]).collect();
            (edge.backward)(&upstream, &mut grads);
            for (input, g) in edge.inputs.iter().zip(grads) {
                if !input.requires_grad { continue; }
                let mut d = input.data.borrow_mut();
                for (dst, src) in d.grad.iter_mut().zip(g) { *dst += src; }
            }
        }
    }
}

// Helper trait so we can clone the tape out of a RefCell without exposing Edge.

fn needs_grad(a: &Tensor, b: &Tensor) -> bool { a.requires_grad || b.requires_grad }

fn flat(a: &Tensor) -> Storage { a.data.clone() }

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        let sa = self.shape();
        let sb = other.shape();
        let (out_shape, is_broadcast) = if sa == sb {
            (sa.clone(), false)
        } else if sb.len() == 1 && sb[0] == *sa.last().unwrap() {
            (sa.clone(), true)
        } else if sa.len() == 1 && sa[0] == *sb.last().unwrap() {
            (sb.clone(), true)
        } else {
            panic!("add: incompatible shapes {sa:?} and {sb:?}");
        };
        let n: usize = out_shape.iter().product();
        let a = flat(self);
        let b = flat(other);
        let mut v = vec![0.0f32; n];
        {
            let ad = a.borrow();
            let bd = b.borrow();
            let rows = if out_shape.len() >= 2 { out_shape[..out_shape.len()-1].iter().product::<usize>() } else { 1 };
            let cols = *out_shape.last().unwrap_or(&1);
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * cols + c;
                    let av = if sa == sb { ad.v[idx] } else if sa.len() == 1 { ad.v[c] } else { ad.v[idx] };
                    let bv = if sa == sb { bd.v[idx] } else if sb.len() == 1 { bd.v[c] } else { bd.v[idx] };
                    v[idx] = av + bv;
                }
            }
        }
        let mut out = Tensor::new(v, &out_shape);
        out.requires_grad = needs_grad(self, other);
        if out.requires_grad {
            let rev_broadcast = is_broadcast && sa.len() == 1;
            out.record(vec![self.clone(), other.clone()], move |dy, grads| {
                if rev_broadcast {
                    for d in dy { grads[0][0] += d; }
                    for g in grads[1].iter_mut() { *g += dy[0]; }
                } else if is_broadcast {
                    for (g, d) in grads[0].iter_mut().zip(dy) { *g += d; }
                    for d in dy { grads[1][0] += d; }
                } else {
                    for (g, d) in grads[0].iter_mut().zip(dy) { *g += d; }
                    for (g, d) in grads[1].iter_mut().zip(dy) { *g += d; }
                }
            });
        }
        out
    }

    pub fn add_scalar(&self, s: f32) -> Tensor {
        let v: Vec<f32> = self.data.borrow().v.iter().map(|x| x + s).collect();
        let mut out = Tensor::new(v, &self.shape());
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            out.record(vec![self.clone()], |dy, grads| {
                for (g, d) in grads[0].iter_mut().zip(dy) { *g += d; }
            });
        }
        out
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let v: Vec<f32> = self.data.borrow().v.iter().map(|x| x * s).collect();
        let mut out = Tensor::new(v, &self.shape());
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            out.record(vec![self.clone()], move |dy, grads| {
                for (g, d) in grads[0].iter_mut().zip(dy) { *g += d * s; }
            });
        }
        out
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let sa = self.shape();
        let sb = other.shape();
        assert_eq!(sa, sb, "mul requires equal shapes");
        let a = flat(self);
        let b = flat(other);
        let v: Vec<f32> = {
            let ad = a.borrow(); let bd = b.borrow();
            ad.v.iter().zip(&bd.v).map(|(x, y)| x * y).collect()
        };
        let mut out = Tensor::new(v, &sa);
        out.requires_grad = needs_grad(self, other);
        if out.requires_grad {
            out.record(vec![self.clone(), other.clone()], move |dy, grads| {
                let ad = a.borrow(); let bd = b.borrow();
                for (i, d) in dy.iter().enumerate() {
                    grads[0][i] += d * bd.v[i];
                    grads[1][i] += d * ad.v[i];
                }
            });
        }
        out
    }

    /// Matrix multiplication for `[m,k] x [k,n]` or batched
    /// `[b,m,k] x [b,k,n]` tensors.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let sa = self.shape();
        let sb = other.shape();
        assert_eq!(sa.len(), sb.len(), "matmul: mismatched ranks");
        assert!(sa.len() == 2 || sa.len() == 3, "matmul supports 2D and 3D");
        let (batch, m, k, n) = if sa.len() == 2 {
            assert_eq!(sb[0], sa[1], "matmul shape mismatch");
            (1usize, sa[0], sa[1], sb[1])
        } else {
            assert_eq!(sa[0], sb[0], "matmul batch mismatch");
            assert_eq!(sb[1], sa[2], "matmul shape mismatch");
            (sa[0], sa[1], sa[2], sb[2])
        };
        let a = flat(self);
        let b = flat(other);
        let mut v = vec![0.0f32; batch * m * n];
        {
            let ad = a.borrow(); let bd = b.borrow();
            for bi in 0..batch {
                let ao = bi * m * k;
                let bo = bi * k * n;
                let co = bi * m * n;
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for t in 0..k { acc += ad.v[ao + i*k + t] * bd.v[bo + t*n + j]; }
                        v[co + i*n + j] = acc;
                    }
                }
            }
        }
        let out_shape = if sa.len() == 2 { vec![m, n] } else { vec![batch, m, n] };
        let mut out = Tensor::new(v, &out_shape);
        out.requires_grad = needs_grad(self, other);
        if out.requires_grad {
            out.record(vec![self.clone(), other.clone()], move |dy, grads| {
                let ad = a.borrow(); let bd = b.borrow();
                for bi in 0..batch {
                    let ao = bi * m * k;
                    let bo = bi * k * n;
                    let co = bi * m * n;
                    for i in 0..m {
                        for j in 0..n {
                            let d = dy[co + i*n + j];
                            if d == 0.0 { continue; }
                            for t in 0..k {
                                grads[0][ao + i*k + t] += d * bd.v[bo + t*n + j];
                                grads[1][bo + t*n + j] += d * ad.v[ao + i*k + t];
                            }
                        }
                    }
                }
            });
        }
        out
    }

    pub fn transpose(&self) -> Tensor {
        let sh = self.shape();
        assert_eq!(sh.len(), 2, "transpose only supports 2D");
        let (r, c) = (sh[0], sh[1]);
        let a = flat(self);
        let mut v = vec![0.0f32; r * c];
        {
            let ad = a.borrow();
            for i in 0..r { for j in 0..c { v[j * r + i] = ad.v[i * c + j]; } }
        }
        let mut out = Tensor::new(v, &[c, r]);
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            out.record(vec![self.clone()], move |dy, grads| {
                for i in 0..r { for j in 0..c { grads[0][i * c + j] += dy[j * r + i]; } }
            });
        }
        out
    }

    /// Swap axes 1 and 2 of a `[a,b,c]` tensor.
    /// Swap axes 0 and 1 of a `[a,b,c]` tensor (`[a,b,c] -> [b,a,c]`).
    pub fn transpose01(&self) -> Tensor {
        let sh = self.shape();
        assert_eq!(sh.len(), 3, "transpose01 needs 3D");
        let (a, b, c) = (sh[0], sh[1], sh[2]);
        let src = flat(self);
        let mut v = vec![0.0f32; a * b * c];
        {
            let s = src.borrow();
            for i in 0..a { for j in 0..b { for k in 0..c {
                v[j * a * c + i * c + k] = s.v[i * b * c + j * c + k];
            }}}
        }
        let mut out = Tensor::new(v, &[b, a, c]);
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            out.record(vec![self.clone()], move |dy, grads| {
                for i in 0..a { for j in 0..b { for k in 0..c {
                    grads[0][i * b * c + j * c + k] += dy[j * a * c + i * c + k];
                }}}
            });
        }
        out
    }

    pub fn transpose12(&self) -> Tensor {
        let sh = self.shape();
        assert_eq!(sh.len(), 3, "transpose12 needs 3D");
        let (a, b, c) = (sh[0], sh[1], sh[2]);
        let src = flat(self);
        let mut v = vec![0.0f32; a * b * c];
        {
            let s = src.borrow();
            for i in 0..a { for j in 0..b { for k in 0..c {
                v[i * c * b + k * b + j] = s.v[i * b * c + j * c + k];
            }}}
        }
        let mut out = Tensor::new(v, &[a, c, b]);
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            out.record(vec![self.clone()], move |dy, grads| {
                for i in 0..a { for j in 0..b { for k in 0..c {
                    grads[0][i * b * c + j * c + k] += dy[i * c * b + k * b + j];
                }}}
            });
        }
        out
    }

    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), self.data.borrow().len());
        let mut out = Tensor::new(self.values(), shape);
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            out.record(vec![self.clone()], |dy, grads| {
                for (g, d) in grads[0].iter_mut().zip(dy) { *g += d; }
            });
        }
        out
    }

    pub fn sum(&self) -> Tensor {
        let total: f32 = self.data.borrow().v.iter().sum();
        let mut out = Tensor::scalar(total);
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            out.record(vec![self.clone()], |dy, grads| {
                for g in &mut grads[0] { *g += dy[0]; }
            });
        }
        out
    }

    pub fn softmax(&self) -> Tensor {
        let sh = self.shape();
        let rows = if sh.len() >= 2 { sh[..sh.len()-1].iter().product::<usize>() } else { 1 };
        let cols = *sh.last().unwrap_or(&1);
        let src = flat(self);
        let mut v = vec![0.0f32; rows * cols];
        {
            let s = src.borrow();
            for r in 0..rows {
                let mx = s.v[r * cols..(r + 1) * cols].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for c in 0..cols {
                    let e = (s.v[r * cols + c] - mx).exp();
                    v[r * cols + c] = e;
                    sum += e;
                }
                for c in 0..cols { v[r * cols + c] /= sum; }
            }
        }
        let mut out = Tensor::new(v, &sh);
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            let probs = out.data.clone();
            out.record(vec![self.clone()], move |dy, grads| {
                let p = probs.borrow();
                for r in 0..rows {
                    let dot: f32 = (0..cols).map(|c| dy[r*cols+c] * p.v[r*cols+c]).sum();
                    for c in 0..cols {
                        grads[0][r*cols+c] += p.v[r*cols+c] * (dy[r*cols+c] - dot);
                    }
                }
            });
        }
        out
    }

    pub fn silu(&self) -> Tensor {
        let v: Vec<f32> = self.data.borrow().v.iter().map(|x| x / (1.0 + (-x).exp())).collect();
        let mut out = Tensor::new(v, &self.shape());
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            let src = flat(self);
            out.record(vec![self.clone()], move |dy, grads| {
                let s = src.borrow();
                for (i, d) in dy.iter().enumerate() {
                    let sig = 1.0 / (1.0 + (-s.v[i]).exp());
                    grads[0][i] += d * sig * (1.0 + s.v[i] * (1.0 - sig));
                }
            });
        }
        out
    }

    pub fn sigmoid(&self) -> Tensor {
        let v: Vec<f32> = self.data.borrow().v.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let mut out = Tensor::new(v, &self.shape());
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            let y = out.data.clone();
            out.record(vec![self.clone()], move |dy, grads| {
                let yv = y.borrow();
                for (i, d) in dy.iter().enumerate() { grads[0][i] += d * yv.v[i] * (1.0 - yv.v[i]); }
            });
        }
        out
    }

    pub fn log(&self) -> Tensor {
        let v: Vec<f32> = self.data.borrow().v.iter().map(|x| x.ln()).collect();
        let mut out = Tensor::new(v, &self.shape());
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            let src = flat(self);
            out.record(vec![self.clone()], move |dy, grads| {
                let s = src.borrow();
                for (i, d) in dy.iter().enumerate() { grads[0][i] += d / s.v[i]; }
            });
        }
        out
    }

    pub fn tanh(&self) -> Tensor {
        let v: Vec<f32> = self.data.borrow().v.iter().map(|x| x.tanh()).collect();
        let mut out = Tensor::new(v, &self.shape());
        out.requires_grad = self.requires_grad;
        if out.requires_grad {
            let y = out.data.clone();
            out.record(vec![self.clone()], move |dy, grads| {
                let yv = y.borrow();
                for (i, d) in dy.iter().enumerate() { grads[0][i] += d * (1.0 - yv.v[i] * yv.v[i]); }
            });
        }
        out
    }

    /// Gather embedding rows: `table[V,D]`, ids `[N]` -> `[N,D]`.
    pub fn embedding(table: &Tensor, ids: &[usize]) -> Tensor {
        let sh = table.shape();
        assert_eq!(sh.len(), 2);
        let (v, d) = (sh[0], sh[1]);
        let tbl = flat(table);
        let mut out_v = vec![0.0f32; ids.len() * d];
        {
            let t = tbl.borrow();
            for (i, &id) in ids.iter().enumerate() {
                assert!(id < v, "embedding id out of range");
                out_v[i*d..(i+1)*d].copy_from_slice(&t.v[id*d..(id+1)*d]);
            }
        }
        let mut out = Tensor::new(out_v, &[ids.len(), d]);
        out.requires_grad = table.requires_grad;
        if out.requires_grad {
            let ids = ids.to_vec();
            out.record(vec![table.clone()], move |dy, _grads| {
                let mut t = tbl.borrow_mut();
                for (i, &id) in ids.iter().enumerate() {
                    for k in 0..d { t.grad[id*d+k] += dy[i*d+k]; }
                }
            });
        }
        out
    }

    /// RMS layer norm over the last dimension, with learned per-dim weight.
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Tensor {
        let sh = self.shape();
        let rows = if sh.len() >= 2 { sh[..sh.len()-1].iter().product::<usize>() } else { 1 };
        let d = *sh.last().unwrap_or(&1);
        let src = flat(self);
        let wgt = flat(weight);
        let mut v = vec![0.0f32; rows * d];
        let mut invs = vec![0.0f32; rows];
        {
            let s = src.borrow(); let w = wgt.borrow();
            for r in 0..rows {
                let ms: f32 = (0..d).map(|k| s.v[r*d+k] * s.v[r*d+k]).sum::<f32>() / d as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                invs[r] = inv;
                for k in 0..d { v[r*d+k] = s.v[r*d+k] * inv * w.v[k]; }
            }
        }
        let mut out = Tensor::new(v, &sh);
        out.requires_grad = needs_grad(self, weight);
        if out.requires_grad {
            out.record(vec![self.clone(), weight.clone()], move |dy, grads| {
                let s = src.borrow(); let w = wgt.borrow();
                for r in 0..rows {
                    let inv = invs[r];
                    let mut dot = 0.0;
                    for k in 0..d { dot += s.v[r*d+k] * w.v[k] * dy[r*d+k]; }
                    for k in 0..d {
                        grads[0][r*d+k] += (w.v[k] * inv) * dy[r*d+k]
                            - (inv * inv * inv / d as f32) * s.v[r*d+k] * dot;
                        grads[1][k] += dy[r*d+k] * s.v[r*d+k] * inv;
                    }
                }
            });
        }
        out
    }
}

/// Cross-entropy from logits `[N,C]` and class ids `[N]`. Returns mean loss.
pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> Tensor {
    let sh = logits.shape();
    assert_eq!(sh.len(), 2);
    let (n, c) = (sh[0], sh[1]);
    assert_eq!(targets.len(), n);
    let src = flat(logits);
    let mut loss = 0.0f32;
    let mut probs = vec![0.0f32; n * c];
    {
        let s = src.borrow();
        for i in 0..n {
            let mx = s.v[i*c..(i+1)*c].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for k in 0..c { let e = (s.v[i*c+k] - mx).exp(); probs[i*c+k] = e; sum += e; }
            for k in 0..c { probs[i*c+k] /= sum; }
            loss -= (probs[i*c+targets[i]] + 1e-12).ln();
        }
    }
    loss /= n as f32;
    let mut out = Tensor::scalar(loss);
    out.requires_grad = logits.requires_grad;
    if out.requires_grad {
        let targets = targets.to_vec();
        out.record(vec![logits.clone()], move |dy, grads| {
            let scale = dy[0] / n as f32;
            for i in 0..n {
                for k in 0..c { grads[0][i*c+k] += scale * probs[i*c+k]; }
                grads[0][i*c+targets[i]] -= scale;
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < tol, "expected {y}, got {x}");
        }
    }

    fn numeric_grad(f: impl Fn(&Tensor) -> Tensor, t: &Tensor, eps: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(t.data.borrow().len());
        for i in 0..t.data.borrow().len() {
            let mut plus = t.values(); plus[i] += eps;
            let mut minus = t.values(); minus[i] -= eps;
            let lp = f(&Tensor::parameter(plus, &t.shape())).values()[0];
            let lm = f(&Tensor::parameter(minus, &t.shape())).values()[0];
            out.push((lp - lm) / (2.0 * eps));
        }
        out
    }

    #[test]
    fn matmul_gradients() {
        reset_tape();
        let a = Tensor::parameter(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::parameter(vec![0.5, -1.0, 2.0, 1.5], &[2, 2]);
        let loss = a.matmul(&b).sum();
        loss.backward();
        close(&a.grad(), &numeric_grad(|t| t.matmul(&b).sum(), &a, 1e-3), 2e-2);
        close(&b.grad(), &numeric_grad(|t| a.matmul(t).sum(), &b, 1e-3), 2e-2);
    }

    #[test]
    fn cross_entropy_gradients() {
        reset_tape();
        let logits = Tensor::parameter(vec![0.1, -0.2, 0.7, 0.4, 0.3, -0.5], &[2, 3]);
        let loss = cross_entropy(&logits, &[2, 1]);
        loss.backward();
        let expected = numeric_grad(|t| cross_entropy(t, &[2, 1]), &logits, 1e-3);
        close(&logits.grad(), &expected, 2e-2);
        let p = logits.softmax().values();
        assert!((p[2] + p[4] - 1.0).abs() < 1e-3 || p[2] > 0.4);
    }

    #[test]
    fn rms_norm_gradients() {
        reset_tape();
        let x = Tensor::parameter(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]);
        let w = Tensor::parameter(vec![0.8, 1.2], &[2]);
        let loss = x.rms_norm(&w, 1e-5).sum();
        loss.backward();
        close(&x.grad(), &numeric_grad(|t| t.rms_norm(&w, 1e-5).sum(), &x, 1e-3), 3e-2);
        close(&w.grad(), &numeric_grad(|t| x.rms_norm(t, 1e-5).sum(), &w, 1e-3), 3e-2);
    }
}
