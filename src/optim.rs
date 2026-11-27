//! AdamW optimizer with decoupled weight decay.

use crate::tensor::{zero_grad_all, Tensor};

pub struct AdamW {
    params: Vec<Tensor>,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: usize,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let (m, v) = params.iter().map(|p| {
            let n = p.data.borrow().len();
            (vec![0.0; n], vec![0.0; n])
        }).unzip();
        Self { params, m, v, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01, step: 0 }
    }

    pub fn with_weight_decay(mut self, wd: f32) -> Self { self.weight_decay = wd; self }
    pub fn zero_grad(&self) { zero_grad_all(&self.params); }

    pub fn step(&mut self) {
        self.step += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let c1 = 1.0 - b1.powi(self.step as i32);
        let c2 = 1.0 - b2.powi(self.step as i32);
        for i in 0..self.params.len() {
            let g = self.params[i].grad();
            let mut p = self.params[i].data.borrow_mut();
            for j in 0..p.v.len() {
                let grad = g[j];
                self.m[i][j] = b1 * self.m[i][j] + (1.0 - b1) * grad;
                self.v[i][j] = b2 * self.v[i][j] + (1.0 - b2) * grad * grad;
                let mh = self.m[i][j] / c1;
                let vh = self.v[i][j] / c2;
                p.v[j] -= self.lr * mh / (vh.sqrt() + self.eps) + self.lr * self.weight_decay * p.v[j];
                p.grad[j] = 0.0;
            }
        }
    }
}
