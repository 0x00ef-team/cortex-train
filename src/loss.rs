//! Loss functions composed from differentiable tensor ops.

use crate::tensor::Tensor;

pub use crate::tensor::cross_entropy as ce;

/// Mean squared error between equal-shaped tensors.
pub fn mse(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.add(&target.scale(-1.0));
    diff.mul(&diff).sum().scale(1.0 / pred.data.borrow().len() as f32)
}

/// Binary cross-entropy from logits and 0/1 targets `[N]`.
///
/// Uses the stable composed form `-(t*log(sigmoid(z)) + (1-t)*log(1-sigmoid(z)))`.
pub fn binary_cross_entropy(logits: &Tensor, targets: &[f32]) -> Tensor {
    let n = targets.len();
    assert_eq!(logits.data.borrow().len(), n);
    let t = Tensor::new(targets.to_vec(), &[n]);
    let one = Tensor::new(vec![1.0; n], &[n]);
    let s = logits.sigmoid();
    let a = t.mul(&s.log());
    let b = one.add(&t.scale(-1.0)).mul(&one.add(&s.scale(-1.0)).log());
    a.add(&b).sum().scale(-1.0 / n as f32)
}
