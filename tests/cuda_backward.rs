//! GPU training smoke test. Run with `cargo test -- --ignored` on a CUDA host.

#[test]
#[ignore]
fn cuda_matmul_backward_trains() {
    use cortex_train::cuda::{Cuda, DeviceTensor};
    let dev = Cuda::new().expect("CUDA device");
    let (k, m) = (64usize, 32usize);
    let w: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.001).collect();
    let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.1).collect();
    let t: Vec<f32> = (0..m).map(|r| (0..k).map(|c| w[r*k+c] * x[c]).sum::<f32>()).collect();
    let (dw, dx, dt) = (
        DeviceTensor::from_slice(&dev, &w).unwrap(),
        DeviceTensor::from_slice(&dev, &x).unwrap(),
        DeviceTensor::from_slice(&dev, &t).unwrap(),
    );
    let (y, loss, dy, dwi, dxi, mm, vv) = (
        DeviceTensor::zeros(&dev, m).unwrap(), DeviceTensor::zeros(&dev, 1).unwrap(),
        DeviceTensor::zeros(&dev, m).unwrap(), DeviceTensor::zeros(&dev, m*k).unwrap(),
        DeviceTensor::zeros(&dev, k).unwrap(), DeviceTensor::zeros(&dev, m*k).unwrap(),
        DeviceTensor::zeros(&dev, m*k).unwrap(),
    );
    for step in 1..=50 {
        dev.matmul_fwd(&dw, &dx, &y, m, k).unwrap();
        dev.mse_fwd(&y, &dt, &loss, m).unwrap();
        dev.mse_bwd(&y, &dt, &dy, m).unwrap();
        dev.matmul_bwd(&dw, &dx, &dy, &dwi, &dxi, m, k).unwrap();
        dev.adamw(&dw, &dwi, &mm, &vv, 0.05, 0.0, 0.9, 0.999, 1e-8,
                  1.0-0.9f32.powi(step), 1.0-0.999f32.powi(step)).unwrap();
    }
    dev.sync();
    assert!(loss.to_vec().unwrap()[0] < 1e-3);
}
