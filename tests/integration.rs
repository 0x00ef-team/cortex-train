use cortex_train::{binary_cross_entropy, checkpoint, mse, AdamW, DataLoader, Dataset, RMSProp, SGD, Tensor, Transformer};

#[test]
fn mse_and_bce_gradients() {
    cortex_train::tensor::reset_tape();
    let p = Tensor::parameter(vec![1.0, -1.0, 0.5], &[3]);
    let t = Tensor::new(vec![0.0, 0.0, 1.0], &[3]);
    mse(&p, &t).backward();
    let expected: Vec<f32> = p.values().iter().zip(t.values()).map(|(a, b)| 2.0 * (a - b) / 3.0).collect();
    for (g, e) in p.grad().iter().zip(&expected) { assert!((g - e).abs() < 1e-4); }

    cortex_train::tensor::reset_tape();
    let z = Tensor::parameter(vec![0.2, -0.7, 1.1], &[3]);
    let loss = binary_cross_entropy(&z, &[1.0, 0.0, 1.0]);
    loss.backward();
    assert_eq!(z.grad().len(), 3);
    assert!(z.grad().iter().all(|g| g.is_finite()));
}

#[test]
fn dataset_and_dataloader() {
    let ids: Vec<usize> = (0..20).collect();
    let ds = Dataset::windows(&ids, 4);
    assert!(ds.len() > 10);
    let (x, y) = ds.get(0);
    assert_eq!(x.len(), 4);
    assert_eq!(y[0], x[1]);
    let mut loader = DataLoader::new(ds, 4, 7);
    let ep1 = loader.epoch_indices();
    let ep2 = loader.epoch_indices();
    assert_eq!(ep1.len(), ep2.len());
    assert_ne!(ep1, ep2);
}

#[test]
fn optimizers_reduce_loss() {
    let target = Tensor::new(vec![0.0], &[1]);
    let x1 = Tensor::parameter(vec![2.0], &[1]);
    let mut a = AdamW::new(vec![x1.clone()], 0.1).with_weight_decay(0.0);
    for _ in 0..100 { a.zero_grad(); mse(&x1, &target).backward(); a.step(); }
    assert!(x1.values()[0].abs() < 0.1);

    let x2 = Tensor::parameter(vec![2.0], &[1]);
    let mut s = SGD::new(vec![x2.clone()], 0.1).with_momentum(0.9);
    for _ in 0..100 { s.zero_grad(); mse(&x2, &target).backward(); s.step(); }
    assert!(x2.values()[0].abs() < 0.1);

    let x3 = Tensor::parameter(vec![2.0], &[1]);
    let mut r = RMSProp::new(vec![x3.clone()], 0.1);
    for _ in 0..100 { r.zero_grad(); mse(&x3, &target).backward(); r.step(); }
    assert!(x3.values()[0].abs() < 0.1);
}

#[test]
fn checkpoint_roundtrip() {
    let model = Transformer::new(8, 8, 1, 2, 16, 3);
    let named = model.named_params();
    let dir = std::env::temp_dir().join(format!("ctrk-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.ctrk");
    checkpoint::save(&path, &named).unwrap();
    let loaded = checkpoint::load(&path).unwrap();
    assert_eq!(loaded.len(), named.len());
    for (name, t) in &named {
        assert_eq!(loaded[name].values(), t.values());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
