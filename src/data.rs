//! Dataset and batching helpers.

/// A dataset of fixed-length input/target pairs.
pub struct Dataset {
    inputs: Vec<Vec<usize>>,
    targets: Vec<Vec<usize>>,
}

impl Dataset {
    pub fn new(inputs: Vec<Vec<usize>>, targets: Vec<Vec<usize>>) -> Self {
        assert_eq!(inputs.len(), targets.len());
        Self { inputs, targets }
    }

    /// Build next-token prediction windows from a token sequence.
    pub fn windows(ids: &[usize], block: usize) -> Self {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for i in 0..ids.len().saturating_sub(block) {
            inputs.push(ids[i..i + block].to_vec());
            targets.push(ids[i + 1..i + block + 1].to_vec());
        }
        Self::new(inputs, targets)
    }

    pub fn len(&self) -> usize { self.inputs.len() }
    pub fn is_empty(&self) -> bool { self.inputs.is_empty() }

    /// Borrow one sample.
    pub fn get(&self, i: usize) -> (&[usize], &[usize]) {
        (&self.inputs[i], &self.targets[i])
    }
}

/// Deterministic mini-batch iterator with xorshift shuffling.
pub struct DataLoader {
    order: Vec<usize>,
    dataset: Dataset,
    batch: usize,
    rng: u64,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch: usize, seed: u64) -> Self {
        let order: Vec<usize> = (0..dataset.len()).collect();
        Self { order, dataset, batch: batch.max(1), rng: seed | 1 }
    }

    fn shuffle(&mut self) {
        for i in (1..self.order.len()).rev() {
            self.rng ^= self.rng << 13; self.rng ^= self.rng >> 7; self.rng ^= self.rng << 17;
            let j = (self.rng as usize) % (i + 1);
            self.order.swap(i, j);
        }
    }

    /// Yield shuffled batches for one epoch.
    pub fn epoch(&mut self) -> Vec<(Vec<usize>, Vec<Vec<usize>>)> {
        self.shuffle();
        let mut out = Vec::new();
        for chunk in self.order.chunks(self.batch) {
            let mut inputs = Vec::with_capacity(chunk.len());
            let mut targets = Vec::with_capacity(chunk.len());
            for &i in chunk {
                let (x, y) = self.dataset.get(i);
                inputs.push(x.to_vec());
                targets.push(y.to_vec());
            }
            out.push((chunk.to_vec(), targets));
            // `inputs` is unused until the model supports padded batches;
            // keep it available for callers that index samples themselves.
            let _ = inputs;
        }
        out
    }

    /// Indices of one shuffled epoch.
    pub fn epoch_indices(&mut self) -> Vec<usize> {
        self.shuffle();
        self.order.clone()
    }
}
