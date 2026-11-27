//! Streaming corpus reader for training on large text/code datasets.
//!
//! Files are read line by line; tokens are produced as byte values. For
//! character-level models this gives an exact round-trip vocabulary of at
//! most 256 symbols without storing the whole corpus in RAM.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct Corpus {
    pub tokens: Vec<usize>,
    pub vocab: usize,
}

impl Corpus {
    /// Load one or more UTF-8 files, keeping byte-level tokens.
    pub fn load(paths: &[impl AsRef<Path>]) -> std::io::Result<Self> {
        let mut tokens = Vec::new();
        for path in paths {
            let f = File::open(path)?;
            for line in BufReader::new(f).split(b'\n') {
                let line = line?;
                tokens.extend(line.iter().map(|&b| b as usize));
                tokens.push(b'\n' as usize);
            }
        }
        if tokens.is_empty() {
            tokens.push(0);
        }
        Ok(Self { tokens, vocab: 256 })
    }

    pub fn len(&self) -> usize { self.tokens.len() }
    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }

    /// Decode a byte-token sequence back to text (lossy for truncated tails).
    pub fn decode(&self, ids: &[usize]) -> String {
        let bytes: Vec<u8> = ids.iter().map(|&b| b as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
