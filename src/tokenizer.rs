//! Small dependency-free byte-BPE tokenizer.
//!
//! Trained on the code corpus, it replaces raw byte tokens with frequent
//! multi-byte subwords. This lowers the entropy of the next-token task, so a
//! tiny transformer can learn structure instead of guessing individual bytes.

use std::collections::HashMap;

pub struct Bpe {
    pub tokens: Vec<Vec<u8>>,
    ranks: HashMap<(u16, u16), u16>,
    root: TrieNode,
    max_len: usize,
}

#[derive(Default)]
struct TrieNode {
    next: HashMap<u8, TrieNode>,
    token: Option<u16>,
}

impl Bpe {
    /// Train on the first `sample_bytes` bytes of `corpus`, with `merges`
    /// BPE merge operations. Vocabulary = 256 + merges.
    pub fn train(corpus: &[u8], sample_bytes: usize, merges: usize) -> Self {
        let n = sample_bytes.min(corpus.len());
        let mut tokens: Vec<Vec<u8>> = (0..=255u16).map(|b| vec![b as u8]).collect();
        let mut ranks = HashMap::new();
        let mut ids: Vec<u16> = corpus[..n].iter().map(|&b| b as u16).collect();

        for _ in 0..merges {
            let mut counts: HashMap<(u16, u16), usize> = HashMap::with_capacity(4096);
            for w in ids.windows(2) {
                *counts.entry((w[0], w[1])).or_insert(0) += 1;
            }
            let best = counts.into_iter()
                .filter(|(_, c)| *c >= 2)
                .max_by_key(|(p, c)| (*c, std::cmp::Reverse(*p)))
                .map(|(p, _)| p);
            let Some(pair) = best else { break };
            let new_id = tokens.len() as u16;
            let mut merged = tokens[pair.0 as usize].clone();
            merged.extend_from_slice(&tokens[pair.1 as usize]);
            tokens.push(merged);
            ranks.insert(pair, new_id);

            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
                    out.push(new_id);
                    i += 2;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            ids = out;
        }

        let mut root = TrieNode::default();
        let mut max_len = 0;
        for (id, tok) in tokens.iter().enumerate() {
            let mut node = &mut root;
            for &b in tok { node = node.next.entry(b).or_default(); }
            node.token = Some(id as u16);
            max_len = max_len.max(tok.len());
        }
        Self { tokens, ranks, root, max_len }
    }

    pub fn vocab(&self) -> usize { self.tokens.len() }

    /// Greedy longest-match encoding over the trie.
    pub fn encode(&self, data: &[u8]) -> Vec<u16> {
        let mut out = Vec::with_capacity(data.len() / 2);
        let mut i = 0;
        while i < data.len() {
            let mut node = &self.root;
            let mut best_id = i as u16; // single-byte token
            let mut best_len = 1usize;
            let end = (i + self.max_len).min(data.len());
            for j in i..end {
                match node.next.get(&data[j]) {
                    Some(next) => {
                        node = next;
                        if let Some(id) = node.token {
                            best_id = id;
                            best_len = j - i + 1;
                        }
                    }
                    None => break,
                }
            }
            out.push(best_id);
            i += best_len;
        }
        out
    }

    pub fn decode(&self, ids: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(t) = self.tokens.get(id as usize) { out.extend_from_slice(t); }
        }
        out
    }

    pub fn merge_rank(&self, pair: (u16, u16)) -> Option<u16> { self.ranks.get(&pair).copied() }
}
