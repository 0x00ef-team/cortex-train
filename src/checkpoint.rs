//! Named tensor checkpoint save/load (dependency-free binary format).
//!
//! Layout: magic `CTRK1`, u64 count, then for each tensor:
//! u64 name length, name bytes, u64 rank, rank*u64 shape, f32 data.

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 5] = b"CTRK1";

pub fn save(path: impl AsRef<Path>, params: &[(String, Tensor)]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&(params.len() as u64).to_le_bytes())?;
    for (name, t) in params {
        let shape = t.shape();
        w.write_all(&(name.len() as u64).to_le_bytes())?;
        w.write_all(name.as_bytes())?;
        w.write_all(&(shape.len() as u64).to_le_bytes())?;
        for d in &shape { w.write_all(&(*d as u64).to_le_bytes())?; }
        for v in t.data.borrow().v.iter() { w.write_all(&v.to_le_bytes())?; }
    }
    w.flush()
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

pub fn load(path: impl AsRef<Path>) -> std::io::Result<HashMap<String, Tensor>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 5];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a CTRK1 checkpoint")); }
    let count = read_u64(&mut r)?;
    let mut out = HashMap::with_capacity(count as usize);
    for _ in 0..count {
        let nlen = read_u64(&mut r)? as usize;
        let mut nb = vec![0u8; nlen];
        r.read_exact(&mut nb)?;
        let name = String::from_utf8(nb).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let rank = read_u64(&mut r)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank { shape.push(read_u64(&mut r)? as usize); }
        let n: usize = shape.iter().product();
        let mut values = vec![0f32; n];
        for v in &mut values {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            *v = f32::from_le_bytes(b);
        }
        out.insert(name, Tensor::parameter(values, &shape));
    }
    Ok(out)
}
