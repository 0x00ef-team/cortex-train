//! Minimal gradient reducer for multi-process data-parallel training.
//!
//! No external dependencies. One process acts as the coordinator (server),
//! the others connect as workers. Before each optimizer step every worker
//! sends its gradient vector and receives the average.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

pub trait GradientReducer {
    fn reduce(&mut self, grad: &mut [f32]);
}

pub struct Local;
impl GradientReducer for Local {
    fn reduce(&mut self, _grad: &mut [f32]) {}
}

pub struct Server {
    workers: usize,
}

impl Server {
    pub fn bind(addr: &str, workers: usize) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let mut conns = Vec::with_capacity(workers);
        while conns.len() < workers {
            let (mut stream, _) = listener.accept()?;
            let mut len = [0u8; 8];
            stream.read_exact(&mut len)?;
            let n = u64::from_le_bytes(len) as usize;
            conns.push((stream, n));
        }
        // Exchange lengths, then average continuously.
        std::thread::spawn(move || loop {
            let mut acc: Option<Vec<f32>> = None;
            for (stream, n) in conns.iter_mut() {
                let mut buf = vec![0u8; (*n) * 4];
                if stream.read_exact(&mut buf).is_err() { return; }
                let g: Vec<f32> = buf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                match &mut acc {
                    None => acc = Some(g),
                    Some(a) => for (x, y) in a.iter_mut().zip(g) { *x += y; },
                }
            }
            let Some(mut avg) = acc else { return; };
            for x in &mut avg { *x /= workers as f32; }
            let mut out = Vec::with_capacity(avg.len() * 4);
            for x in avg { out.extend_from_slice(&x.to_le_bytes()); }
            for (stream, _) in conns.iter_mut() {
                if stream.write_all(&out).is_err() { return; }
            }
        });
        Ok(Self { workers })
    }

    pub fn workers(&self) -> usize { self.workers }
}

pub struct Worker {
    stream: TcpStream,
}

impl Worker {
    pub fn connect(addr: &str, n: usize) -> std::io::Result<Self> {
        let mut stream = TcpStream::connect(addr)?;
        stream.write_all(&(n as u64).to_le_bytes())?;
        Ok(Self { stream })
    }
}

impl GradientReducer for Worker {
    fn reduce(&mut self, grad: &mut [f32]) {
        let mut buf = Vec::with_capacity(grad.len() * 4);
        for &g in grad.iter() { buf.extend_from_slice(&g.to_le_bytes()); }
        self.stream.write_all(&buf).expect("send gradients");
        let mut resp = vec![0u8; grad.len() * 4];
        self.stream.read_exact(&mut resp).expect("receive averaged gradients");
        for (g, c) in grad.iter_mut().zip(resp.chunks_exact(4)) {
            *g = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    }
}
