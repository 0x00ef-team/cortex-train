//! Software thermal guard for GPU training.
//!
//! No sudo / nvidia-smi clock lock required: we poll the GPU temperature and
//! stall the training loop when it exceeds a soft limit. Duty cycling keeps
//! the card below the target without changing system-wide clock settings.

use std::process::Command;
use std::thread;
use std::time::Duration;

/// Read the first GPU temperature in Celsius via `nvidia-smi`.
pub fn gpu_temp_c() -> Option<u32> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() { return None; }
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse().ok()
}

/// Sleep until the GPU cools back to `max_c` (checked every 500 ms).
pub fn cool_down(max_c: u32) {
    while let Some(t) = gpu_temp_c() {
        if t <= max_c { break; }
        thread::sleep(Duration::from_millis(500));
    }
}

/// Call from a training loop; only polls every `every` steps to keep overhead low.
pub fn throttle_if_hot(max_c: u32, step: usize, every: usize) {
    if step == 0 || step % every != 0 { return; }
    if let Some(t) = gpu_temp_c() {
        if t > max_c {
            eprintln!("[thermal] GPU {t}C > {max_c}C, pausing training");
            cool_down(max_c);
        }
    }
}
