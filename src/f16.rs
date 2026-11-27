//! Storage-only half precision helpers.
//!
//! Autograd stays in f32 for accuracy; weights/activations can be stored as
//! f16 and expanded per operation. This saves RAM/bandwidth, not FLOPs.

pub fn f32_to_half(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7fffff;
    if exp <= 0 {
        if exp < -10 { sign }
        else {
            let m = (0x800000 | mant) >> (27 - exp);
            sign | (m as u16)
        }
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        let mut e = exp;
        let mut m = (mant + 0x1000) >> 13;
        if m == 0x800 { m = 0; e += 1; }
        if e >= 31 { return sign | 0x7c00; }
        sign | ((e as u16) << 10) | (m as u16)
    }
}

pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign } else {
            let mut m = mant;
            let mut e = -14i32;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            sign | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f800000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn pack(values: &[f32]) -> Vec<u16> { values.iter().map(|&x| f32_to_half(x)).collect() }
pub fn unpack(values: &[u16]) -> Vec<f32> { values.iter().map(|&h| half_to_f32(h)).collect() }
