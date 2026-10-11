//! WAV reader and resampler for getting clips into 16 kHz mono float PCM.

use std::f64::consts::PI;
use std::path::Path;

use crate::{err, Result};

/// PCM 8/16/24/32-bit or 32/64-bit float WAV, any channel count -> (mono samples, rate).
pub fn read_wav(b: &[u8]) -> Result<(Vec<f32>, u32)> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return err("not a WAV file");
    }
    let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    // (format, channels, rate, bits)
    let mut fmt: Option<(u16, usize, u32, usize)> = None;
    let mut p = 12;
    while p + 8 <= b.len() {
        let (id, len, body) = (&b[p..p + 4], u32_at(p + 4) as usize, p + 8);
        if id == b"fmt " && body + 16 <= b.len() {
            let mut format = u16_at(body);
            if format == 0xfffe && len >= 26 && body + 26 <= b.len() {
                format = u16_at(body + 24);
            }
            fmt = Some((format, u16_at(body + 2) as usize, u32_at(body + 4), u16_at(body + 14) as usize));
        } else if id == b"data" {
            let Some((format, ch, rate, bits)) = fmt else {
                return err("WAV data before fmt");
            };
            let bytes = bits >> 3;
            let float = format == 3;
            if ch == 0 || !matches!((float, bytes), (false, 1..=4) | (true, 4) | (true, 8)) {
                return err(format!("unsupported WAV encoding (format {format}, {bits} bits, {ch} channels)"));
            }
            let n = len.min(b.len() - body) / (bytes * ch);
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut s = 0f64;
                for c in 0..ch {
                    let o = body + (i * ch + c) * bytes;
                    s += match (float, bytes) {
                        (true, 8) => f64::from_le_bytes(b[o..o + 8].try_into().unwrap()),
                        (true, _) => f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64,
                        (false, 1) => (b[o] as f64 - 128.0) / 128.0,
                        (false, 2) => i16::from_le_bytes([b[o], b[o + 1]]) as f64 / 32768.0,
                        (false, 3) => (i32::from_le_bytes([0, b[o], b[o + 1], b[o + 2]]) >> 8) as f64 / 8388608.0,
                        (false, _) => i32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64 / 2147483648.0,
                    };
                }
                out.push((s / ch as f64) as f32);
            }
            return Ok((out, rate));
        }
        p = body + len + (len & 1);
    }
    err("WAV data chunk missing")
}

pub fn read_wav_file(path: impl AsRef<Path>) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => return err(format!("{}: {e}", path.display())),
    };
    read_wav(&bytes).or_else(|e| err(format!("{}: {e}", path.display())))
}

/// Windowed-sinc resampler, good enough for speech when the file is not 16 kHz.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || from == 0 || to == 0 {
        return x.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let n = (x.len() as f64 / ratio).round() as usize;
    let cutoff = (to as f64 / from as f64).min(1.0);
    let half = 16f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = i as f64 * ratio;
        let lo = ((c - half / cutoff).ceil() as isize).max(0);
        let hi = ((c + half / cutoff).floor() as isize).min(x.len() as isize - 1);
        let (mut acc, mut wsum) = (0f64, 0f64);
        for j in lo..=hi {
            let t = (j as f64 - c) * cutoff;
            let mut w = if t == 0.0 { 1.0 } else { (PI * t).sin() / (PI * t) };
            w *= 0.5 + 0.5 * (PI * t / half).cos();
            acc += x[j as usize] as f64 * w;
            wsum += w;
        }
        out.push(if wsum != 0.0 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}
