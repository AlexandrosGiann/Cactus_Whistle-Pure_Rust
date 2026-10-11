//! Quantised linear algebra: CQ matrices and the int8 activation side of a matmul.

use std::sync::{Arc, OnceLock};

use crate::cact::{Cact, CQ};
use crate::{err, Result};

pub(crate) const INV127: f32 = 1.0 / 127.0;

/// In-place unnormalised fast Walsh-Hadamard transform.
pub(crate) fn fwht(a: &mut [f32]) {
    let n = a.len();
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let (x, y) = (a[j], a[j + len]);
                a[j] = x + y;
                a[j + len] = x - y;
            }
            i += len << 1;
        }
        len <<= 1;
    }
}

/// Round to nearest, halves away from zero.
#[inline]
fn round_away(y: f32) -> f64 {
    let y = y as f64;
    if y < 0.0 { -(0.5 - y).floor() } else { (y + 0.5).floor() }
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// A CQ matrix [out, in], kept the way the engine consumes it: one int8 centroid
/// per weight (in the Hadamard-rotated domain) plus one float scale per row group.
pub(crate) struct QMat {
    pub out: usize,
    pub inp: usize,
    g: usize,
    n_g: usize,
    in_pad: usize,
    bits: usize,
    row_bytes: usize,
    table: Vec<i8>,
    data: Arc<Vec<u8>>,
    base: usize,
    nf: Vec<f32>, // group norm * codebook scale
    w: OnceLock<Vec<i8>>,
}

impl QMat {
    pub fn new(cact: &Cact, index: usize) -> Result<QMat> {
        let t = cact.tensor(index)?;
        if t.dtype != CQ || t.shape.len() != 2 {
            return err(format!("tensor #{index} is not a CQ matrix"));
        }
        let (out, inp, g, bits) = (t.shape[0], t.shape[1], t.group, t.bits);
        if g == 0 || !g.is_power_of_two() {
            return err(format!("tensor #{index} has an unsupported CQ group of {g}"));
        }
        let cb = cact.codebook(bits)?;
        let n_g = inp.div_ceil(g);
        let in_pad = n_g * g;
        let row_bytes = (in_pad * bits) >> 3;
        if out * row_bytes + out * n_g * 2 != t.nbytes {
            return err(format!("tensor #{index} has an unexpected CQ layout"));
        }
        let mx = cb.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
        let cb_scale = (mx / 127.0) as f32;
        let table = cb.iter().map(|&v| (v as f64 / cb_scale as f64 + 0.5).floor() as i8).collect();
        let norms = t.offset + out * row_bytes;
        let nf = (0..out * n_g).map(|i| cact.half_at(norms + 2 * i) * cb_scale).collect();
        Ok(QMat {
            out, inp, g, n_g, in_pad, bits, row_bytes, table,
            data: cact.data.clone(),
            base: t.offset,
            nf,
            w: OnceLock::new(),
        })
    }

    fn decode_row_into(&self, r: usize, dst: &mut [i8]) {
        let p = self.base + r * self.row_bytes;
        let src = &self.data[p..p + self.row_bytes];
        let table = &self.table;
        match self.bits {
            2 => {
                for (d, &b) in dst.chunks_exact_mut(4).zip(src) {
                    d[0] = table[(b & 3) as usize];
                    d[1] = table[((b >> 2) & 3) as usize];
                    d[2] = table[((b >> 4) & 3) as usize];
                    d[3] = table[(b >> 6) as usize];
                }
            }
            4 => {
                for (d, &b) in dst.chunks_exact_mut(2).zip(src) {
                    d[0] = table[(b & 15) as usize];
                    d[1] = table[(b >> 4) as usize];
                }
            }
            bits => {
                let mask = (1usize << bits) - 1;
                for (k, d) in dst.iter_mut().enumerate().take(self.in_pad) {
                    let bit = k * bits;
                    let (byte, sh) = (bit >> 3, bit & 7);
                    let lo = src[byte] as usize;
                    let hi = src.get(byte + 1).copied().unwrap_or(0) as usize;
                    *d = table[((lo | (hi << 8)) >> sh) & mask];
                }
            }
        }
    }

    /// Every weight as an int8 centroid, decoded on first use. Matrices that are
    /// only read row by row (the engram tables) never pay for this.
    fn weights(&self) -> &[i8] {
        self.w.get_or_init(|| {
            let mut w = vec![0i8; self.out * self.in_pad];
            for (r, row) in w.chunks_exact_mut(self.in_pad).enumerate() {
                self.decode_row_into(r, row);
            }
            w
        })
    }

    /// out = W times the activation held by `act`.
    pub fn apply(&self, act: &Quantizer, out: &mut [f32]) {
        self.apply_rows(act, 0, out);
    }

    /// out[i] = row (row_start + i) of W times the activation held by `act`.
    pub fn apply_rows(&self, act: &Quantizer, row_start: usize, out: &mut [f32]) {
        assert_eq!(self.g, act.g, "activation and weight group sizes differ");
        assert!(act.n_g >= self.n_g && row_start + out.len() <= self.out);
        let (g, n_g, in_pad) = (self.g, self.n_g, self.in_pad);
        let w = self.weights();
        for (i, o) in out.iter_mut().enumerate() {
            let r = row_start + i;
            let row = &w[r * in_pad..(r + 1) * in_pad];
            let nf = &self.nf[r * n_g..(r + 1) * n_g];
            let mut acc = 0f64;
            for gi in 0..n_g {
                let d = dot_i8(&row[gi * g..(gi + 1) * g], &act.q[gi * g..(gi + 1) * g]);
                acc += d as f64 * nf[gi] as f64 * act.s[gi] as f64;
            }
            *o = acc as f32;
        }
    }

    /// One fully dequantised row (for table lookups such as the engrams).
    pub fn row_float(&self, r: usize, dst: &mut [f32]) {
        let (g, n_g) = (self.g, self.n_g);
        let mut tmp = vec![0i8; self.in_pad];
        let mut f = vec![0f32; g];
        let inv = 1.0 / (g as f64).sqrt();
        self.decode_row_into(r, &mut tmp);
        for gi in 0..n_g {
            let n = self.nf[r * n_g + gi];
            for k in 0..g {
                f[k] = tmp[gi * g + k] as f32 * n;
            }
            fwht(&mut f);
            for k in 0..g {
                if gi * g + k >= self.inp {
                    break;
                }
                dst[gi * g + k] = (f[k] as f64 * inv) as f32;
            }
        }
    }
}

/// The activation side of a matmul: each group of G inputs is Hadamard-rotated and
/// quantised to int8 with its own scale.
pub(crate) struct Quantizer {
    g: usize,
    inv: f32,
    tmp: Vec<f32>,
    q: Vec<i8>,
    s: Vec<f32>,
    n_g: usize,
}

impl Quantizer {
    pub fn new(g: usize) -> Quantizer {
        Quantizer {
            g,
            inv: (1.0 / (g as f64).sqrt()) as f32,
            tmp: vec![0.0; g],
            q: Vec::new(),
            s: Vec::new(),
            n_g: 0,
        }
    }

    pub fn set(&mut self, x: &[f32]) {
        let (g, dim) = (self.g, x.len());
        let n_g = dim.div_ceil(g);
        if self.q.len() < n_g * g {
            self.q.resize(n_g * g, 0);
            self.s.resize(n_g, 0.0);
        }
        let tmp = &mut self.tmp;
        for gi in 0..n_g {
            let base = gi * g;
            let n = g.min(dim - base);
            tmp[..n].copy_from_slice(&x[base..base + n]);
            tmp[n..].fill(0.0);
            fwht(tmp);
            let mut mx = 0f64;
            for v in tmp.iter_mut() {
                let wide = *v as f64 * self.inv as f64;
                *v = wide as f32;
                mx = mx.max(wide.abs());
            }
            let sc = if mx > 0.0 { (mx * INV127 as f64) as f32 } else { 1.0 };
            let r = 1.0 / sc;
            self.s[gi] = sc;
            for (q, &v) in self.q[base..base + g].iter_mut().zip(tmp.iter()) {
                *q = round_away(v * r).clamp(-127.0, 127.0) as i8;
            }
        }
        self.n_g = n_g;
    }
}

/// Symmetric int8 fake quantisation with one scale (KV cache numerics).
pub(crate) fn fake_quant8(x: &mut [f32]) {
    let mx = x.iter().fold(0f32, |m, &v| m.max(v.abs()));
    if !(mx > 0.0) {
        return;
    }
    let sc = mx * INV127;
    let r = 1.0 / sc;
    for v in x.iter_mut() {
        *v = round_away(*v * r) as f32 * sc;
    }
}
