//! Blocks shared by the encoder and the decoder: RMS norm, the Hadamard MLP,
//! the multi-lane residual (mHC), rotary positions and the convolutional stem.

use std::sync::Arc;

use crate::cact::Cact;
use crate::frontend::Config;
use crate::quant::{QMat, Quantizer};
use crate::Result;

#[inline]
pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub(crate) fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// acc + a * b with a double intermediate, stored back as f32.
#[inline(always)]
pub(crate) fn madd(acc: f32, a: f32, b: f32) -> f32 {
    (acc as f64 + a as f64 * b as f64) as f32
}

#[inline(always)]
pub(crate) fn madd_d(acc: f32, a: f64, b: f32) -> f32 {
    (acc as f64 + a * b as f64) as f32
}

fn rms_inverse(x: &[f32]) -> f64 {
    let mut ss = 0f64;
    for &v in x {
        ss += v as f64 * v as f64;
    }
    1.0 / (ss / x.len() as f64 + 1e-6).sqrt()
}

/// out = (1 + scale) * x / sqrt(mean(x^2) + 1e-6)
pub(crate) fn rms_norm(x: &[f32], scale: Option<&[f32]>, out: &mut [f32]) {
    let inv = rms_inverse(x);
    match scale {
        Some(s) => {
            for i in 0..x.len() {
                out[i] = ((1.0 + s[i] as f64) * x[i] as f64 * inv) as f32;
            }
        }
        None => {
            for i in 0..x.len() {
                out[i] = (x[i] as f64 * inv) as f32;
            }
        }
    }
}

pub(crate) fn rms_norm_in_place(x: &mut [f32], scale: &[f32]) {
    let inv = rms_inverse(x);
    for (v, &s) in x.iter_mut().zip(scale) {
        *v = ((1.0 + s as f64) * *v as f64 * inv) as f32;
    }
}

pub(crate) fn softmax_in_place(a: &mut [f64]) {
    let m = a.iter().fold(f64::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    let mut s = 0f64;
    for v in a.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in a.iter_mut() {
        *v /= s;
    }
}

// --------------------------------------------------------- Hadamard MLP

pub(crate) type Perms = Arc<[Vec<usize>; 2]>;

pub(crate) struct HadaMlp {
    d1: Vec<f32>,
    d2: Vec<f32>,
    b2: Vec<f32>,
    d3: Vec<f32>,
    d4: Vec<f32>,
    wa: [Vec<f32>; 3],
    wb: [Vec<f32>; 3],
    cond_v: Vec<f32>,
    cond_u: Vec<f32>,
    n: usize,
    ba: usize,
    bb: usize,
    rank: usize,
    perms: Perms,
}

impl HadaMlp {
    /// base: index of d1; order d1 d2 b2 d3 d4 w1a w1b w2a w2b w3a w3b cond_v cond_u
    pub fn new(cact: &Cact, base: usize, perms: &Perms) -> Result<HadaMlp> {
        let f = |k: usize| cact.fp16(base + k);
        let d1 = f(0)?;
        let wa = [f(5)?, f(7)?, f(9)?];
        let wb = [f(6)?, f(8)?, f(10)?];
        let cond_u = f(12)?;
        let n = d1.len();
        Ok(HadaMlp {
            ba: (wa[0].len() as f64).sqrt().round() as usize,
            bb: (wb[0].len() as f64).sqrt().round() as usize,
            rank: cond_u.len() / n.max(1),
            d1,
            d2: f(1)?,
            b2: f(2)?,
            d3: f(3)?,
            d4: f(4)?,
            wa,
            wb,
            cond_v: f(11)?,
            cond_u,
            n,
            perms: perms.clone(),
        })
    }

    /// out[k,l] = sum_ij z[i,j] a[i,k] b[j,l]
    fn kron(&self, z: &[f32], a: &[f32], b: &[f32], t: &mut [f32], out: &mut [f32]) {
        let (na, nb) = (self.ba, self.bb);
        for i in 0..na {
            let trow = &mut t[i * nb..(i + 1) * nb];
            trow.fill(0.0);
            for j in 0..nb {
                let v = z[i * nb + j];
                if v == 0.0 {
                    continue;
                }
                let brow = &b[j * nb..(j + 1) * nb];
                for l in 0..nb {
                    trow[l] = madd(trow[l], v, brow[l]);
                }
            }
        }
        out[..na * nb].fill(0.0);
        for i in 0..na {
            let trow = &t[i * nb..(i + 1) * nb];
            for k in 0..na {
                let w = a[i * na + k];
                let orow = &mut out[k * nb..(k + 1) * nb];
                for l in 0..nb {
                    orow[l] = madd(orow[l], w, trow[l]);
                }
            }
        }
    }

    pub fn run(&self, x: &[f32], out: &mut [f32]) {
        let (n, rank) = (self.n, self.rank);
        let (p1, p2) = (&self.perms[0], &self.perms[1]);
        let mut z = vec![0f32; n];
        let mut t = vec![0f32; n];
        let mut y = vec![0f32; n];
        let mut sm = vec![0f64; rank];
        for (r, s) in sm.iter_mut().enumerate() {
            let mut a = 0f64;
            for i in 0..n {
                a += x[i] as f64 * self.cond_v[i * rank + r] as f64;
            }
            *s = a;
        }
        softmax_in_place(&mut sm);
        for i in 0..n {
            z[i] = self.d1[i] * x[i];
        }
        self.kron(&z, &self.wa[0], &self.wb[0], &mut t, &mut y);
        for i in 0..n {
            let mut c = 1f64;
            for r in 0..rank {
                c += sm[r] * self.cond_u[r * n + i] as f64;
            }
            z[i] = silu(self.d2[i] as f64 * c * y[p1[i]] as f64 + self.b2[i] as f64) as f32;
        }
        self.kron(&z, &self.wa[1], &self.wb[1], &mut t, &mut y);
        for i in 0..n {
            z[i] = self.d3[i] * y[p2[i]];
        }
        self.kron(&z, &self.wa[2], &self.wb[2], &mut t, &mut y);
        for i in 0..n {
            out[i] = self.d4[i] * y[i];
        }
    }
}

// ------------------------------------------------- multi-lane residual (mHC)

pub(crate) struct Mhc {
    a_pre: Vec<f32>,
    a_post: Vec<f32>,
    a_res: Vec<f32>,
    b_pre: Vec<f32>,
    b_post: Vec<f32>,
    b_res: Vec<f32>,
    phi_pre: QMat,
    phi_post: QMat,
    phi_res: QMat,
    lanes: usize,
    width: usize,
}

/// The mixing weights begin() leaves for finish().
#[derive(Clone)]
pub(crate) struct MhcState {
    hpost: Vec<f64>,
    hres: Vec<f64>,
}

impl Mhc {
    /// base: a_pre a_post a_res b_pre b_post b_res phi_pre phi_post phi_res
    pub fn new(cact: &Cact, base: usize, lanes: usize, width: usize) -> Result<Mhc> {
        Ok(Mhc {
            a_pre: cact.fp16(base)?,
            a_post: cact.fp16(base + 1)?,
            a_res: cact.fp16(base + 2)?,
            b_pre: cact.fp16(base + 3)?,
            b_post: cact.fp16(base + 4)?,
            b_res: cact.fp16(base + 5)?,
            phi_pre: QMat::new(cact, base + 6)?,
            phi_post: QMat::new(cact, base + 7)?,
            phi_res: QMat::new(cact, base + 8)?,
            lanes,
            width,
        })
    }

    pub fn state(&self) -> MhcState {
        MhcState { hpost: vec![0.0; self.lanes], hres: vec![0.0; self.lanes * self.lanes] }
    }

    /// Reads the lane stream [lanes*width], writes the block input u and the mixing state.
    pub fn begin(&self, layer: usize, stream: &[f32], u: &mut [f32], st: &mut MhcState, quant: &mut Quantizer) {
        let (n, c_n) = (self.lanes, self.width);
        let nn = n * n;
        let mut nx = vec![0f32; n * c_n];
        rms_norm(stream, None, &mut nx);
        quant.set(&nx);
        let mut pre = vec![0f32; n];
        let mut post = vec![0f32; n];
        let mut res = vec![0f32; nn];
        self.phi_pre.apply_rows(quant, layer * n, &mut pre);
        self.phi_post.apply_rows(quant, layer * n, &mut post);
        self.phi_res.apply_rows(quant, layer * nn, &mut res);
        let lane = layer % n;
        u.fill(0.0);
        for i in 0..n {
            let bias = if i == lane { 4.0 } else { -4.0 };
            let hpre = sigmoid(
                self.a_pre[layer] as f64 * pre[i] as f64 + self.b_pre[layer * n + i] as f64 + bias,
            );
            let src = &stream[i * c_n..(i + 1) * c_n];
            for c in 0..c_n {
                u[c] = madd_d(u[c], hpre, src[c]);
            }
            let bias = if i == lane { 0.0 } else { -4.0 };
            st.hpost[i] = 2.0
                * sigmoid(self.a_post[layer] as f64 * post[i] as f64 + self.b_post[layer * n + i] as f64 + bias);
        }
        // Sinkhorn in the log domain, 20 rounds of row then column normalisation.
        let k = &mut st.hres;
        for i in 0..nn {
            k[i] = self.a_res[layer] as f64 * res[i] as f64 + self.b_res[layer * nn + i] as f64;
        }
        for _ in 0..20 {
            for i in 0..n {
                let mut m = f64::NEG_INFINITY;
                for j in 0..n {
                    if k[i * n + j] > m {
                        m = k[i * n + j];
                    }
                }
                let mut sum = 0f64;
                for j in 0..n {
                    sum += (k[i * n + j] - m).exp();
                }
                let sum = m + sum.ln();
                for j in 0..n {
                    k[i * n + j] -= sum;
                }
            }
            for j in 0..n {
                let mut m = f64::NEG_INFINITY;
                for i in 0..n {
                    if k[i * n + j] > m {
                        m = k[i * n + j];
                    }
                }
                let mut sum = 0f64;
                for i in 0..n {
                    sum += (k[i * n + j] - m).exp();
                }
                let sum = m + sum.ln();
                for i in 0..n {
                    k[i * n + j] -= sum;
                }
            }
        }
        for v in k.iter_mut() {
            *v = v.exp();
        }
    }

    /// stream <- hres @ stream + hpost * (x - u)
    pub fn finish(&self, stream: &mut [f32], x: &[f32], u: &[f32], st: &MhcState) {
        let (n, c_n) = (self.lanes, self.width);
        let mut col = vec![0f64; n];
        for c in 0..c_n {
            let y = x[c] as f64 - u[c] as f64;
            for j in 0..n {
                col[j] = stream[j * c_n + c] as f64;
            }
            for i in 0..n {
                let mut a = st.hpost[i] * y;
                for j in 0..n {
                    a += st.hres[i * n + j] * col[j];
                }
                stream[i * c_n + c] = a as f32;
            }
        }
    }
}

// --------------------------------------------------------------- rotary

pub(crate) struct Rope {
    freq: Vec<f64>,
}

impl Rope {
    pub fn new(dim: usize, theta: f32) -> Rope {
        let half = dim >> 1;
        Rope {
            freq: (0..half).map(|i| 1.0 / (theta as f64).powf((2 * i) as f64 / dim as f64)).collect(),
        }
    }

    pub fn apply(&self, x: &mut [f32], pos: usize) {
        let h = self.freq.len();
        for i in 0..h {
            let (s, c) = (pos as f64 * self.freq[i]).sin_cos();
            let (a, b) = (x[i] as f64, x[h + i] as f64);
            x[i] = (a * c - b * s) as f32;
            x[h + i] = (b * c + a * s) as f32;
        }
    }
}

// ------------------------------------------------------------------ stem

pub(crate) struct Stem {
    c: usize,
    mels: usize,
    w: Vec<f32>,
    dw: Vec<Vec<f32>>,
    pw: Vec<QMat>,
    out: QMat,
}

struct Map {
    data: Vec<f32>,
    t: usize,
    f: usize,
}

/// 3x3, stride 2, pad 1 over (time, freq). in_c == 1 for the first stage, else depthwise.
fn conv3s2(inp: &[f32], t_n: usize, f_n: usize, in_c: usize, w: &[f32], c_n: usize) -> Map {
    let (to, fo) = ((t_n + 1) >> 1, (f_n + 1) >> 1);
    let mut out = vec![0f32; to * fo * c_n];
    for s in 0..to {
        for p in 0..fo {
            let o = &mut out[(s * fo + p) * c_n..][..c_n];
            for m in 0..3 {
                let t = 2 * s + m;
                if t < 1 || t > t_n {
                    continue;
                }
                let t = t - 1;
                for kf in 0..3 {
                    let f = 2 * p + kf;
                    if f < 1 || f > f_n {
                        continue;
                    }
                    let f = f - 1;
                    let wk = &w[(m * 3 + kf) * c_n..][..c_n];
                    if in_c == 1 {
                        let v = inp[t * f_n + f];
                        for c in 0..c_n {
                            o[c] = madd(o[c], wk[c], v);
                        }
                    } else {
                        let src = &inp[(t * f_n + f) * c_n..][..c_n];
                        for c in 0..c_n {
                            o[c] = madd(o[c], wk[c], src[c]);
                        }
                    }
                }
            }
        }
    }
    Map { data: out, t: to, f: fo }
}

impl Stem {
    /// base: w, then (dw, pw) per further stage, then out
    pub fn new(cact: &Cact, base: usize, cfg: &Config) -> Result<Stem> {
        let mut dw = Vec::new();
        let mut pw = Vec::new();
        for s in 1..cfg.stem_stages {
            dw.push(cact.fp16(base + 2 * s - 1)?);
            pw.push(QMat::new(cact, base + 2 * s)?);
        }
        Ok(Stem {
            c: cfg.stem_width,
            mels: cfg.mel_bins,
            w: cact.fp16(base)?,
            dw,
            pw,
            out: QMat::new(cact, base + 2 * cfg.stem_stages - 1)?,
        })
    }

    /// feat: [T, mels] -> ([T', width], T')
    pub fn run(&self, feat: &[f32], t_n: usize, quant: &mut Quantizer) -> (Vec<f32>, usize) {
        let c_n = self.c;
        let mut x = conv3s2(feat, t_n, self.mels, 1, &self.w, c_n);
        for v in x.data.iter_mut() {
            *v = silu(*v as f64) as f32;
        }
        for (dw, pw) in self.dw.iter().zip(&self.pw) {
            let y = conv3s2(&x.data, x.t, x.f, c_n, dw, c_n);
            let rows = y.t * y.f;
            let mut z = vec![0f32; rows * pw.out];
            for i in 0..rows {
                quant.set(&y.data[i * c_n..(i + 1) * c_n]);
                pw.apply(quant, &mut z[i * pw.out..(i + 1) * pw.out]);
            }
            for v in z.iter_mut() {
                *v = silu(*v as f64) as f32;
            }
            x = Map { data: z, t: y.t, f: y.f };
        }
        let (f_n, w_n) = (x.f, self.out.out);
        let mut flat = vec![0f32; c_n * f_n];
        let mut out = vec![0f32; x.t * w_n];
        for t in 0..x.t {
            for f in 0..f_n {
                let src = &x.data[(t * f_n + f) * c_n..][..c_n];
                for c in 0..c_n {
                    flat[c * f_n + f] = src[c];
                }
            }
            quant.set(&flat);
            self.out.apply(quant, &mut out[t * w_n..(t + 1) * w_n]);
        }
        (out, x.t)
    }
}
