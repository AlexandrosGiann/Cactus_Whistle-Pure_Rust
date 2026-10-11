//! Audio encoder: 8 blocks on 4 residual lanes.

use crate::cact::{Cact, Header};
use crate::frontend::Config;
use crate::layers::{
    madd, madd_d, rms_norm, rms_norm_in_place, sigmoid, silu, softmax_in_place, HadaMlp, Mhc, Perms, Rope,
};
use crate::quant::{fake_quant8, QMat, Quantizer};
use crate::Result;

pub(crate) const TENSORS_PER_LAYER: usize = 43;

struct EncoderLayer {
    pre_hada0: Vec<f32>,
    norm_in: Vec<f32>,
    post_attn: Vec<f32>,
    conv_norm: Vec<f32>,
    conv_out_norm: Vec<f32>,
    pre_hada: Vec<f32>,
    q: QMat,
    k: QMat,
    v: QMat,
    gate: QMat,
    out_proj: QMat,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_gate: f64,
    pw1: QMat,
    pw2: QMat,
    dw: Vec<f32>,
    hada0: HadaMlp,
    hada: HadaMlp,
}

impl EncoderLayer {
    fn new(cact: &Cact, b: usize, perms: &Perms) -> Result<EncoderLayer> {
        Ok(EncoderLayer {
            pre_hada0: cact.fp16(b)?,
            norm_in: cact.fp16(b + 1)?,
            post_attn: cact.fp16(b + 2)?,
            conv_norm: cact.fp16(b + 3)?,
            conv_out_norm: cact.fp16(b + 4)?,
            pre_hada: cact.fp16(b + 5)?,
            q: QMat::new(cact, b + 6)?,
            k: QMat::new(cact, b + 7)?,
            v: QMat::new(cact, b + 8)?,
            gate: QMat::new(cact, b + 9)?,
            out_proj: QMat::new(cact, b + 10)?,
            q_norm: cact.fp16(b + 11)?,
            k_norm: cact.fp16(b + 12)?,
            attn_gate: sigmoid(cact.fp16_scalar(b + 13)? as f64),
            pw1: QMat::new(cact, b + 14)?,
            pw2: QMat::new(cact, b + 15)?,
            dw: cact.fp16(b + 16)?,
            hada0: HadaMlp::new(cact, b + 17, perms)?,
            hada: HadaMlp::new(cact, b + 30, perms)?,
        })
    }
}

pub(crate) struct Encoder {
    w: usize,
    lanes: usize,
    heads: usize,
    kv_heads: usize,
    qk: usize,
    vd: usize,
    kernel: usize,
    layers: Vec<EncoderLayer>,
    mhc: Mhc,
    final_norm: Vec<f32>,
    rope: Rope,
}

impl Encoder {
    pub fn new(cact: &Cact, base: usize, cfg: &Config, hdr: &Header, perms: &Perms) -> Result<Encoder> {
        let n_layers = cfg.enc_layers;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(EncoderLayer::new(cact, base + TENSORS_PER_LAYER * l, perms)?);
        }
        let tail = base + TENSORS_PER_LAYER * n_layers;
        Ok(Encoder {
            w: hdr.width,
            lanes: cfg.enc_lanes,
            heads: hdr.heads,
            kv_heads: hdr.kv_heads,
            qk: hdr.qk_dim,
            vd: hdr.v_dim,
            kernel: cfg.conv_kernel,
            layers,
            mhc: Mhc::new(cact, tail, cfg.enc_lanes, hdr.width)?,
            final_norm: cact.fp16(tail + 9)?,
            rope: Rope::new(hdr.qk_dim, hdr.rope_theta),
        })
    }

    /// x: [T, W] stem output. Returns [T, W].
    pub fn run(&self, x: &[f32], t_n: usize, quant: &mut Quantizer, progress: &mut dyn FnMut(f32)) -> Vec<f32> {
        let (w, n) = (self.w, self.lanes);
        let nc = n * w;
        let (h_n, kh_n, qk, vd) = (self.heads, self.kv_heads, self.qk, self.vd);
        let (rep, half) = (h_n / kh_n, self.kernel >> 1);
        let (qd, kd, vdim, od) = (h_n * qk, kh_n * qk, kh_n * vd, h_n * vd);
        let scale = 1.0 / (qk as f64).sqrt();

        let mut stream = vec![0f32; t_n * nc];
        for t in 0..t_n {
            for i in 0..n {
                stream[t * nc + i * w..][..w].copy_from_slice(&x[t * w..(t + 1) * w]);
            }
        }
        let mut states = vec![self.mhc.state(); t_n];
        let mut u = vec![0f32; t_n * w];
        let mut h = vec![0f32; t_n * w];
        let mut tmp = vec![0f32; w];
        let mut y = vec![0f32; w];
        let mut q_all = vec![0f32; t_n * qd];
        let mut k_all = vec![0f32; t_n * kd];
        let mut v_all = vec![0f32; t_n * vdim];
        let mut gate = vec![0f32; t_n * od];
        let mut att = vec![0f32; t_n * od];
        let mut p1 = vec![0f32; 2 * w];
        let mut glu = vec![0f32; t_n * w];
        let mut cv = vec![0f32; t_n * w];
        let mut logit = vec![0f64; t_n];

        for (l, ly) in self.layers.iter().enumerate() {
            for t in 0..t_n {
                self.mhc.begin(l, &stream[t * nc..(t + 1) * nc], &mut u[t * w..(t + 1) * w], &mut states[t], quant);
            }
            h.copy_from_slice(&u);
            // half-step Hadamard MLP
            for t in 0..t_n {
                let ht = &mut h[t * w..(t + 1) * w];
                rms_norm(ht, Some(&ly.pre_hada0), &mut tmp);
                ly.hada0.run(&tmp, &mut y);
                for c in 0..w {
                    ht[c] = madd_d(ht[c], 0.5, y[c]);
                }
            }
            // self attention (not causal)
            for t in 0..t_n {
                rms_norm(&h[t * w..(t + 1) * w], Some(&ly.norm_in), &mut tmp);
                quant.set(&tmp);
                ly.q.apply(quant, &mut q_all[t * qd..(t + 1) * qd]);
                ly.k.apply(quant, &mut k_all[t * kd..(t + 1) * kd]);
                ly.v.apply(quant, &mut v_all[t * vdim..(t + 1) * vdim]);
                ly.gate.apply(quant, &mut gate[t * od..(t + 1) * od]);
                for hd in 0..h_n {
                    let q = &mut q_all[t * qd + hd * qk..][..qk];
                    rms_norm_in_place(q, &ly.q_norm);
                    self.rope.apply(q, t);
                    fake_quant8(q);
                }
                for hd in 0..kh_n {
                    let k = &mut k_all[t * kd + hd * qk..][..qk];
                    rms_norm_in_place(k, &ly.k_norm);
                    self.rope.apply(k, t);
                    fake_quant8(k);
                    fake_quant8(&mut v_all[t * vdim + hd * vd..][..vd]);
                }
            }
            for t in 0..t_n {
                for hd in 0..h_n {
                    let kh = hd / rep;
                    let q = &q_all[t * qd + hd * qk..][..qk];
                    for s in 0..t_n {
                        let k = &k_all[s * kd + kh * qk..][..qk];
                        let mut dot = 0f64;
                        for d in 0..qk {
                            dot += q[d] as f64 * k[d] as f64;
                        }
                        logit[s] = dot * scale;
                    }
                    softmax_in_place(&mut logit);
                    let o = &mut att[t * od + hd * vd..][..vd];
                    o.fill(0.0);
                    for s in 0..t_n {
                        let wgt = logit[s];
                        let v = &v_all[s * vdim + kh * vd..][..vd];
                        for d in 0..vd {
                            o[d] = madd_d(o[d], wgt, v[d]);
                        }
                    }
                }
                for c in t * od..(t + 1) * od {
                    att[c] = (att[c] as f64 * sigmoid(gate[c] as f64)) as f32;
                }
            }
            for t in 0..t_n {
                quant.set(&att[t * od..(t + 1) * od]);
                ly.out_proj.apply(quant, &mut y);
                rms_norm_in_place(&mut y, &ly.post_attn);
                let ht = &mut h[t * w..(t + 1) * w];
                for c in 0..w {
                    ht[c] = madd_d(ht[c], ly.attn_gate, y[c]);
                }
            }
            // convolution module: norm, pointwise x2, GLU, depthwise over time, norm, SiLU, pointwise
            for t in 0..t_n {
                rms_norm(&h[t * w..(t + 1) * w], Some(&ly.conv_norm), &mut tmp);
                quant.set(&tmp);
                ly.pw1.apply(quant, &mut p1);
                for c in 0..w {
                    glu[t * w + c] = (p1[c] as f64 * sigmoid(p1[w + c] as f64)) as f32;
                }
            }
            for t in 0..t_n {
                let o = &mut cv[t * w..(t + 1) * w];
                o.fill(0.0);
                for kx in 0..self.kernel {
                    let src = t + kx;
                    if src < half || src - half >= t_n {
                        continue;
                    }
                    let (wk, g) = (&ly.dw[kx * w..(kx + 1) * w], &glu[(src - half) * w..][..w]);
                    for c in 0..w {
                        o[c] = madd(o[c], wk[c], g[c]);
                    }
                }
                rms_norm_in_place(o, &ly.conv_out_norm);
                for v in o.iter_mut() {
                    *v = silu(*v as f64) as f32;
                }
            }
            for t in 0..t_n {
                quant.set(&cv[t * w..(t + 1) * w]);
                ly.pw2.apply(quant, &mut y);
                let ht = &mut h[t * w..(t + 1) * w];
                for c in 0..w {
                    ht[c] = (ht[c] as f64 + y[c] as f64) as f32;
                }
                // second half-step Hadamard MLP
                rms_norm(ht, Some(&ly.pre_hada), &mut tmp);
                ly.hada.run(&tmp, &mut y);
                for c in 0..w {
                    ht[c] = madd_d(ht[c], 0.5, y[c]);
                }
                self.mhc.finish(&mut stream[t * nc..(t + 1) * nc], ht, &u[t * w..(t + 1) * w], &states[t]);
            }
            progress((l + 1) as f32 / self.layers.len() as f32);
        }
        let mut out = vec![0f32; t_n * w];
        for t in 0..t_n {
            for c in 0..w {
                let mut a = 0f64;
                for i in 0..n {
                    a += stream[t * nc + i * w + c] as f64;
                }
                tmp[c] = (a / n as f64) as f32;
            }
            rms_norm(&tmp, Some(&self.final_norm), &mut out[t * w..(t + 1) * w]);
        }
        out
    }
}
