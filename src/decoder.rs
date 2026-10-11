//! Text decoder: causal attention with conv taps, gated cross attention over the
//! clip, Hadamard MLP, engram lookups, tied embedding as the output head.

use std::sync::Arc;

use crate::cact::{Cact, Header};
use crate::layers::{madd, madd_d, rms_norm, rms_norm_in_place, sigmoid, HadaMlp, Mhc, Perms, Rope};
use crate::quant::{fake_quant8, QMat, Quantizer};
use crate::Result;

/// Where the decoder's tensors sit in the archive.
pub(crate) struct Layout {
    pub embedding: usize,
    pub dec_layer0: usize,
    pub dec_per_layer: usize,
    pub dec_mhc: usize,
    pub engram0: usize,
    pub final_norm: usize,
    pub cross0: usize,
    pub pe_gate: usize,
}

struct Engram {
    tables: QMat,
    key_proj: QMat,
    value_proj: QMat,
    taps: Vec<f32>,
    slots: usize,
    sub: usize,
    n_tables: usize,
    orders: Vec<usize>,
    heads: usize,
    stride: usize,
    n_taps: usize,
    dilation: usize,
    w: usize,
}

impl Engram {
    fn new(cact: &Cact, base: usize, hdr: &Header) -> Result<Engram> {
        let heads = hdr.engram_tables / hdr.engram_orders.len().max(1);
        Ok(Engram {
            tables: QMat::new(cact, base)?,
            key_proj: QMat::new(cact, base + 1)?,
            value_proj: QMat::new(cact, base + 2)?,
            taps: cact.fp16(base + 3)?,
            slots: hdr.engram_slots,
            sub: hdr.engram_sub_dim,
            n_tables: hdr.engram_tables,
            orders: hdr.engram_orders.clone(),
            heads,
            stride: if hdr.engram_seed_heads != 0 { hdr.engram_seed_heads } else { heads },
            n_taps: hdr.engram_taps,
            dilation: hdr.engram_dilation,
            w: hdr.width,
        })
    }

    /// Key and (int8 cached) value projection of the n-gram rows at position pos.
    fn project(&self, tokens: &[u32], pos: usize, quant: &mut Quantizer) -> (Vec<f32>, Vec<f32>) {
        let mut e = vec![0f32; self.n_tables * self.sub];
        for k in 0..self.n_tables {
            let oi = k / self.heads;
            let (h, order) = (k - oi * self.heads, self.orders[oi]);
            if pos + 1 < order {
                continue;
            }
            let mut acc = 0x9E37_79B9u32.wrapping_mul((oi * self.stride + h + 1) as u32);
            for j in 0..order {
                acc = (acc ^ tokens[pos - j]).wrapping_mul(16_777_619);
            }
            acc ^= acc >> 15;
            let row = k * self.slots + acc as usize % self.slots;
            self.tables.row_float(row, &mut e[k * self.sub..(k + 1) * self.sub]);
        }
        quant.set(&e);
        let mut key = vec![0f32; self.w];
        let mut val = vec![0f32; self.w];
        self.key_proj.apply(quant, &mut key);
        self.value_proj.apply(quant, &mut val);
        for chunk in val.chunks_exact_mut(32) {
            fake_quant8(chunk);
        }
        (key, val)
    }

    /// vals: per-position cached values up to pos. Returns the tapped value at pos.
    fn tapped(&self, vals: &[Arc<[f32]>], pos: usize) -> Vec<f32> {
        let w = self.w;
        let mut out = vec![0f32; w];
        for p in 0..self.n_taps {
            if pos < p * self.dilation {
                continue;
            }
            let v = &vals[pos - p * self.dilation];
            let taps = &self.taps[p * w..(p + 1) * w];
            for c in 0..w {
                out[c] = madd(out[c], v[c], taps[c]);
            }
        }
        out
    }
}

struct Taps {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

struct DecoderLayer {
    norm_in: Vec<f32>,
    q: QMat,
    k: QMat,
    v: QMat,
    taps: Option<Taps>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: QMat,
    out_proj: QMat,
    post_attn: Vec<f32>,
    attn_gate: f64,
    pre_hada: Vec<f32>,
    hada: HadaMlp,
    cross_norm: Vec<f32>,
    cq: QMat,
    ck: QMat,
    cv: QMat,
    c_gate: QMat,
    c_out: QMat,
    cq_norm: Vec<f32>,
    ck_norm: Vec<f32>,
    post_cross: Vec<f32>,
    cross_gate: f64,
}

impl DecoderLayer {
    fn new(cact: &Cact, b: usize, c: usize, perms: &Perms, has_taps: bool) -> Result<DecoderLayer> {
        let (taps, o) = if has_taps {
            (Some(Taps { q: cact.fp16(b + 4)?, k: cact.fp16(b + 5)?, v: cact.fp16(b + 6)? }), 7)
        } else {
            (None, 4)
        };
        Ok(DecoderLayer {
            norm_in: cact.fp16(b)?,
            q: QMat::new(cact, b + 1)?,
            k: QMat::new(cact, b + 2)?,
            v: QMat::new(cact, b + 3)?,
            taps,
            q_norm: cact.fp16(b + o)?,
            k_norm: cact.fp16(b + o + 1)?,
            gate: QMat::new(cact, b + o + 2)?,
            out_proj: QMat::new(cact, b + o + 3)?,
            post_attn: cact.fp16(b + o + 4)?,
            attn_gate: sigmoid(cact.fp16_scalar(b + o + 5)? as f64),
            pre_hada: cact.fp16(b + o + 6)?,
            hada: HadaMlp::new(cact, b + o + 7, perms)?,
            cross_norm: cact.fp16(c)?,
            cq: QMat::new(cact, c + 1)?,
            ck: QMat::new(cact, c + 2)?,
            cv: QMat::new(cact, c + 3)?,
            c_gate: QMat::new(cact, c + 4)?,
            c_out: QMat::new(cact, c + 5)?,
            cq_norm: cact.fp16(c + 6)?,
            ck_norm: cact.fp16(c + 7)?,
            post_cross: cact.fp16(c + 8)?,
            cross_gate: sigmoid(cact.fp16_scalar(c + 9)? as f64),
        })
    }
}

/// Per-layer cross-attention keys [T, H*qk] and values [T, H*vd] for one clip.
pub(crate) struct Cross {
    t: usize,
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

/// A hypothesis: its tokens plus everything the decoder caches per position.
/// Cloning shares the cached rows, so forking a beam is cheap.
#[derive(Clone)]
pub(crate) struct State {
    tokens: Vec<u32>,
    k: Vec<Vec<Arc<[f32]>>>,
    v: Vec<Vec<Arc<[f32]>>>,
    raw: Vec<Vec<Arc<[f32]>>>,
    evals: Vec<Vec<Arc<[f32]>>>,
}

pub(crate) struct Decoder {
    w: usize,
    h: usize,
    kh: usize,
    qk: usize,
    vd: usize,
    lanes: usize,
    n_taps: usize,
    embedding: QMat,
    layers: Vec<DecoderLayer>,
    mhc: Mhc,
    sites: Vec<usize>,
    engrams: Vec<Engram>,
    final_norm: Vec<f32>,
    pe_gate: f64,
    rope: Rope,
    emb_scale: f64,
}

impl Decoder {
    pub fn new(cact: &Cact, idx: &Layout, hdr: &Header, perms: &Perms) -> Result<Decoder> {
        let mut layers = Vec::with_capacity(hdr.layers);
        for l in 0..hdr.layers {
            layers.push(DecoderLayer::new(
                cact,
                idx.dec_layer0 + idx.dec_per_layer * l,
                idx.cross0 + 10 * l,
                perms,
                hdr.taps > 0,
            )?);
        }
        let mut engrams = Vec::new();
        for l in 0..hdr.engram_sites.len() {
            engrams.push(Engram::new(cact, idx.engram0 + 4 * l, hdr)?);
        }
        Ok(Decoder {
            w: hdr.width,
            h: hdr.heads,
            kh: hdr.kv_heads,
            qk: hdr.qk_dim,
            vd: hdr.v_dim,
            lanes: hdr.lanes,
            n_taps: hdr.taps,
            embedding: QMat::new(cact, idx.embedding)?,
            layers,
            mhc: Mhc::new(cact, idx.dec_mhc, hdr.lanes, hdr.width)?,
            sites: hdr.engram_sites.clone(),
            engrams,
            final_norm: cact.fp16(idx.final_norm)?,
            pe_gate: sigmoid(cact.fp16_scalar(idx.pe_gate)? as f64),
            rope: Rope::new(hdr.qk_dim, hdr.rope_theta),
            emb_scale: (hdr.width as f64).sqrt(),
        })
    }

    /// Sinusoidal positions gated into the encoder output, then per-layer cross K / V.
    pub fn prepare_cross(&self, enc: &[f32], t_n: usize, quant: &mut Quantizer) -> Cross {
        let (w, h_n, qk, vd) = (self.w, self.h, self.qk, self.vd);
        let half = w >> 1;
        let inc = -9.210340371976184 / (half as f64 - 1.0);
        let g = self.pe_gate;
        let mut mem = vec![0f32; t_n * w];
        for t in 0..t_n {
            for h in 0..half {
                let a = (h as f64 * inc).exp() * t as f64;
                mem[t * w + h] = (g * a.sin() + enc[t * w + h] as f64) as f32;
                mem[t * w + half + h] = (g * a.cos() + enc[t * w + half + h] as f64) as f32;
            }
        }
        let (qd, od) = (h_n * qk, h_n * vd);
        let mut layers: Vec<(Vec<f32>, Vec<f32>)> =
            self.layers.iter().map(|_| (vec![0f32; t_n * qd], vec![0f32; t_n * od])).collect();
        for t in 0..t_n {
            quant.set(&mem[t * w..(t + 1) * w]);
            for (ly, (k_all, v_all)) in self.layers.iter().zip(layers.iter_mut()) {
                let k = &mut k_all[t * qd..(t + 1) * qd];
                let v = &mut v_all[t * od..(t + 1) * od];
                ly.ck.apply(quant, k);
                ly.cv.apply(quant, v);
                for h in 0..h_n {
                    let kh = &mut k[h * qk..(h + 1) * qk];
                    rms_norm_in_place(kh, &ly.ck_norm);
                    fake_quant8(kh);
                    fake_quant8(&mut v[h * vd..(h + 1) * vd]);
                }
            }
        }
        Cross { t: t_n, layers }
    }

    pub fn new_state(&self) -> State {
        let per_layer = vec![Vec::new(); self.layers.len()];
        State {
            tokens: Vec::new(),
            k: per_layer.clone(),
            v: per_layer.clone(),
            raw: per_layer,
            evals: vec![Vec::new(); self.engrams.len()],
        }
    }

    /// Feeds one token and returns the logits for the next one.
    pub fn step(&self, st: &mut State, token: u32, cross: &Cross, quant: &mut Quantizer) -> Vec<f32> {
        let (w, n, h_n, kh_n, qk, vd) = (self.w, self.lanes, self.h, self.kh, self.qk, self.vd);
        let (rep, pos, t_n) = (h_n / kh_n, st.tokens.len(), cross.t);
        let (qd, kd, vdim, od) = (h_n * qk, kh_n * qk, kh_n * vd, h_n * vd);
        let scale = 1.0 / (qk as f64).sqrt();
        let mut logit = vec![0f64; t_n.max(pos + 1)];
        let mut stream = vec![0f32; n * w];
        let mut u = vec![0f32; w];
        let mut x = vec![0f32; w];
        let mut tmp = vec![0f32; w];
        let mut y = vec![0f32; w];
        let mut qv = vec![0f32; qd];
        let mut gt = vec![0f32; od];
        let mut att = vec![0f32; od];
        let mut mst = self.mhc.state();

        st.tokens.push(token);
        self.embedding.row_float(token as usize, &mut tmp);
        for i in 0..n {
            for c in 0..w {
                stream[i * w + c] = (tmp[c] as f64 * self.emb_scale) as f32;
            }
        }
        let mut eg = Vec::with_capacity(self.engrams.len());
        for (engram, vals) in self.engrams.iter().zip(st.evals.iter_mut()) {
            let (key, val) = engram.project(&st.tokens, pos, quant);
            vals.push(Arc::from(val));
            eg.push((key, engram.tapped(vals, pos)));
        }

        for (l, ly) in self.layers.iter().enumerate() {
            self.mhc.begin(l, &stream, &mut u, &mut mst, quant);
            x.copy_from_slice(&u);
            if let Some(site) = self.sites.iter().position(|&s| s == l) {
                let (key, val) = &eg[site];
                let (mut su, mut sk, mut dot) = (0f64, 0f64, 0f64);
                for c in 0..w {
                    let (uc, kc) = (u[c] as f64, key[c] as f64);
                    su += uc * uc;
                    sk += kc * kc;
                    dot += uc * kc;
                }
                dot /= (su / w as f64 + 1e-6).sqrt() * (sk / w as f64 + 1e-6).sqrt();
                let alpha = sigmoid(dot / (w as f64).sqrt());
                for c in 0..w {
                    x[c] = madd_d(x[c], alpha, val[c]);
                }
            }
            // causal self attention with conv taps on Q, K, V
            rms_norm(&x, Some(&ly.norm_in), &mut tmp);
            quant.set(&tmp);
            let mut raw = vec![0f32; qd + kd + vdim];
            ly.q.apply(quant, &mut raw[..qd]);
            ly.k.apply(quant, &mut raw[qd..qd + kd]);
            ly.v.apply(quant, &mut raw[qd + kd..]);
            ly.gate.apply(quant, &mut gt);
            let mut kt = vec![0f32; kd];
            let mut vt = vec![0f32; vdim];
            match &ly.taps {
                Some(taps) => {
                    st.raw[l].push(Arc::from(raw));
                    qv.fill(0.0);
                    for j in 0..self.n_taps.min(pos + 1) {
                        let r = &st.raw[l][pos - j];
                        for c in 0..qd {
                            qv[c] = madd(qv[c], taps.q[j * qd + c], r[c]);
                        }
                        for c in 0..kd {
                            kt[c] = madd(kt[c], taps.k[j * kd + c], r[qd + c]);
                        }
                        for c in 0..vdim {
                            vt[c] = madd(vt[c], taps.v[j * vdim + c], r[qd + kd + c]);
                        }
                    }
                }
                None => {
                    qv.copy_from_slice(&raw[..qd]);
                    kt.copy_from_slice(&raw[qd..qd + kd]);
                    vt.copy_from_slice(&raw[qd + kd..]);
                }
            }
            for hd in 0..h_n {
                let q = &mut qv[hd * qk..(hd + 1) * qk];
                rms_norm_in_place(q, &ly.q_norm);
                self.rope.apply(q, pos);
                fake_quant8(q);
            }
            for hd in 0..kh_n {
                let k = &mut kt[hd * qk..(hd + 1) * qk];
                rms_norm_in_place(k, &ly.k_norm);
                self.rope.apply(k, pos);
                fake_quant8(k);
                fake_quant8(&mut vt[hd * vd..(hd + 1) * vd]);
            }
            st.k[l].push(Arc::from(kt));
            st.v[l].push(Arc::from(vt));
            let (ks, vs, len) = (&st.k[l], &st.v[l], pos + 1);
            for hd in 0..h_n {
                let kh = hd / rep;
                let q = &qv[hd * qk..(hd + 1) * qk];
                let (mut m, mut sum) = (f64::NEG_INFINITY, 0f64);
                for s in 0..len {
                    let k = &ks[s][kh * qk..(kh + 1) * qk];
                    let mut dot = 0f64;
                    for d in 0..qk {
                        dot += q[d] as f64 * k[d] as f64;
                    }
                    logit[s] = dot * scale;
                    if logit[s] > m {
                        m = logit[s];
                    }
                }
                for s in 0..len {
                    logit[s] = (logit[s] - m).exp();
                    sum += logit[s];
                }
                let o = &mut att[hd * vd..(hd + 1) * vd];
                o.fill(0.0);
                for s in 0..len {
                    let wgt = logit[s] / sum;
                    let v = &vs[s][kh * vd..(kh + 1) * vd];
                    for d in 0..vd {
                        o[d] = madd_d(o[d], wgt, v[d]);
                    }
                }
            }
            for c in 0..od {
                att[c] = (att[c] as f64 * sigmoid(gt[c] as f64)) as f32;
            }
            quant.set(&att);
            ly.out_proj.apply(quant, &mut y);
            rms_norm_in_place(&mut y, &ly.post_attn);
            for c in 0..w {
                x[c] = madd_d(x[c], ly.attn_gate, y[c]);
            }
            // gated cross attention over the clip
            rms_norm(&x, Some(&ly.cross_norm), &mut tmp);
            quant.set(&tmp);
            ly.cq.apply(quant, &mut qv);
            ly.c_gate.apply(quant, &mut gt);
            let (ck, cv) = &cross.layers[l];
            for hd in 0..h_n {
                let q = &mut qv[hd * qk..(hd + 1) * qk];
                rms_norm_in_place(q, &ly.cq_norm);
                fake_quant8(q);
                let (mut m, mut sum) = (f64::NEG_INFINITY, 0f64);
                for s in 0..t_n {
                    let k = &ck[s * qd + hd * qk..][..qk];
                    let mut dot = 0f64;
                    for d in 0..qk {
                        dot += q[d] as f64 * k[d] as f64;
                    }
                    logit[s] = dot * scale;
                    if logit[s] > m {
                        m = logit[s];
                    }
                }
                for s in 0..t_n {
                    logit[s] = (logit[s] - m).exp();
                    sum += logit[s];
                }
                let o = &mut att[hd * vd..(hd + 1) * vd];
                o.fill(0.0);
                for s in 0..t_n {
                    let wgt = logit[s] / sum;
                    let v = &cv[s * od + hd * vd..][..vd];
                    for d in 0..vd {
                        o[d] = madd_d(o[d], wgt, v[d]);
                    }
                }
            }
            for c in 0..od {
                att[c] = (att[c] as f64 * sigmoid(gt[c] as f64)) as f32;
            }
            quant.set(&att);
            ly.c_out.apply(quant, &mut y);
            rms_norm_in_place(&mut y, &ly.post_cross);
            for c in 0..w {
                x[c] = madd_d(x[c], ly.cross_gate, y[c]);
            }
            // Hadamard MLP
            rms_norm(&x, Some(&ly.pre_hada), &mut tmp);
            ly.hada.run(&tmp, &mut y);
            for c in 0..w {
                x[c] = (x[c] as f64 + y[c] as f64) as f32;
            }
            self.mhc.finish(&mut stream, &x, &u, &mst);
        }
        for c in 0..w {
            let mut a = 0f64;
            for i in 0..n {
                a += stream[i * w + c] as f64;
            }
            tmp[c] = (a / n as f64) as f32;
        }
        rms_norm_in_place(&mut tmp, &self.final_norm);
        quant.set(&tmp);
        let mut out = vec![0f32; self.embedding.out];
        self.embedding.apply(quant, &mut out);
        out
    }
}
