//! `.cact` archive reader.

use std::sync::Arc;

use crate::{err, Result};

pub(crate) const TAG: u32 = 0x05e1_2a84;
pub(crate) const FP16: u8 = 1;
pub(crate) const FP32: u8 = 2;
pub(crate) const CQ: u8 = 3;
pub(crate) const RAW: u8 = 4;

const HEADER_BYTES: usize = 196;
const DIR_ENTRY_BYTES: usize = 44;

#[derive(Debug, Clone)]
pub struct Header {
    pub kv_window: usize,
    pub kv_bits: usize,
    pub vocab: usize,
    pub out_vocab: usize,
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub layers: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub max_seq: usize,
    pub hada_n: usize,
    pub lanes: usize,
    pub sliding_window: usize,
    pub taps: usize,
    pub engram_slots: usize,
    pub engram_sub_dim: usize,
    pub engram_tables: usize,
    pub engram_taps: usize,
    pub engram_dilation: usize,
    pub engram_seed_heads: usize,
    pub engram_orders: Vec<usize>,
    pub engram_sites: Vec<usize>,
    pub rope_theta: f32,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub index: usize,
    pub dtype: u8,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
    pub group: usize,
    pub bits: usize,
}

pub struct Cact {
    pub(crate) data: Arc<Vec<u8>>,
    pub header: Header,
    pub tensors: Vec<TensorInfo>,
    codebook: Vec<f32>,
}

fn u16_at(d: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([d[o], d[o + 1]])
}

fn u32_at(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

fn f32_at(d: &[u8], o: usize) -> f32 {
    f32::from_bits(u32_at(d, o))
}

pub(crate) fn half_to_f32(h: u16) -> f32 {
    let (s, e, f) = (h >> 15, (h >> 10) & 31, (h & 1023) as f32);
    let v = if e == 0 {
        f * 2f32.powi(-24)
    } else if e == 31 {
        if f != 0.0 { f32::NAN } else { f32::INFINITY }
    } else {
        (1.0 + f / 1024.0) * 2f32.powi(e as i32 - 15)
    };
    if s != 0 { -v } else { v }
}

impl Cact {
    pub fn new(bytes: Vec<u8>) -> Result<Cact> {
        let d = &bytes[..];
        if d.len() < HEADER_BYTES {
            return err("short .cact file");
        }
        if u32_at(d, 0) != TAG {
            return err(format!("not a Needle 3 / Whistle .cact archive (tag 0x{:x})", u32_at(d, 0)));
        }
        let u = |i: usize| u32_at(d, i * 4) as usize;
        let num_tensors = u(1);
        let codebook_len = u(2);
        // The order and site lists have four header words each.
        let header = Header {
            kv_window: u(3),
            kv_bits: u(4),
            vocab: u(5),
            out_vocab: u(6),
            width: u(7),
            heads: u(8),
            kv_heads: u(9),
            layers: u(10),
            qk_dim: u(11),
            v_dim: u(12),
            max_seq: u(13),
            hada_n: u(14),
            lanes: u(15),
            sliding_window: u(16),
            taps: u(19),
            engram_slots: u(20),
            engram_sub_dim: u(21),
            engram_tables: u(22),
            engram_taps: u(23),
            engram_dilation: u(24),
            engram_seed_heads: u(25),
            engram_orders: (0..u(26).min(4)).map(|i| u(27 + i)).collect(),
            engram_sites: (0..u(31).min(4)).map(|i| u(32 + i)).collect(),
            rope_theta: f32_at(d, 192),
        };

        let dir = HEADER_BYTES + codebook_len * 4;
        if dir + num_tensors * DIR_ENTRY_BYTES > d.len() {
            return err("truncated .cact directory");
        }
        let codebook = (0..codebook_len).map(|i| f32_at(d, HEADER_BYTES + i * 4)).collect();

        let mut tensors = Vec::with_capacity(num_tensors);
        for i in 0..num_tensors {
            let o = dir + i * DIR_ENTRY_BYTES;
            let ndim = (d[o + 1] as usize).min(4);
            if u32_at(d, o + 24) != 0 || u32_at(d, o + 32) != 0 {
                return err("tensor offset beyond 4 GiB");
            }
            let t = TensorInfo {
                index: i,
                dtype: d[o],
                shape: (0..ndim).map(|k| u32_at(d, o + 4 + k * 4) as usize).collect(),
                offset: u32_at(d, o + 20) as usize,
                nbytes: u32_at(d, o + 28) as usize,
                group: u32_at(d, o + 36) as usize,
                bits: u32_at(d, o + 40) as usize,
            };
            if t.offset + t.nbytes > d.len() {
                return err(format!("tensor #{i} out of bounds"));
            }
            tensors.push(t);
        }
        Ok(Cact { data: Arc::new(bytes), header, tensors, codebook })
    }

    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn tensor(&self, i: usize) -> Result<&TensorInfo> {
        match self.tensors.get(i) {
            Some(t) => Ok(t),
            None => err(format!("tensor #{i} is missing ({} in the archive)", self.tensors.len())),
        }
    }

    /// Centroids of the CQ codebook for `bits`-wide indices.
    pub(crate) fn codebook(&self, bits: usize) -> Result<&[f32]> {
        let range = match bits {
            2 => 0..4,
            3 => 4..12,
            4 => 12..28,
            _ => return err(format!("unsupported CQ width {bits}")),
        };
        match self.codebook.get(range) {
            Some(cb) => Ok(cb),
            None => err(format!("the archive holds no CQ{bits} codebook")),
        }
    }

    pub(crate) fn fp16(&self, i: usize) -> Result<Vec<f32>> {
        let t = self.tensor(i)?;
        if t.dtype != FP16 {
            return err(format!("tensor #{i} is not FP16"));
        }
        let d = &self.data[t.offset..t.offset + t.nbytes];
        Ok(d.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect())
    }

    pub(crate) fn fp32(&self, i: usize) -> Result<Vec<f32>> {
        let t = self.tensor(i)?;
        if t.dtype != FP32 {
            return err(format!("tensor #{i} is not FP32"));
        }
        let d = &self.data[t.offset..t.offset + t.nbytes];
        Ok(d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// The first value of an FP16 tensor (the scalar gates).
    pub(crate) fn fp16_scalar(&self, i: usize) -> Result<f32> {
        match self.fp16(i)?.first() {
            Some(&v) => Ok(v),
            None => err(format!("tensor #{i} is empty")),
        }
    }

    pub(crate) fn raw(&self, i: usize) -> Result<&[u8]> {
        let t = self.tensor(i)?;
        Ok(&self.data[t.offset..t.offset + t.nbytes])
    }

    pub(crate) fn half_at(&self, offset: usize) -> f32 {
        half_to_f32(u16_at(&self.data, offset))
    }
}
