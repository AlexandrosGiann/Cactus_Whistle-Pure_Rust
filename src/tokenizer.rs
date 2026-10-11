//! The SentencePiece BPE dump attached to the archive (decode side only).

use crate::{err, Result};

const TK_UNKNOWN: u8 = 1;
const TK_CONTROL: u8 = 2;
const TK_BYTE: u8 = 4;

pub struct Tokenizer {
    pub pad_id: u32,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    add_dummy_prefix: bool,
    bytes: Vec<Vec<u8>>,
    types: Vec<u8>,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl Tokenizer {
    pub fn new(blob: &[u8]) -> Result<Tokenizer> {
        if blob.len() < 24 {
            return err("short tokenizer blob");
        }
        let n = u32_at(blob, 0) as usize;
        let mut off = 24;
        let (mut bytes, mut types) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            // each entry: 4 bytes of score, type, piece length, piece
            if off + 7 > blob.len() {
                return err(format!("tokenizer blob ends at piece {i} of {n}"));
            }
            let ty = blob[off + 4];
            let len = u16::from_le_bytes([blob[off + 5], blob[off + 6]]) as usize;
            off += 7;
            if off + len > blob.len() {
                return err(format!("tokenizer blob ends at piece {i} of {n}"));
            }
            let piece = &blob[off..off + len];
            off += len;
            types.push(ty);
            if ty == TK_BYTE {
                // written as <0xNN>
                let hex = piece.get(3..5).and_then(|h| std::str::from_utf8(h).ok());
                let byte = hex.and_then(|h| u8::from_str_radix(h, 16).ok());
                bytes.push(byte.map(|b| vec![b]).unwrap_or_default());
            } else {
                bytes.push(piece.to_vec());
            }
        }
        Ok(Tokenizer {
            pad_id: u32_at(blob, 4),
            eos_id: u32_at(blob, 8),
            bos_id: u32_at(blob, 12),
            unk_id: u32_at(blob, 16),
            add_dummy_prefix: blob[20] != 0,
            bytes,
            types,
        })
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut buf = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id >= self.types.len() || self.types[id] == TK_CONTROL || self.types[id] == TK_UNKNOWN {
                continue;
            }
            buf.extend_from_slice(&self.bytes[id]);
        }
        let text = String::from_utf8_lossy(&buf).replace('\u{2581}', " ");
        match text.strip_prefix(' ') {
            Some(rest) if self.add_dummy_prefix => rest.to_string(),
            _ => text,
        }
    }
}
