//! The model: archive layout, beam search, transcription.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::cact::{Cact, Header, FP32, RAW};
use crate::decoder::{Decoder, Layout, State};
use crate::encoder::{Encoder, TENSORS_PER_LAYER};
use crate::frontend::{Config, FrontEnd};
use crate::layers::{Perms, Stem};
use crate::quant::Quantizer;
use crate::tokenizer::Tokenizer;
use crate::{err, Error, Result};

pub const LANGUAGES: [&str; 7] = ["en", "de", "fr", "es", "it", "nl", "pl"];
const MAX_AUDIO_SAMPLES: usize = 480_000; // one pass covers 30 s
const SILENCE_RANGE_DB: f64 = 5.0;
const QUANT_GROUP: usize = 128;

#[derive(Debug, Clone)]
pub struct Options {
    /// "en", "de", "fr", "es", "it", "nl" or "pl"; None detects it.
    pub language: Option<String>,
    /// Beam width; 5 is the engine's, 1 is greedy.
    pub beams: usize,
}

impl Default for Options {
    fn default() -> Options {
        Options { language: None, beams: 5 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    FrontEnd,
    Stem,
    Encoder,
    Decoder,
}

/// Receives progress while a clip is transcribed. Every method defaults to nothing.
pub trait Observer {
    fn progress(&mut self, _stage: Stage, _fraction: f32) {}
    /// Called after every decoding step with the best text so far.
    fn step(&mut self, _step: usize, _partial: &str) {}
    /// Audio over 30 s only: called after each window with the text so far.
    fn window(&mut self, _text: &str, _fraction: f32) {}
}

impl Observer for () {}

/// Milliseconds per stage; `frontend` includes the stem.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub frontend: f64,
    pub encoder: f64,
    pub decoder: f64,
    pub total: f64,
    pub steps: usize,
}

#[derive(Debug, Clone)]
pub struct Transcript {
    pub text: String,
    pub language: String,
    pub tokens: Vec<u32>,
    /// The clip was silence or steady noise and was not decoded.
    pub silent: bool,
    pub frames: usize,
    pub score: f64,
    pub timings: Timings,
}

/// The encoder output: one row of `width` floats per 80 ms frame.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub data: Vec<f32>,
    pub frames: usize,
    pub width: usize,
}

pub struct Model {
    cact: Cact,
    cfg: Config,
    front_end: FrontEnd,
    stem: Stem,
    encoder: Encoder,
    decoder: Decoder,
    tokenizer: Tokenizer,
    languages: Vec<&'static str>,
}

struct Beam {
    state: State,
    tokens: Vec<u32>,
    score: f64,
}

struct Decoded {
    tokens: Vec<u32>,
    language: String,
    score: f64,
    steps: usize,
}

/// If the tail of `t` repeats with a period of 1..8 tokens four times over, returns
/// the length that keeps a single copy of the loop.
fn repetition_cut(t: &[u32]) -> Option<usize> {
    let n = t.len();
    for p in 1..=8 {
        if n < 4 * p {
            return None;
        }
        if (n - 4 * p..n - p).all(|c| t[c] == t[c + p]) {
            return Some(n - 3 * p);
        }
    }
    None
}

fn ms(from: Instant, to: Instant) -> f64 {
    to.duration_since(from).as_secs_f64() * 1000.0
}

impl Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Model> {
        let path = path.as_ref();
        match std::fs::read(path) {
            Ok(bytes) => Model::new(bytes),
            Err(e) => err(format!("{}: {e}", path.display())),
        }
    }

    /// bytes: the contents of `whistle.cact`.
    pub fn new(bytes: Vec<u8>) -> Result<Model> {
        let cact = Cact::new(bytes)?;
        let hdr = cact.header.clone();
        let layers = hdr.layers;
        let per_layer = 20 + if hdr.taps > 0 { 7 } else { 4 };
        let mut p = 1 + per_layer * layers;
        let dec_mhc = p;
        p += 9;
        let perms_at = p;
        p += 2;
        let engram0 = p;
        p += 4 * hdr.engram_sites.len();
        let final_norm = p;
        let manifest = p + 1;
        p += 2;
        let is_manifest = |i: usize| cact.tensors.get(i).is_some_and(|t| t.dtype == FP32 && t.shape.len() == 1);
        if !is_manifest(manifest) {
            return err("this .cact holds no Whistle audio manifest (is it a Needle text model?)");
        }
        let m = cact.fp32(manifest)?;
        if m.len() < 13 {
            return err("the audio manifest is too short");
        }
        let at = |i: usize| m[i] as usize;
        let cfg = Config {
            text_vocab: at(2),
            enc_layers: at(3),
            enc_lanes: at(4),
            conv_kernel: at(5),
            stem_width: at(6),
            stem_stages: at(7),
            mel_bins: at(8),
            sample_rate: at(9),
            n_fft: at(10),
            window_samples: at(11),
            hop_samples: at(12),
        };
        let cross0 = p;
        p += 10 * layers;
        let pe_gate = p;
        p += 1;
        let enc0 = p;
        p += TENSORS_PER_LAYER * cfg.enc_layers + 9 + 1; // layers, mHC, final norm
        let stem_at = p;
        p += 2 * cfg.stem_stages;
        let mel = p;
        let tokenizer_at = p + 1;
        p += 2;
        if p != cact.num_tensors() || cact.tensor(tokenizer_at)?.dtype != RAW {
            return err(format!("unexpected tensor layout: {} tensors, expected {p}", cact.num_tensors()));
        }
        if !cfg.n_fft.is_power_of_two() || cfg.hop_samples == 0 || cfg.window_samples > cfg.n_fft {
            return err("unsupported front end parameters in the audio manifest");
        }
        if hdr.kv_heads == 0 || hdr.heads % hdr.kv_heads != 0 || cfg.text_vocab > hdr.vocab {
            return err("unsupported attention or vocabulary sizes in the header");
        }

        let to_perm = |i: usize| -> Result<Vec<usize>> { Ok(cact.fp32(i)?.iter().map(|&v| v as usize).collect()) };
        let perms: Perms = Arc::new([to_perm(perms_at)?, to_perm(perms_at + 1)?]);
        let idx = Layout {
            embedding: 0,
            dec_layer0: 1,
            dec_per_layer: per_layer,
            dec_mhc,
            engram0,
            final_norm,
            cross0,
            pe_gate,
        };
        let mel_fb = cact.fp32(mel)?;
        if mel_fb.len() != ((cfg.n_fft >> 1) + 1) * cfg.mel_bins {
            return err("the mel filterbank does not match the front end parameters");
        }
        let n_lang = (hdr.vocab - cfg.text_vocab).min(LANGUAGES.len());
        Ok(Model {
            front_end: FrontEnd::new(&cfg, &mel_fb),
            stem: Stem::new(&cact, stem_at, &cfg)?,
            encoder: Encoder::new(&cact, enc0, &cfg, &hdr, &perms)?,
            decoder: Decoder::new(&cact, &idx, &hdr, &perms)?,
            tokenizer: Tokenizer::new(cact.raw(tokenizer_at)?)?,
            languages: LANGUAGES[..n_lang].to_vec(),
            cfg,
            cact,
        })
    }

    pub fn header(&self) -> &Header {
        &self.cact.header
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn num_tensors(&self) -> usize {
        self.cact.num_tensors()
    }

    /// The languages this archive can be forced to.
    pub fn languages(&self) -> &[&'static str] {
        &self.languages
    }

    /// The loudness range (dB) between the 99th and 10th percentile frame; below
    /// SILENCE_RANGE_DB the clip is silence or steady noise.
    fn loudness_range(db: &[f32]) -> f64 {
        if db.len() < 2 {
            return 0.0;
        }
        let mut s = db.to_vec();
        s.sort_by(f32::total_cmp);
        let f = (db.len() - 1) as f32;
        s[(f * 0.99f32).floor() as usize] as f64 - s[(f * 0.1f32).floor() as usize] as f64
    }

    /// Beam search over one encoded clip.
    fn decode(
        &self,
        enc: &[f32],
        frames: usize,
        opts: &Options,
        quant: &mut Quantizer,
        obs: &mut dyn Observer,
    ) -> Result<Decoded> {
        let (dec, tk, text_vocab) = (&self.decoder, &self.tokenizer, self.cfg.text_vocab);
        let beam_width = opts.beams.max(1);
        let per_beam = beam_width + 1;
        let max_len = self.cact.header.max_seq.saturating_sub(1);
        let cross = dec.prepare_cross(enc, frames, quant);
        let mut st0 = dec.new_state();
        let lg = dec.step(&mut st0, tk.bos_id, &cross, quant);
        let lang = match &opts.language {
            Some(name) => match self.languages.iter().position(|l| l == name) {
                Some(i) => i,
                None => return err(format!("unknown language \"{name}\"")),
            },
            None => {
                let mut best = 0;
                for i in text_vocab + 1..lg.len() {
                    if lg[i] > lg[text_vocab + best] {
                        best = i - text_vocab;
                    }
                }
                best
            }
        };
        let lang_token = (text_vocab + lang) as u32;

        let mut beams = vec![Beam { state: st0, tokens: Vec::new(), score: 0.0 }];
        let mut best: Option<Vec<u32>> = None;
        let (mut best_score, mut finished, mut steps) = (-1e30f64, 0usize, 0usize);
        while !beams.is_empty() && finished < beam_width {
            // (score, beam, token)
            let mut cands: Vec<(f64, usize, u32)> = Vec::with_capacity(beams.len() * per_beam);
            for (b, bm) in beams.iter_mut().enumerate() {
                let input = bm.tokens.last().copied().unwrap_or(lang_token);
                let lg = dec.step(&mut bm.state, input, &cross, quant);
                let lg = &lg[..text_vocab];
                let mx = lg.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
                let mut sum = 0f64;
                for &v in lg {
                    sum += (v as f64 - mx).exp();
                }
                let base = bm.score - (sum.ln() + mx);
                // the per_beam best continuations of this beam
                let mut top: Vec<(f32, u32)> = Vec::with_capacity(per_beam + 1);
                for (i, &v) in lg.iter().enumerate() {
                    if top.len() == per_beam {
                        if v <= top[per_beam - 1].0 {
                            continue;
                        }
                        top.pop();
                    }
                    let k = top.iter().position(|t| t.0 < v).unwrap_or(top.len());
                    top.insert(k, (v, i as u32));
                }
                cands.extend(top.iter().map(|&(v, token)| (base + v as f64, b, token)));
            }
            cands.sort_by(|x, y| y.0.total_cmp(&x.0));
            // (parent beam, tokens, score)
            let mut next: Vec<(usize, Vec<u32>, f64)> = Vec::with_capacity(beam_width);
            for &(score, b, token) in &cands {
                if next.len() >= beam_width {
                    break;
                }
                let mut toks = beams[b].tokens.clone();
                let mut cut = None;
                let mut done = token == tk.eos_id;
                if !done {
                    toks.push(token);
                    cut = repetition_cut(&toks);
                    done = cut.is_some() || toks.len() >= max_len;
                }
                if !done {
                    next.push((b, toks, score));
                    continue;
                }
                if finished == beam_width {
                    continue;
                }
                let norm = score / toks.len().max(1) as f64;
                if !(norm <= best_score) {
                    if let Some(keep) = cut {
                        toks.truncate(keep);
                    }
                    best = Some(toks);
                    best_score = norm;
                }
                finished += 1;
            }
            // every child but the last works on its own copy of the parent's caches
            let mut children = vec![0usize; beams.len()];
            for (parent, _, _) in &next {
                children[*parent] += 1;
            }
            let mut parents: Vec<Option<State>> = beams.into_iter().map(|b| Some(b.state)).collect();
            beams = Vec::with_capacity(next.len());
            for (parent, tokens, score) in next {
                children[parent] -= 1;
                let state = if children[parent] == 0 { parents[parent].take() } else { parents[parent].clone() };
                let state = state.expect("a parent state is taken only by its last child");
                beams.push(Beam { state, tokens, score });
            }
            steps += 1;
            let partial = match (&best, beams.first()) {
                (Some(b), _) => tk.decode(b),
                (None, Some(b)) => tk.decode(&b.tokens),
                (None, None) => String::new(),
            };
            obs.step(steps, &partial);
        }
        Ok(Decoded {
            tokens: best.unwrap_or_default(),
            language: self.languages.get(lang).copied().unwrap_or("").to_string(),
            score: best_score,
            steps,
        })
    }

    /// Speech to text for 16 kHz mono float PCM in [-1, 1].
    pub fn transcribe(&self, pcm: &[f32], opts: &Options) -> Result<Transcript> {
        self.transcribe_observed(pcm, opts, &mut ())
    }

    pub fn transcribe_observed(&self, pcm: &[f32], opts: &Options, obs: &mut dyn Observer) -> Result<Transcript> {
        if pcm.len() > MAX_AUDIO_SAMPLES {
            return self.transcribe_long(pcm, opts, obs);
        }
        let t0 = Instant::now();
        let mut quant = Quantizer::new(QUANT_GROUP);
        obs.progress(Stage::FrontEnd, 0.0);
        let fe = self.front_end.run(pcm);
        if Self::loudness_range(&fe.db) < SILENCE_RANGE_DB {
            return Ok(Transcript {
                text: String::new(),
                language: String::new(),
                tokens: Vec::new(),
                silent: true,
                frames: 0,
                score: 0.0,
                timings: Timings { total: ms(t0, Instant::now()), ..Timings::default() },
            });
        }
        obs.progress(Stage::Stem, 0.0);
        let (stem, frames) = self.stem.run(&fe.feat, fe.frames, &mut quant);
        let t1 = Instant::now();
        let enc = self.encoder.run(&stem, frames, &mut quant, &mut |f| obs.progress(Stage::Encoder, f));
        let t2 = Instant::now();
        obs.progress(Stage::Decoder, 0.0);
        let r = self.decode(&enc, frames, opts, &mut quant, obs)?;
        let t3 = Instant::now();
        Ok(Transcript {
            text: self.tokenizer.decode(&r.tokens),
            language: r.language,
            tokens: r.tokens,
            silent: false,
            frames,
            score: r.score,
            timings: Timings {
                frontend: ms(t0, t1),
                encoder: ms(t1, t2),
                decoder: ms(t2, t3),
                total: ms(t0, t3),
                steps: r.steps,
            },
        })
    }

    /// Longer audio: consecutive windows of up to 30 s, each transcribed on its own.
    /// A window that is not the last one ends at the quietest 100 ms of its final five
    /// seconds, so the cut falls in a pause rather than inside a word.
    fn transcribe_long(&self, pcm: &[f32], opts: &Options, obs: &mut dyn Observer) -> Result<Transcript> {
        let mut opts = opts.clone();
        let mut parts: Vec<String> = Vec::new();
        let mut tokens = Vec::new();
        let mut timings = Timings::default();
        let (mut frames, mut off) = (0, 0);
        let search = 5 * self.cfg.sample_rate;
        let block = (self.cfg.sample_rate / 10).max(2);
        while off < pcm.len() {
            let mut len = MAX_AUDIO_SAMPLES.min(pcm.len() - off);
            if off + len < pcm.len() {
                let (mut best_e, mut best_at) = (f64::INFINITY, len);
                let mut s = len.saturating_sub(search);
                while s + block <= len {
                    let e: f64 = pcm[off + s..off + s + block].iter().map(|&x| x as f64 * x as f64).sum();
                    if e < best_e {
                        best_e = e;
                        best_at = s + (block >> 1);
                    }
                    s += block >> 1;
                }
                len = best_at;
            }
            let r = self.transcribe_observed(&pcm[off..off + len], &opts, obs)?;
            if !r.text.is_empty() {
                parts.push(r.text);
            }
            if opts.language.is_none() && !r.language.is_empty() {
                opts.language = Some(r.language);
            }
            tokens.extend_from_slice(&r.tokens);
            frames += r.frames;
            timings.frontend += r.timings.frontend;
            timings.encoder += r.timings.encoder;
            timings.decoder += r.timings.decoder;
            timings.total += r.timings.total;
            timings.steps += r.timings.steps;
            off += len;
            obs.window(&parts.join(" "), off as f32 / pcm.len() as f32);
        }
        Ok(Transcript {
            text: parts.join(" "),
            language: opts.language.unwrap_or_default(),
            tokens,
            silent: parts.is_empty(),
            frames,
            score: 0.0,
            timings,
        })
    }

    /// The encoder output for a clip of up to 30 s.
    pub fn embed(&self, pcm: &[f32]) -> Embedding {
        let mut quant = Quantizer::new(QUANT_GROUP);
        let fe = self.front_end.run(pcm);
        let (stem, frames) = self.stem.run(&fe.feat, fe.frames, &mut quant);
        let data = self.encoder.run(&stem, frames, &mut quant, &mut |_| {});
        Embedding { data, frames, width: self.cact.header.width }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error(e.to_string())
    }
}
