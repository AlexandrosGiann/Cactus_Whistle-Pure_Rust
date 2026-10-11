//! Whistle — pure Rust runtime for Cactus Compute's Whistle speech-to-text model.
//!
//! No dependencies beyond `std`, no C, no WASM. Reads the official `whistle.cact`
//! archive (tag 0x05E12A84) and runs the whole pipeline: log-mel front end,
//! convolutional stem, 8-block audio encoder, 8-block decoder with gated cross
//! attention and engram lookups, beam search and the SentencePiece detokeniser.
//!
//! The numerics follow the reference engine: CQ weights are consumed as int8
//! centroids, activations are Hadamard-rotated per group of 128 and quantised
//! to int8 before every matmul. The arithmetic mirrors `whistle.js` operation
//! for operation (f64 intermediates, f32 stores), so the two ports agree far
//! more closely than either agrees with the engine.
//!
//! ```no_run
//! let model = whistle::Model::open("whistle.cact")?;
//! let (samples, rate) = whistle::audio::read_wav_file("clip.wav")?;
//! let pcm = whistle::audio::resample(&samples, rate, 16000);
//! let result = model.transcribe(&pcm, &whistle::Options::default())?;
//! println!("{} ({})", result.text, result.language);
//! # Ok::<(), whistle::Error>(())
//! ```

pub mod audio;
mod cact;
mod decoder;
mod encoder;
mod frontend;
mod layers;
mod model;
mod quant;
mod tokenizer;

pub use cact::{Cact, Header, TensorInfo};
pub use frontend::Config;
pub use model::{Embedding, Model, Observer, Options, Stage, Timings, Transcript, LANGUAGES};
pub use tokenizer::Tokenizer;

use std::fmt;

#[derive(Debug, Clone)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error(msg.into()))
}
