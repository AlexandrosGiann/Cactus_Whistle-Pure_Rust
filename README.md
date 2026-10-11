# Whistle in pure Rust

A dependency-free Rust runtime for [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle),
the 16.9 MB speech-to-text model. Only `std`: no C, no WASM, no crates. It reads the official
`whistle.cact` archive and runs the whole pipeline, as a library and as a command line tool.

It is a port of the pure JavaScript runtime `whistle.js`, and matches it and the official engine on the
test clips (see [Verification](#verification)).

The model is not in this repository. Download `whistle.cact` from
<https://huggingface.co/Cactus-Compute/whistle> and put it in this folder.

## Build and run

Needs a Rust toolchain (<https://rustup.rs>; written against Rust 1.99, should build on 1.80 or newer).

```bash
cargo build --release
```

```bash
target/release/whistle tests/data/jfk.wav
```

Options: `--language en|de|fr|es|it|nl|pl` (default: detect), `--greedy` (one beam, faster),
`--beams n`, `--json`, `--model path/to/whistle.cact`. Without `--model`, `whistle.cact` is looked for
in the working directory, its parent, and the folders above the executable.

Input is WAV (PCM 8/16/24/32-bit or float, any channel count and sample rate; it is mixed to mono and
resampled to 16 kHz). AMR is not read here: the JavaScript version does that through a bundled
third-party decoder.

## Use it from code

```rust
let model = whistle::Model::open("whistle.cact")?;
let (samples, rate) = whistle::audio::read_wav_file("clip.wav")?;
let pcm = whistle::audio::resample(&samples, rate, 16000);   // 16 kHz mono, [-1, 1]
let result = model.transcribe(&pcm, &whistle::Options::default())?;
println!("{} ({})", result.text, result.language);
```

- `Options { language: Option<String>, beams: usize }`; `beams` defaults to 5 (the engine's), 1 is greedy.
- `transcribe` returns `Transcript { text, language, tokens, silent, frames, score, timings }`.
- `transcribe_observed(pcm, &opts, &mut observer)` reports progress through the `Observer` trait:
  `progress(stage, fraction)`, `step(step, partial_text)` after every decoding step, and
  `window(text, fraction)` for audio over 30 s.
- `model.embed(&pcm)` returns the encoder output, one row of 512 floats per 80 ms frame.

`Model` is `Send + Sync` and `transcribe` takes `&self`, so one loaded model can serve several threads.
Inference itself is single-threaded.

## Files

| file | role |
|---|---|
| `src/cact.rs` | `.cact` reader: header, codebooks, tensor directory, FP16 / FP32 / raw tensors |
| `src/quant.rs` | CQ matrices as int8 centroids, the Hadamard-rotated int8 activation quantiser |
| `src/frontend.rs` | loudness normalisation, STFT, mel filterbank, log, mean / variance normalisation |
| `src/layers.rs` | RMS norm, Hadamard MLP, multi-lane residual (mHC), rotary positions, stem |
| `src/encoder.rs` | the 8-block audio encoder |
| `src/decoder.rs` | the 8-block decoder: causal attention with taps, cross attention, engrams |
| `src/tokenizer.rs` | SentencePiece detokeniser |
| `src/model.rs` | archive layout, silence gate, beam search, long-audio windows |
| `src/audio.rs` | WAV reader and resampler |
| `src/main.rs` | the command line tool |
| `tests/reference.rs`, `tests/data` | regression test against the official engine's outputs |

## Verification

```bash
cargo test
```

checks transcripts, language detection, silence handling, a forced language and the encoder output
against the references in `tests/data`, which were produced by the official engine (`needle.wasm`).
It needs `whistle.cact` in this folder or the one above it, or its path in `WHISTLE_MODEL`. The test
on `whistle_recording_16k.wav` is skipped when that clip is not present.

Measured on this machine (one thread, release build):

| clip | whistle.js (Node 24) | whistle-rs | encoder cosine vs engine | transcript |
|---|---|---|---|---|
| `whistle_recording_16k.wav`, 3.0 s | 1.5 s | 0.77 s | 0.99994 | identical |
| `tests/data/jfk.wav`, 11.0 s | 5.5 s | 2.4 s | 0.99985 | identical |
| `jfk.wav` three times over, 34.5 s | 15.3 s | 8.1 s | not measured | identical to whistle.js |

As with the JavaScript port, the output is not bit-identical to the engine and cannot be: every matmul
input is rounded to int8, so a last-digit difference in float summation order flips a rounding now and
then. The arithmetic here follows `whistle.js` operation for operation (double intermediates, single
precision stores), which is why the two ports reach the same cosine against the engine.

## Differences from whistle.js

- Matmuls always multiply int8 weights with int8 activations directly; the per-activation lookup
  tables of the JavaScript version are not needed. The integer sums are exact either way, so the
  results are the same. Matrices are expanded to one byte per weight on first use (the engram tables,
  which are only read row by row, are not), so the process uses a few times the archive's size in RAM.
- No AMR input, no browser page, no microphone capture.

## Not implemented

Same list as the JavaScript port: word timestamps and per-word probabilities, keyword biasing, live
streaming, the engine's sentence-aware cut for audio over 30 s (longer clips are split at the quietest
100 ms in the last five seconds of each 30 s window), and the decoder ladder (`--audio-depth`).
