//! Regression test: cargo test
//!
//! The expected transcripts and the encoder outputs in tests/data were produced by
//! the official engine (needle.wasm from Cactus-Compute/needle3) on the same clips.
//!
//! The model is not part of the crate: put `whistle.cact` in the crate folder (or
//! the one above it), or point WHISTLE_MODEL at it.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use whistle::{audio, Model, Options};

/// A file in the crate folder, in tests/data, or in the folder above the crate.
fn find(file: &str) -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    [root.join(file), root.join("tests/data").join(file), root.join("..").join(file)]
        .into_iter()
        .find(|p| p.is_file())
}

fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        let path = std::env::var_os("WHISTLE_MODEL").map(PathBuf::from).or_else(|| find("whistle.cact"));
        let path = path.expect("whistle.cact not found: put it in the crate folder or set WHISTLE_MODEL");
        Model::open(path).unwrap()
    })
}

fn clip(file: &str) -> Option<Vec<f32>> {
    let Some(path) = find(file) else {
        eprintln!("skip {file}: not in this checkout");
        return None;
    };
    let (samples, rate) = audio::read_wav_file(&path).unwrap();
    Some(audio::resample(&samples, rate, 16000))
}

fn f32_file(file: &str) -> Vec<f32> {
    let b = std::fs::read(find(file).unwrap_or_else(|| panic!("{file} is missing"))).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut x, mut y) = (0f64, 0f64, 0f64);
    for (&p, &q) in a.iter().zip(b) {
        d += p as f64 * q as f64;
        x += p as f64 * p as f64;
        y += q as f64 * q as f64;
    }
    d / (x * y).sqrt()
}

fn check(file: &str, text: &str, language: &str, embed: Option<&str>) {
    let Some(pcm) = clip(file) else { return };
    let model = model();
    if let Some(embed) = embed {
        let (e, reference) = (model.embed(&pcm), f32_file(embed));
        assert_eq!(e.data.len(), reference.len(), "{file}: encoder frame count");
        let cos = cosine(&e.data, &reference);
        println!("{file}: encoder cosine {cos:.6}");
        assert!(cos > 0.9995, "{file}: encoder output differs from the reference engine (cosine {cos:.6})");
    }
    let r = model.transcribe(&pcm, &Options::default()).unwrap();
    println!("{file}: {:?} ({}) in {:.0} ms", r.text, r.language, r.timings.total);
    assert_eq!(r.text, text, "{file}: transcript");
    assert_eq!(r.language, language, "{file}: language");
}

#[test]
fn short_recording() {
    check(
        "whistle_recording_16k.wav",
        "Hello, this is Hate Est.",
        "en",
        Some("whistle_recording_16k.embed.f32"),
    );
}

#[test]
fn jfk() {
    check(
        "jfk.wav",
        "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
        "en",
        Some("jfk.embed.f32"),
    );
}

#[test]
fn steady_tone_is_not_speech() {
    let Some(pcm) = clip("whistle_demo_1khz.wav") else { return };
    let r = model().transcribe(&pcm, &Options::default()).unwrap();
    assert!(r.silent && r.text.is_empty() && r.language.is_empty());
}

#[test]
fn forced_language_keeps_the_transcript() {
    let Some(pcm) = clip("jfk.wav") else { return };
    let opts = Options { language: Some("pl".into()), ..Options::default() };
    let r = model().transcribe(&pcm[..60000], &opts).unwrap();
    assert_eq!((r.language.as_str(), r.text.as_str()), ("pl", "And so my fellow Americans."));
}

#[test]
fn unknown_language_is_an_error() {
    let Some(pcm) = clip("jfk.wav") else { return };
    let opts = Options { language: Some("xx".into()), ..Options::default() };
    assert!(model().transcribe(&pcm[..60000], &opts).is_err());
}
