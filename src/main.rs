//! Command line front end:
//! whistle clip.wav [more] [--language en] [--greedy] [--beams n] [--json] [--model whistle.cact]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use whistle::{audio, Model, Options};

const USAGE: &str = "usage: whistle clip.wav [more] [--language en|de|fr|es|it|nl|pl] [--greedy] [--beams n] \
[--json] [--model whistle.cact]";

/// `whistle.cact` in the working directory, its parent, or beside the executable.
fn default_model() -> PathBuf {
    let mut places = vec![PathBuf::from("whistle.cact"), PathBuf::from("../whistle.cact")];
    if let Ok(exe) = std::env::current_exe() {
        places.extend(exe.ancestors().skip(1).take(4).map(|dir| dir.join("whistle.cact")));
    }
    places.iter().find(|p| p.is_file()).cloned().unwrap_or_else(|| PathBuf::from("whistle.cact"))
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn run() -> Result<ExitCode, whistle::Error> {
    let mut opts = Options::default();
    let (mut model_path, mut json, mut help, mut files) = (None, false, false, Vec::new());
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| whistle::Error(format!("{name} needs a value")));
        match a.as_str() {
            "--language" | "-l" => opts.language = Some(value("--language")?),
            "--model" | "-m" => model_path = Some(PathBuf::from(value("--model")?)),
            "--greedy" => opts.beams = 1,
            "--beams" => {
                opts.beams = match value("--beams")?.parse() {
                    Ok(n) if n > 0 => n,
                    _ => return Err(whistle::Error("--beams needs a positive number".into())),
                }
            }
            "--json" => json = true,
            "--help" | "-h" => help = true,
            _ => files.push(a),
        }
    }
    if help || files.is_empty() {
        println!("{USAGE}");
        return Ok(if help { ExitCode::SUCCESS } else { ExitCode::FAILURE });
    }
    let model = Model::open(model_path.unwrap_or_else(default_model))?;
    for file in &files {
        let (samples, rate) = audio::read_wav_file(file)?;
        let pcm = audio::resample(&samples, rate, 16000);
        let seconds = pcm.len() as f64 / 16000.0;
        let r = model.transcribe(&pcm, &opts)?;
        if json {
            println!(
                "{{\"file\":{},\"text\":{},\"language\":{},\"seconds\":{},\"ms\":{}}}",
                json_string(file),
                json_string(&r.text),
                json_string(&r.language),
                seconds,
                r.timings.total.round()
            );
        } else {
            println!("{}", if r.silent { "(no speech)" } else { &r.text });
            let name = Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned());
            eprintln!(
                "  {}: {:.2} s of audio, {}, {:.2} s",
                name.unwrap_or_else(|| file.clone()),
                seconds,
                if r.language.is_empty() { "no language" } else { &r.language },
                r.timings.total / 1000.0
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("whistle: {e}");
            ExitCode::from(2)
        }
    }
}
