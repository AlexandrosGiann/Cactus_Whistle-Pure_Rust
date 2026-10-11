//! Audio front end: loudness normalisation, STFT, mel filterbank, log, CMVN.

use std::f64::consts::PI;

/// The audio side of the model, read from the archive's manifest tensor.
#[derive(Debug, Clone)]
pub struct Config {
    pub text_vocab: usize,
    pub enc_layers: usize,
    pub enc_lanes: usize,
    pub conv_kernel: usize,
    pub stem_width: usize,
    pub stem_stages: usize,
    pub mel_bins: usize,
    pub sample_rate: usize,
    pub n_fft: usize,
    pub window_samples: usize,
    pub hop_samples: usize,
}

pub(crate) struct FrontEnd {
    n_fft: usize,
    win: usize,
    hop: usize,
    mels: usize,
    bins: usize,
    window: Vec<f32>,
    fb: Vec<f32>, // mel-major, with each filter's bin range in lo..hi
    lo: Vec<usize>,
    hi: Vec<usize>,
    band_lo: usize,
    band_hi: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
    rev: Vec<usize>,
}

pub(crate) struct Features {
    pub feat: Vec<f32>, // [frames, mels]
    pub frames: usize,
    pub db: Vec<f32>, // loudness of the 250..3500 Hz band per frame
}

fn hann_symmetric(n: usize) -> Vec<f32> {
    (0..n).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos()) as f32).collect()
}

#[inline]
fn in_range(x: f32) -> bool {
    x > -65536.0 && x < 65536.0
}

impl FrontEnd {
    /// mel_filterbank is stored [bin][mel].
    pub fn new(cfg: &Config, mel_filterbank: &[f32]) -> FrontEnd {
        let (n, m) = (cfg.n_fft, cfg.mel_bins);
        let bins = (n >> 1) + 1;
        let mut fb = vec![0f32; m * bins];
        let (mut lo, mut hi) = (vec![0usize; m], vec![0usize; m]);
        for c in 0..m {
            let (mut l, mut h) = (bins, 0);
            for b in 0..bins {
                let v = mel_filterbank[b * m + c];
                fb[c * bins + b] = v;
                if v != 0.0 {
                    l = l.min(b);
                    h = b + 1;
                }
            }
            lo[c] = l;
            hi[c] = h;
        }
        // Mel channels whose whole filter lies inside 250..3500 Hz: the loudness band.
        let (mut band_lo, mut band_hi) = (m, 0);
        let hz = cfg.sample_rate as f64 / n as f64;
        for c in 0..m {
            if (lo[c] as f64) * hz < 250.0 || (hi[c] as f64) * hz > 3500.0 {
                continue;
            }
            band_lo = band_lo.min(c);
            band_hi = c + 1;
        }
        let bits = n.trailing_zeros();
        FrontEnd {
            n_fft: n,
            win: cfg.window_samples,
            hop: cfg.hop_samples,
            mels: m,
            bins,
            window: hann_symmetric(cfg.window_samples),
            fb,
            lo,
            hi,
            band_lo,
            band_hi,
            cos: (0..n >> 1).map(|b| (2.0 * PI * b as f64 / n as f64).cos()).collect(),
            sin: (0..n >> 1).map(|b| (2.0 * PI * b as f64 / n as f64).sin()).collect(),
            rev: (0..n).map(|b| if bits == 0 { 0 } else { b.reverse_bits() >> (usize::BITS - bits) }).collect(),
        }
    }

    /// Loudness normalisation gain: the 99th percentile RMS of 20 ms blocks maps to 0.1.
    fn gain(&self, pcm: &[f32]) -> f64 {
        let block = self.hop * 2;
        let k = pcm.len() / block;
        if k == 0 {
            return 1.0;
        }
        let mut rms: Vec<f32> = pcm
            .chunks_exact(block)
            .map(|b| {
                let mut s = 0f64;
                for &x in b {
                    if in_range(x) {
                        s += x as f64 * x as f64;
                    }
                }
                (s / block as f64).sqrt() as f32
            })
            .collect();
        rms.sort_by(f32::total_cmp);
        let q = rms[((k - 1) as f32 * 0.99f32).floor() as usize];
        if q > 1e-6 { 0.1f32 as f64 / q as f64 } else { 1.0 }
    }

    /// pcm: 16 kHz mono samples.
    pub fn run(&self, pcm: &[f32]) -> Features {
        let padded;
        let pcm = if pcm.len() < self.win {
            let mut p = pcm.to_vec();
            p.resize(self.win, 0.0);
            padded = p;
            &padded[..]
        } else {
            pcm
        };
        let n = pcm.len();
        let (t_n, m, n_fft, half, w) = (n / self.hop, self.mels, self.n_fft, self.n_fft >> 1, self.win);
        let gain = self.gain(pcm);
        let mut feat = vec![0f32; t_n * m];
        let mut db = vec![0f32; t_n];
        let mut re = vec![0f64; n_fft];
        let mut im = vec![0f64; n_fft];
        let mut pow = vec![0f64; half + 1];
        for t in 0..t_n {
            // frame t covers samples [hop*t - w/2, hop*t + w/2), zero outside the clip
            let start = (self.hop * t) as isize - (w >> 1) as isize;
            re.fill(0.0);
            im.fill(0.0);
            for a in 0..w {
                let idx = a as isize + start;
                if idx < 0 || idx >= n as isize {
                    continue;
                }
                let x = pcm[idx as usize];
                if !in_range(x) {
                    continue;
                }
                re[self.rev[a]] = gain * x as f64 * self.window[a] as f64;
            }
            let mut len = 2;
            while len <= n_fft {
                let (h2, step) = (len >> 1, n_fft / len);
                let mut i = 0;
                while i < n_fft {
                    for j in 0..h2 {
                        let (wr, wi) = (self.cos[j * step], -self.sin[j * step]);
                        let (p, q) = (i + j, i + j + h2);
                        let xr = re[q] * wr - im[q] * wi;
                        let xi = re[q] * wi + im[q] * wr;
                        re[q] = re[p] - xr;
                        im[q] = im[p] - xi;
                        re[p] += xr;
                        im[p] += xi;
                    }
                    i += len;
                }
                len <<= 1;
            }
            for b in 0..=half {
                pow[b] = re[b] * re[b] + im[b] * im[b];
            }
            let mut band = 0f64;
            for c in 0..m {
                let row = &self.fb[c * self.bins..(c + 1) * self.bins];
                let mut e = 0f64;
                for b in self.lo[c]..self.hi[c] {
                    e += row[b] as f64 * pow[b];
                }
                feat[t * m + c] = (e + 5.960464477539063e-08).ln() as f32;
                if c >= self.band_lo && c < self.band_hi {
                    band += e;
                }
            }
            db[t] = if self.band_hi > 0 { (10.0 * (band + 1e-30).log10()) as f32 } else { -300.0 };
        }
        // Per-channel mean / variance normalisation over the clip.
        let denom = 1.0 / (t_n.max(2) as f64 - 1.0);
        for c in 0..m {
            let mut mean = 0f64;
            for t in 0..t_n {
                mean += feat[t * m + c] as f64;
            }
            mean /= t_n as f64;
            let mut v = 0f64;
            for t in 0..t_n {
                let d = feat[t * m + c] as f64 - mean;
                v += d * d;
            }
            let inv = 1.0 / ((v * denom).sqrt() + 1e-5);
            for t in 0..t_n {
                feat[t * m + c] = ((feat[t * m + c] as f64 - mean) * inv) as f32;
            }
        }
        Features { feat, frames: t_n, db }
    }
}
