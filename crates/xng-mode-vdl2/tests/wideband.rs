//! Wideband regression for the shared VDL2 front end: the vendored off-air
//! capture (50 kS/s, see tests/data/README.md) is upsampled into a wideband
//! capture at each common SDR rate, placed off-center between an empty
//! adjacent channel (25 kHz away) and an empty mirror-image channel, and
//! decoded through `Vdl2MultiChannelDecoder`. Every frame the direct 50 kS/s
//! decode finds must come out of the right channel, and nothing out of the
//! empty ones.

use num_complex::Complex;
use std::collections::BTreeSet;
use xng_mode_vdl2::{Vdl2ChannelDecoder, Vdl2MultiChannelDecoder};

const FIXTURE_RATE: usize = 50_000;

fn fixture() -> Vec<Complex<f32>> {
    let raw = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/vdl2_offair_6s.i16"))
        .expect("fixture present");
    raw.chunks_exact(4)
        .map(|b| {
            Complex::new(
                i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                i16::from_le_bytes([b[2], b[3]]) as f32 / 32768.0,
            )
        })
        .collect()
}

/// Frames from the direct channel-rate decode, as raw AVLC octets.
fn baseline(x: &[Complex<f32>]) -> BTreeSet<Vec<u8>> {
    let mut dec = Vdl2ChannelDecoder::new(FIXTURE_RATE as f64, 0.0).unwrap();
    let mut out = BTreeSet::new();
    for c in x.chunks(65_536) {
        out.extend(dec.process(c).into_iter().map(|f| f.avlc.raw));
    }
    out
}

/// Streams `x` upsampled by an integer factor and shifted to `offset_hz`,
/// in chunks, so a 10 MS/s capture never has to sit in memory.
struct Wideband<'a> {
    x: &'a [Complex<f32>],
    l: usize,
    phases: Vec<Vec<f32>>,
    pos: usize,
    n_out: u64,
    w: f64,
}

/// Taps per polyphase branch of the interpolation filter.
const TAPS_PER_PHASE: usize = 16;

impl<'a> Wideband<'a> {
    fn new(x: &'a [Complex<f32>], rate: usize, offset_hz: f64) -> Self {
        assert_eq!(rate % FIXTURE_RATE, 0, "integer upsampling only");
        let l = rate / FIXTURE_RATE;
        // Windowed-sinc lowpass at the fixture's band edge, gain l.
        let taps = xng_dsp::lowpass_taps(0.45 / l as f64, l * TAPS_PER_PHASE);
        let phases = (0..l)
            .map(|p| (0..TAPS_PER_PHASE).map(|k| taps[k * l + p] * l as f32).collect())
            .collect();
        let w = std::f64::consts::TAU * offset_hz / rate as f64;
        Self { x, l, phases, pos: 0, n_out: 0, w }
    }

    /// Next chunk of up to `n_in` input samples' worth of output, or None.
    fn next_chunk(&mut self, n_in: usize) -> Option<Vec<Complex<f32>>> {
        if self.pos >= self.x.len() {
            return None;
        }
        let end = (self.pos + n_in).min(self.x.len());
        let mut out = Vec::with_capacity((end - self.pos) * self.l);
        for n in self.pos..end {
            for ph in &self.phases {
                let mut acc = Complex::new(0.0f32, 0.0);
                for (k, &h) in ph.iter().enumerate() {
                    if n >= k {
                        acc += self.x[n - k] * h;
                    }
                }
                let a = self.w * self.n_out as f64;
                out.push(acc * Complex::new(a.cos() as f32, a.sin() as f32));
                self.n_out += 1;
            }
        }
        self.pos = end;
        Some(out)
    }
}

fn check_rate(rate: usize) {
    let x = fixture();
    let want = baseline(&x);
    assert!(want.len() >= 2, "baseline decode found {} frames", want.len());

    // Signal channel well off-center on the 12.5 kHz raster; an empty
    // adjacent channel 25 kHz below it; an empty mirror-image channel.
    let target = ((0.35 * rate as f64) / 12_500.0).round() * 12_500.0;
    let offsets = [target, target - 25_000.0, -target];

    let mut dec = Vdl2MultiChannelDecoder::new(rate as f64, &offsets).unwrap();
    assert_eq!(dec.front_end(), "fast-convolution", "{rate} S/s fell back to {}", dec.front_end());

    let mut got: Vec<BTreeSet<Vec<u8>>> = vec![BTreeSet::new(); offsets.len()];
    let mut wb = Wideband::new(&x, rate, target);
    while let Some(chunk) = wb.next_chunk(4_096) {
        for (i, frames) in dec.process(&chunk) {
            got[i].extend(frames.into_iter().map(|f| f.avlc.raw));
        }
    }
    let missing: Vec<_> = want.difference(&got[0]).collect();
    assert!(missing.is_empty(), "{rate} S/s: {} of {} baseline frames missing", missing.len(), want.len());
    assert!(got[1].is_empty(), "{rate} S/s: adjacent channel decoded {} frames", got[1].len());
    assert!(got[2].is_empty(), "{rate} S/s: mirror channel decoded {} frames", got[2].len());
}

#[test]
fn rtl_sdr_2_4_msps() {
    check_rate(2_400_000);
}

#[test]
fn airspy_r2_2_5_msps() {
    check_rate(2_500_000);
}

#[test]
fn airspy_mini_6_msps() {
    check_rate(6_000_000);
}

#[test]
fn airspy_r2_10_msps() {
    check_rate(10_000_000);
}
