//! Fast-convolution (overlap-save) multi-channel downconverter.
//!
//! Same contract as [`SharedDdc`](crate::SharedDdc) and
//! [`ChannelizedDdc`](crate::ChannelizedDdc), built on the overlap-save filter
//! bank (Borgerding, "Turning overlap-save into a multiband mixing,
//! downsampling filter bank", IEEE SPM 2006):
//!
//! ```text
//!   wideband ── N-point FFT (shared, once per block of L new samples)
//!                 ├─ bins around ch0 × H ── N/D-point IFFT ─ phase/NCO ─ resample ─► ch0
//!                 ├─ bins around ch1 × H ── N/D-point IFFT ─ phase/NCO ─ resample ─► ch1
//!                 └─ …
//! ```
//!
//! One forward FFT serves every channel; each channel costs an N/D-bin
//! multiply and a small inverse FFT, which also decimates by D. Unlike the
//! polyphase channelizer, channels are not snapped to a fixed bin grid: each
//! is shifted to the nearest FFT bin (≤ fs/2N away) and a low-rate NCO removes
//! the remainder, so placement is exact anywhere in the capture. Unlike the
//! shared decimation, cost does not grow with how far the channels sit from
//! the capture center.
//!
//! The channel filter `H` is one windowed-sinc lowpass (cutoff = passband,
//! transition ≈ passband) shared by every channel; its frequency response is
//! computed once.

use crate::fir::lowpass_taps;
use crate::nco::Nco;
use crate::resample::Resampler;
use crate::IqSample;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Smallest / largest forward FFT considered.
const MIN_FFT: usize = 1024;
const MAX_FFT: usize = 1 << 16;
/// Fewest bins a channel's inverse FFT may have.
const MIN_HEAD_BINS: usize = 16;

struct Head {
    /// Forward-FFT bin nearest the channel center.
    bin: usize,
    /// Per-block phase correction: overlap-save references each block's
    /// shift to the block start, so successive blocks differ by
    /// e^{-j2π·bin·L/N}; `phase` accumulates that to keep the output
    /// continuous.
    step: IqSample,
    phase: IqSample,
    /// Removes the channel's remaining offset from its bin center.
    nco: Nco,
    resampler: Option<Resampler>,
    spec: Vec<IqSample>,
    dec: Vec<IqSample>,
}

pub struct FastConvDdc {
    n: usize,
    /// Fresh input samples consumed per block (N − filter overlap).
    hop: usize,
    /// Leading decimated samples of each block to discard (overlap / D).
    skip: usize,
    nd: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    /// The channel filter's response on the N/D bins around DC, in IFFT
    /// order (index j mod N/D for j in −N/2D .. N/2D), pre-scaled by 1/N.
    h: Vec<IqSample>,
    /// Pending input: the previous block's overlap tail plus new samples.
    buf: Vec<IqSample>,
    spectrum: Vec<IqSample>,
    fwd_scratch: Vec<IqSample>,
    inv_scratch: Vec<IqSample>,
    heads: Vec<Head>,
}

impl FastConvDdc {
    /// * `input_rate` — wideband capture rate.
    /// * `output_rate` — per-channel rate (same for all channels).
    /// * `offsets` — each channel's center relative to the capture center.
    /// * `passband_hz` — one-sided width of each channel's signal.
    pub fn new(
        input_rate: f64,
        output_rate: f64,
        offsets: &[f64],
        passband_hz: f64,
    ) -> Result<Self, String> {
        if offsets.is_empty() {
            return Err("FastConvDdc needs at least one channel".to_string());
        }
        if output_rate < 2.0 * passband_hz {
            return Err(format!(
                "output rate {output_rate} cannot carry a ±{passband_hz} Hz passband"
            ));
        }
        // Filter: cutoff at the passband with a transition about as wide again
        // (Blackman-Harris needs ~4·fs/Δf taps). The overlap holds the taps, so
        // N = 4·overlap keeps 75% of every FFT as new samples.
        let taps_wanted = (4.0 * input_rate / passband_hz).ceil() as usize;
        let n = (4 * taps_wanted).next_power_of_two().clamp(MIN_FFT, MAX_FFT);
        let overlap = n / 4;
        let taps = overlap + 1;
        let hop = n - overlap;

        // Decimation: the largest power of two that keeps the decimated rate
        // above the filter's stopband (passband + transition = 2·passband)
        // on both sides and at or above the output rate, with a little margin
        // for the final resampler. Must divide the overlap so block edges land
        // on output samples.
        let min_rate = (4.0 * passband_hz).max(output_rate) * 1.25;
        let mut d = 1usize;
        while d * 2 <= overlap && n / (d * 2) >= MIN_HEAD_BINS && input_rate / (d * 2) as f64 >= min_rate {
            d *= 2;
        }
        let nd = n / d;
        let dec_rate = input_rate / d as f64;

        for &off in offsets {
            if 2.0 * (off.abs() + passband_hz) > input_rate {
                return Err(format!(
                    "channel offset {off} Hz is outside the {input_rate} S/s capture"
                ));
            }
        }

        let mut planner = FftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(nd);

        // The filter response on all N bins; keep the N/D around DC.
        let mut hfull: Vec<IqSample> = lowpass_taps(passband_hz / input_rate, taps)
            .into_iter()
            .map(|t| IqSample::new(t, 0.0))
            .chain(std::iter::repeat(IqSample::new(0.0, 0.0)))
            .take(n)
            .collect();
        fwd.process(&mut hfull);
        let scale = 1.0 / n as f32;
        let h: Vec<IqSample> = (0..nd)
            .map(|i| {
                // i in IFFT order: 0..nd/2 are j = i, nd/2..nd are j = i − nd.
                let j = if i < nd / 2 { i as isize } else { i as isize - nd as isize };
                hfull[j.rem_euclid(n as isize) as usize] * scale
            })
            .collect();

        let bin_hz = input_rate / n as f64;
        let heads = offsets
            .iter()
            .map(|&off| {
                let k = (off / bin_hz).round() as isize;
                let residual = off - k as f64 * bin_hz;
                let bin = k.rem_euclid(n as isize) as usize;
                let ang = -std::f64::consts::TAU * (k as f64) * (hop as f64) / n as f64;
                Head {
                    bin,
                    step: IqSample::new(ang.cos() as f32, ang.sin() as f32),
                    phase: IqSample::new(1.0, 0.0),
                    nco: Nco::new(residual, dec_rate),
                    resampler: ((dec_rate - output_rate).abs() > 1e-6)
                        .then(|| Resampler::new(dec_rate, output_rate)),
                    spec: vec![IqSample::new(0.0, 0.0); nd],
                    dec: Vec::new(),
                }
            })
            .collect();

        let (fwd_scratch_len, inv_scratch_len) =
            (fwd.get_inplace_scratch_len(), inv.get_inplace_scratch_len());
        Ok(Self {
            n,
            hop,
            skip: overlap / d,
            nd,
            fwd,
            inv,
            h,
            // Start with a zero overlap so output sample 0 maps to input 0.
            buf: vec![IqSample::new(0.0, 0.0); overlap],
            spectrum: vec![IqSample::new(0.0, 0.0); n],
            fwd_scratch: vec![IqSample::new(0.0, 0.0); fwd_scratch_len],
            inv_scratch: vec![IqSample::new(0.0, 0.0); inv_scratch_len],
            heads,
        })
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.heads.len()
    }

    /// Downconvert a wideband block into all channels. `out` must hold one
    /// vector per channel; each is cleared and filled with that channel's
    /// output samples.
    pub fn process(&mut self, input: &[IqSample], out: &mut [Vec<IqSample>]) {
        for o in out.iter_mut() {
            o.clear();
        }
        self.buf.extend_from_slice(input);
        let (n, nd) = (self.n, self.nd);
        let mut start = 0;
        while self.buf.len() - start >= n {
            self.spectrum.copy_from_slice(&self.buf[start..start + n]);
            self.fwd.process_with_scratch(&mut self.spectrum, &mut self.fwd_scratch);
            for (head, o) in self.heads.iter_mut().zip(out.iter_mut()) {
                for (i, s) in head.spec.iter_mut().enumerate() {
                    let j = if i < nd / 2 { i as isize } else { i as isize - nd as isize };
                    let b = (head.bin as isize + j).rem_euclid(n as isize) as usize;
                    *s = self.spectrum[b] * self.h[i];
                }
                self.inv.process_with_scratch(&mut head.spec, &mut self.inv_scratch);
                head.dec.clear();
                head.dec.extend(head.spec[self.skip..].iter().map(|&s| s * head.phase));
                head.nco.mix(&mut head.dec);
                head.phase *= head.step;
                // Renormalize the accumulated rotation against drift.
                head.phase /= head.phase.norm();
                match &mut head.resampler {
                    Some(r) => r.process(&head.dec, o),
                    None => o.extend_from_slice(&head.dec),
                }
            }
            start += self.hop;
        }
        self.buf.drain(..start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn tone(fs: f64, f: f64, len: usize) -> Vec<IqSample> {
        (0..len)
            .map(|i| {
                let p = TAU * f * i as f64 / fs;
                IqSample::new(p.cos() as f32, p.sin() as f32)
            })
            .collect()
    }

    fn mean_power(x: &[IqSample]) -> f64 {
        x.iter().map(|s| s.norm_sqr() as f64).sum::<f64>() / x.len().max(1) as f64
    }

    #[test]
    fn tone_on_channel_comes_out_at_baseband_with_unity_gain() {
        let fs = 10_000_000.0;
        let off = 3_712_500.0;
        let mut ddc = FastConvDdc::new(fs, 100_000.0, &[off], 8_500.0).unwrap();
        // 2 kHz above the channel center: should appear at +2 kHz baseband.
        let x = tone(fs, off + 2_000.0, 2_000_000);
        let mut out = vec![Vec::new()];
        let mut y = Vec::new();
        for chunk in x.chunks(65_536) {
            ddc.process(chunk, &mut out);
            y.extend_from_slice(&out[0]);
        }
        // ~100 kS/s for 0.2 s.
        assert!((y.len() as i64 - 20_000).abs() < 400, "len {}", y.len());
        let settled = &y[2_000..];
        let p = mean_power(settled);
        assert!((p - 1.0).abs() < 0.02, "power {p}");
        // Frequency: mean phase advance per sample ≈ 2π·2000/100000.
        let mut acc = 0.0f64;
        for w in settled.windows(2) {
            acc += (w[1] * w[0].conj()).arg() as f64;
        }
        let f_est = acc / (settled.len() - 1) as f64 / TAU * 100_000.0;
        assert!((f_est - 2_000.0).abs() < 5.0, "f_est {f_est}");
    }

    #[test]
    fn adjacent_channel_is_rejected() {
        let fs = 10_000_000.0;
        let off = 3_712_500.0;
        let mut ddc = FastConvDdc::new(fs, 100_000.0, &[off], 8_500.0).unwrap();
        // A full-scale signal one 25 kHz channel away.
        let x = tone(fs, off + 25_000.0, 1_000_000);
        let mut out = vec![Vec::new()];
        let mut y = Vec::new();
        for chunk in x.chunks(65_536) {
            ddc.process(chunk, &mut out);
            y.extend_from_slice(&out[0]);
        }
        let p_db = 10.0 * mean_power(&y[2_000..]).log10();
        assert!(p_db < -60.0, "adjacent leak {p_db} dB");
    }

    #[test]
    fn many_channels_each_see_only_their_own_tone() {
        let fs = 10_000_000.0;
        let offs = [3_112_500.0, 3_662_500.0, 3_712_500.0, 3_812_500.0, 3_987_500.0, -3_987_500.0];
        let mut ddc = FastConvDdc::new(fs, 100_000.0, &offs, 8_500.0).unwrap();
        // One tone on channel 2 only.
        let x = tone(fs, offs[2] + 1_000.0, 1_000_000);
        let mut out = vec![Vec::new(); offs.len()];
        let mut ys = vec![Vec::new(); offs.len()];
        for chunk in x.chunks(50_000) {
            ddc.process(chunk, &mut out);
            for (y, o) in ys.iter_mut().zip(&out) {
                y.extend_from_slice(o);
            }
        }
        for (i, y) in ys.iter().enumerate() {
            let p = mean_power(&y[2_000..]);
            if i == 2 {
                assert!((p - 1.0).abs() < 0.02, "ch{i} power {p}");
            } else {
                assert!(p < 1e-6, "ch{i} leak {p}");
            }
        }
    }

    #[test]
    fn output_is_continuous_across_blocks() {
        // A tone whose bin shift is not block-periodic: any per-block phase
        // error shows up as a jump in the sample-to-sample phase step.
        let fs = 10_000_000.0;
        let off = -1_234_567.0;
        let mut ddc = FastConvDdc::new(fs, 100_000.0, &[off], 8_500.0).unwrap();
        let x = tone(fs, off + 500.0, 1_500_000);
        let mut out = vec![Vec::new()];
        let mut y = Vec::new();
        for chunk in x.chunks(12_345) {
            ddc.process(chunk, &mut out);
            y.extend_from_slice(&out[0]);
        }
        let expect = TAU * 500.0 / 100_000.0;
        for w in y[2_000..].windows(2) {
            let d = (w[1] * w[0].conj()).arg() as f64;
            assert!((d - expect).abs() < 0.01, "phase step {d} vs {expect}");
        }
    }
}
