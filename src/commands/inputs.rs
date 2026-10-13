//! Station inputs and decoders: `[[sdr]]` blocks own a device (or IQ file)
//! and its RF settings; `[[decoder]]` blocks name the `[[sdr]]` they read and
//! say what to decode on it. Every decoder on one input shares a single
//! capture, whose center and sample rate are planned from the union of their
//! channels unless the `[[sdr]]` pins them.
//!
//! ```toml
//! [[sdr]]
//! id = "airspy"
//! device = "driver=airspy,serial=0123456789ABCDEF"
//! gain = 20
//!
//! [[decoder]]
//! sdr = "airspy"
//! mode = "acars"
//! channels = ["131.125", "131.550", "131.725"]
//!
//! [[decoder]]
//! sdr = "airspy"
//! mode = "vdl2"
//! channels = ["136.650", "136.800", "136.975"]
//! ```
//!
//! Each decoder becomes an ordinary station session (same device string,
//! rate, center and gain), so the shared-device runtime is unchanged.

use super::scan::RateCaps;
use super::station::{SessionToml, StationFile};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::PathBuf;
use xng_types::Mode;

/// An input: one SDR, or an IQ file, shared by the decoders that name it.
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct SdrToml {
    /// Name decoders refer to.
    pub id: String,
    /// SDR selector (`driver=airspy,serial=…`); mutually exclusive with `file`.
    pub device: Option<String>,
    /// IQ file replay instead of live hardware.
    pub file: Option<PathBuf>,
    /// Sample format for `file` (cf32/cs16/cs8/cu8); guessed from the
    /// extension when omitted.
    pub format: Option<String>,
    pub gain: Option<f64>,
    /// Capture rate; planned from the decoders' channels when omitted.
    pub sample_rate: Option<f64>,
    /// Capture center; planned from the decoders' channels when omitted.
    pub center: Option<String>,
}

/// A decoder: one mode on a set of channels, reading an `[[sdr]]`.
#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct DecoderToml {
    /// `id` of the `[[sdr]]` this decoder reads.
    pub sdr: String,
    pub mode: String,
    pub channels: Vec<String>,
    pub receiver_pos: Option<String>,
    pub demod_effort: Option<String>,
    /// VDL2 only: reject bursts whose carrier offset exceeds this many ppm.
    pub max_ppm: Option<f64>,
    #[serde(default)]
    pub ais_include_types: Vec<u8>,
    #[serde(default)]
    pub ais_exclude_types: Vec<u8>,
    #[serde(default)]
    pub ais_include_mmsi: Vec<u32>,
    #[serde(default)]
    pub ais_exclude_mmsi: Vec<u32>,
    pub ais_min_interval: Option<f64>,
    pub ais_dedup_window: Option<f64>,
    /// Disable Airframes feeding for this decoder.
    pub feed: Option<bool>,
    /// Override the Airframes station id for this decoder's messages.
    pub airframes_station_id: Option<String>,
}

/// When planning the rate, keep every channel inside this fraction of the
/// capture: filter roll-off near the band edges (an Airspy R2 at 10 MS/s is
/// flat to about ±4.5 MHz) would otherwise cost the outermost channels.
const USABLE_FRACTION: f64 = 0.9;

/// Plan one input's capture. `needs` are each channel's center and one-sided
/// passband (Hz); a zero passband marks a whole-capture mode (Mode S, UAT…),
/// which must sit at the capture center. `min_rate` is the slowest capture
/// any of the decoders can work from. An explicit `rate`/`center` is used as
/// given (and checked); otherwise the center is the midpoint of the channels,
/// nudged off any channel that would land at DC, and the rate is the smallest
/// the device supports that keeps every channel within the usable band.
pub fn plan_capture(
    needs: &[(u64, f64)],
    rate: Option<f64>,
    center: Option<u64>,
    min_rate: f64,
    caps: &RateCaps,
) -> Result<(f64, u64), String> {
    let lo = needs.iter().map(|n| n.0).min().ok_or("no channels to plan for")?;
    let hi = needs.iter().map(|n| n.0).max().unwrap_or(lo);
    let pinned: Vec<u64> = needs.iter().filter(|n| n.1 == 0.0).map(|n| n.0).collect();
    let center = match (center, pinned.first()) {
        (_, Some(&c)) if pinned.iter().any(|&f| f != c) => {
            return Err("whole-capture decoders on one input need the same center frequency".into())
        }
        (Some(c), Some(&p)) if c != p => {
            return Err(format!("a whole-capture decoder needs the center at {:.4} MHz", p as f64 / 1e6))
        }
        (_, Some(&p)) => p,
        (Some(c), None) => c,
        (None, None) => {
            let mid = (lo + hi) / 2;
            if needs.iter().any(|n| n.0 == mid) { mid + 25_000 } else { mid }
        }
    };
    // Smallest rate that keeps every channel's passband inside the capture.
    let half = needs
        .iter()
        .map(|&(f, pb)| (f as f64 - center as f64).abs() + pb)
        .fold(0.0f64, f64::max);
    let mhz = |hz: f64| hz / 1e6;
    if let Some(r) = rate {
        if r < min_rate {
            return Err(format!("sample-rate {:.3} MS/s is below the {:.3} MS/s a decoder needs", mhz(r), mhz(min_rate)));
        }
        if 2.0 * half > r {
            return Err(format!(
                "channels span {:.3}–{:.3} MHz, which needs at least {:.3} MS/s around {:.4} MHz; sample-rate is {:.3} MS/s",
                mhz(lo as f64), mhz(hi as f64), mhz(2.0 * half), mhz(center as f64), mhz(r)
            ));
        }
        return Ok((r, center));
    }
    let need = (2.0 * half / USABLE_FRACTION).max(min_rate);
    let picked = match caps {
        RateCaps::Discrete(rates) => {
            rates.iter().map(|&r| r as f64).filter(|&r| r >= need).fold(None, |m: Option<f64>, r| Some(m.map_or(r, |m| m.min(r))))
        }
        RateCaps::Ranges(ranges) => ranges
            .iter()
            .filter_map(|&(min, max, step)| {
                // Round up to a 100 kHz-friendly rate (2.4, 2.5 MS/s, …) or
                // the range's own step, staying inside the range.
                let grid = if step > 0.0 { step } else { 100_000.0 };
                let v = need.max(min);
                let v = if step > 0.0 { min + ((v - min) / grid).ceil() * grid } else { (v / grid).ceil() * grid };
                (v <= max).then_some(v)
            })
            .fold(None, |m: Option<f64>, r| Some(m.map_or(r, |m| m.min(r)))),
        RateCaps::Unknown => {
            return Err("can't tell which sample rates this input supports; set `sample-rate`".into())
        }
    };
    picked.map(|r| (r, center)).ok_or_else(|| {
        format!(
            "channels span {:.3}–{:.3} MHz, which needs about {:.3} MS/s; the device offers no rate that wide",
            mhz(lo as f64), mhz(hi as f64), mhz(need)
        )
    })
}

impl StationFile {
    /// Turn `[[sdr]]` + `[[decoder]]` blocks into station sessions, planning
    /// each input's capture. `probe` reports what rates a device supports.
    pub fn expand_inputs(&mut self, probe: &dyn Fn(&str) -> RateCaps) -> anyhow::Result<()> {
        if self.sdrs.is_empty() {
            anyhow::ensure!(self.decoders.is_empty(), "[[decoder]] blocks need an [[sdr]] to read from");
            return Ok(());
        }
        let mut ids = HashSet::new();
        for s in &self.sdrs {
            anyhow::ensure!(ids.insert(s.id.as_str()), "[[sdr]] id `{}` is defined twice", s.id);
            anyhow::ensure!(
                s.device.is_some() ^ s.file.is_some(),
                "[[sdr]] {}: exactly one of `device` or `file` is required",
                s.id
            );
            if let Some(dev) = &s.device {
                anyhow::ensure!(
                    !self.sessions.iter().any(|x| x.sdr.as_deref().map(str::trim) == Some(dev.trim())),
                    "[[sdr]] {}: device `{dev}` is also used by a [[session]]; configure it in one place",
                    s.id
                );
            }
        }
        for d in &self.decoders {
            anyhow::ensure!(
                ids.contains(d.sdr.as_str()),
                "[[decoder]] ({}) reads sdr `{}`, which no [[sdr]] defines",
                d.mode,
                d.sdr
            );
            anyhow::ensure!(!d.channels.is_empty(), "[[decoder]] ({} on {}) needs `channels`", d.mode, d.sdr);
        }

        let mut planned = Vec::new();
        for s in &self.sdrs {
            let decoders: Vec<&DecoderToml> = self.decoders.iter().filter(|d| d.sdr == s.id).collect();
            if decoders.is_empty() {
                tracing::warn!("[[sdr]] {}: no decoder reads it; not opened", s.id);
                continue;
            }
            let mut needs = Vec::new();
            let mut min_rate = 0.0f64;
            for d in &decoders {
                let mode: Mode = d.mode.parse().map_err(|e: String| anyhow::anyhow!("[[decoder]] on {}: {e}", s.id))?;
                let pb = super::scan::passband(mode);
                if pb == 0.0 {
                    // Whole-capture modes demodulate the capture itself.
                    min_rate = min_rate.max(super::scan::plan(mode).0);
                }
                for c in &d.channels {
                    needs.push((crate::freq::parse_hz(c)?, pb));
                }
            }
            let center = s.center.as_deref().map(crate::freq::parse_hz).transpose()?;
            if s.file.is_some() {
                anyhow::ensure!(
                    s.sample_rate.is_some() && center.is_some(),
                    "[[sdr]] {}: IQ file inputs need `sample-rate` and `center`",
                    s.id
                );
            }
            let caps = match (&s.device, s.sample_rate) {
                (Some(dev), None) => {
                    let caps = probe(dev);
                    anyhow::ensure!(
                        !matches!(caps, RateCaps::Unknown),
                        "[[sdr]] {}: couldn't ask `{dev}` which sample rates it supports (not connected, in use by another program, or no driver for it in this build); set `sample-rate` on the [[sdr]]",
                        s.id
                    );
                    caps
                }
                _ => RateCaps::Unknown,
            };
            let (rate, center) = plan_capture(&needs, s.sample_rate, center, min_rate, &caps)
                .map_err(|e| anyhow::anyhow!("[[sdr]] {}: {e}", s.id))?;
            tracing::info!(
                "[[sdr]] {}: {} decoder(s), {} channel(s) → {:.4} MHz at {} S/s",
                s.id,
                decoders.len(),
                needs.len(),
                center as f64 / 1e6,
                rate as u64
            );
            for d in decoders {
                planned.push(SessionToml {
                    sdr: s.device.clone(),
                    file: s.file.clone(),
                    format: s.format.clone(),
                    gain: s.gain,
                    mode: d.mode.clone(),
                    sample_rate: Some(rate),
                    center: Some(format!("{center}Hz")),
                    channels: d.channels.clone(),
                    receiver_pos: d.receiver_pos.clone(),
                    demod_effort: d.demod_effort.clone(),
                    max_ppm: d.max_ppm,
                    ais_include_types: d.ais_include_types.clone(),
                    ais_exclude_types: d.ais_exclude_types.clone(),
                    ais_include_mmsi: d.ais_include_mmsi.clone(),
                    ais_exclude_mmsi: d.ais_exclude_mmsi.clone(),
                    ais_min_interval: d.ais_min_interval,
                    ais_dedup_window: d.ais_dedup_window,
                    feed: d.feed,
                    airframes_station_id: d.airframes_station_id.clone(),
                });
            }
        }
        self.sessions.extend(planned);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACARS: [u64; 14] = [
        129_000_000, 129_350_000, 130_025_000, 130_425_000, 130_450_000, 130_850_000, 131_125_000,
        131_425_000, 131_475_000, 131_525_000, 131_550_000, 131_670_000, 131_725_000, 131_825_000,
    ];
    const VDL2: [u64; 5] = [136_100_000, 136_650_000, 136_700_000, 136_800_000, 136_975_000];

    fn needs() -> Vec<(u64, f64)> {
        let a = super::super::scan::passband(Mode::AcarsPoa);
        let v = super::super::scan::passband(Mode::Vdl2);
        ACARS.iter().map(|&f| (f, a)).chain(VDL2.iter().map(|&f| (f, v))).collect()
    }

    fn r2() -> RateCaps {
        RateCaps::Discrete(vec![10_000_000, 2_500_000])
    }

    #[test]
    fn plans_acars_and_vdl2_on_an_r2() {
        assert_eq!(plan_capture(&needs(), None, None, 0.0, &r2()), Ok((10_000_000.0, 132_987_500)));
    }

    #[test]
    fn narrow_set_takes_the_smaller_rate() {
        let n: Vec<_> = VDL2[1..].iter().map(|&f| (f, 8_500.0)).collect();
        let (rate, _) = plan_capture(&n, None, None, 0.0, &r2()).unwrap();
        assert_eq!(rate, 2_500_000.0);
    }

    #[test]
    fn rtl_sdr_ranges_round_up_to_a_friendly_rate() {
        let rtl = RateCaps::Ranges(vec![(225_001.0, 300_000.0, 0.0), (900_001.0, 3_200_000.0, 0.0)]);
        let n: Vec<_> = ACARS[6..].iter().map(|&f| (f, 5_000.0)).collect();
        let (rate, center) = plan_capture(&n, None, None, 0.0, &rtl).unwrap();
        assert_eq!(rate % 100_000.0, 0.0, "{rate}");
        assert!((900_001.0..=3_200_000.0).contains(&rate));
        for &(f, pb) in &n {
            assert!((f as f64 - center as f64).abs() + pb <= rate / 2.0 * USABLE_FRACTION);
        }
    }

    #[test]
    fn too_wide_for_the_device_is_an_error() {
        let n = vec![(129_000_000, 5_000.0), (162_000_000, 5_000.0)];
        let e = plan_capture(&n, None, None, 0.0, &r2()).unwrap_err();
        assert!(e.contains("no rate that wide"), "{e}");
    }

    #[test]
    fn explicit_rate_and_center_are_checked_not_replaced() {
        assert_eq!(plan_capture(&needs(), Some(10e6), Some(133_000_000), 0.0, &r2()), Ok((10e6, 133_000_000)));
        let e = plan_capture(&needs(), Some(2.5e6), None, 0.0, &r2()).unwrap_err();
        assert!(e.contains("needs at least"), "{e}");
    }

    #[test]
    fn unknown_rates_need_an_explicit_rate() {
        assert!(plan_capture(&needs(), None, None, 0.0, &RateCaps::Unknown).is_err());
    }

    #[test]
    fn center_moves_off_a_channel_at_dc() {
        let n = vec![(136_000_000, 8_500.0), (136_500_000, 8_500.0), (137_000_000, 8_500.0)];
        let (_, center) = plan_capture(&n, None, None, 0.0, &r2()).unwrap();
        assert_ne!(center, 136_500_000);
    }

    #[test]
    fn whole_capture_modes_pin_the_center_and_set_a_floor() {
        let rtl = RateCaps::Ranges(vec![(225_001.0, 300_000.0, 0.0), (900_001.0, 3_200_000.0, 0.0)]);
        let n = vec![(1_090_000_000, 0.0)];
        assert_eq!(plan_capture(&n, None, None, 2_000_000.0, &rtl), Ok((2_000_000.0, 1_090_000_000)));
        assert!(plan_capture(&n, None, Some(1_089_000_000), 2e6, &rtl).is_err());
        assert!(plan_capture(&n, Some(1e6), None, 2e6, &rtl).unwrap_err().contains("below"));
    }

    const TOML: &str = r#"
station-id = "X"

[[sdr]]
id = "airspy"
device = "driver=airspy,serial=0123456789ABCDEF"
gain = 20

[[decoder]]
sdr = "airspy"
mode = "acars"
channels = ["129.000", "131.825"]

[[decoder]]
sdr = "airspy"
mode = "vdl2"
channels = ["136.100", "136.975"]
feed = false
"#;

    fn parse(t: &str) -> StationFile {
        toml::from_str(t).unwrap()
    }

    #[test]
    fn decoders_become_sessions_sharing_one_capture() {
        let mut f = parse(TOML);
        f.expand_inputs(&|_| r2()).unwrap();
        assert_eq!(f.sessions.len(), 2);
        let (a, v) = (&f.sessions[0], &f.sessions[1]);
        assert_eq!(a.sdr, v.sdr);
        assert_eq!(a.sdr.as_deref(), Some("driver=airspy,serial=0123456789ABCDEF"));
        assert_eq!((a.sample_rate, a.gain), (Some(10e6), Some(20.0)));
        assert_eq!((a.sample_rate, &a.center, a.gain), (v.sample_rate, &v.center, v.gain));
        assert_eq!(crate::freq::parse_hz(a.center.as_deref().unwrap()).unwrap(), 132_987_500);
        assert_eq!((a.mode.as_str(), v.mode.as_str()), ("acars", "vdl2"));
        assert_eq!(v.feed, Some(false));
    }

    #[test]
    fn rejects_unknown_sdr_duplicate_ids_and_double_configuration() {
        let mut f = parse(&TOML.replace("sdr = \"airspy\"\nmode = \"vdl2\"", "sdr = \"rtl\"\nmode = \"vdl2\""));
        assert!(f.expand_inputs(&|_| r2()).unwrap_err().to_string().contains("no [[sdr]] defines"));

        let dup = format!("{TOML}\n[[sdr]]\nid = \"airspy\"\ndevice = \"driver=rtlsdr\"\n");
        assert!(parse(&dup).expand_inputs(&|_| r2()).unwrap_err().to_string().contains("defined twice"));

        let both = format!(
            "{TOML}\n[[session]]\nsdr = \"driver=airspy,serial=0123456789ABCDEF\"\nmode = \"ais\"\n"
        );
        assert!(parse(&both).expand_inputs(&|_| r2()).unwrap_err().to_string().contains("configure it in one place"));
    }

    #[test]
    fn plans_the_example_config() {
        let mut f = parse(include_str!("../../contrib/station.example.toml"));
        f.expand_inputs(&|dev: &str| {
            if dev.contains("airspy") {
                r2()
            } else {
                RateCaps::Ranges(vec![(225_001.0, 300_000.0, 0.0), (900_001.0, 3_200_000.0, 0.0)])
            }
        })
        .unwrap();
        assert_eq!(f.sessions.len(), 4);
        let by_mode = |sdr: &str, mode: &str| {
            f.sessions.iter().find(|s| s.sdr.as_deref().is_some_and(|d| d.contains(sdr)) && s.mode == mode).unwrap()
        };
        let (a, v) = (by_mode("airspy", "acars"), by_mode("airspy", "vdl2"));
        assert_eq!((a.sample_rate, &a.center), (v.sample_rate, &v.center), "decoders on one input share a capture");
        assert_eq!(a.sample_rate, Some(10e6));
        let adsb = by_mode("00000002", "adsb");
        assert_eq!(crate::freq::parse_hz(adsb.center.as_deref().unwrap()).unwrap(), 1_090_000_000);
        assert!(adsb.sample_rate.unwrap() >= 2e6);
    }

    #[test]
    fn file_inputs_need_rate_and_center() {
        let t = TOML.replace("device = \"driver=airspy,serial=0123456789ABCDEF\"", "file = \"x.cs16\"");
        assert!(parse(&t).expand_inputs(&|_| r2()).unwrap_err().to_string().contains("IQ file inputs need"));
        let t = t.replace("gain = 20", "sample-rate = 10000000\ncenter = \"132.9875M\"");
        let mut f = parse(&t);
        f.expand_inputs(&|_| RateCaps::Unknown).unwrap();
        assert_eq!(f.sessions[0].file.as_deref(), Some(std::path::Path::new("x.cs16")));
    }
}
