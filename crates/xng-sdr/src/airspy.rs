//! Native Airspy R2 / Mini capture via `libairspy-rs` (BSD-3-Clause AND MIT).
//!
//! `libairspy-rs` is a pure-Rust port of libairspy on `nusb`: same USB wire
//! behavior, and real-to-IQ converters bit-exact against the C library's
//! golden vectors. No C library, libusb, or pkg-config is needed at build or
//! run time. The driver delivers float32 IQ on its own consumer thread;
//! samples cross into [`crate::IqSource::read`] through
//! [`crate::pump::SamplePump`].
//!
//! Rates: the device advertises a discrete list (R2: 2.5/10 Msps, Mini:
//! 3/6 Msps), but firmware >= 1.0.7 accepts arbitrary rates, which the driver
//! forwards in kHz when the request is not in the list. We therefore attempt
//! whatever was asked and only report the advertised list if the device
//! refuses.

use crate::pump::{SamplePump, sample_channel};
use crate::{IqSource, SdrError};
use libairspy::conversion::Samples;
use libairspy::stream::Transfer;
use libairspy::Device;
use num_complex::Complex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn dev_err(what: &str, e: libairspy::Error) -> SdrError {
    SdrError::Device(format!("{what}: {e}"))
}

/// Driver version triple (smoke test; works without hardware).
pub fn lib_version() -> (u32, u32, u32) {
    let v = libairspy::lib_version();
    (v.major, v.minor, v.revision)
}

/// Serial numbers of connected Airspy devices.
pub fn enumerate() -> Result<Vec<u64>, SdrError> {
    libairspy::list_devices().map_err(|e| dev_err("airspy list_devices", e))
}

/// Tuner range from airspy.h: 24 MHz .. 1.75 GHz.
pub const FREQ_MIN_HZ: u64 = 24_000_000;
pub const FREQ_MAX_HZ: u64 = 1_750_000_000;

fn open_device(serial: Option<u64>) -> Result<Device, libairspy::Error> {
    match serial {
        Some(sn) => Device::open_serial(sn),
        None => Device::open(),
    }
}

/// The sample rates a connected device advertises (briefly opens it).
pub fn device_rates(serial: Option<u64>) -> Result<Vec<u32>, SdrError> {
    let dev = open_device(serial).map_err(|e| dev_err("airspy open", e))?;
    Ok(dev.samplerates())
}

/// Map a requested dB gain onto the R820T linearity-optimized composite gain
/// (22 steps spanning roughly 0..45 dB).
pub fn linearity_index(gain_db: f64) -> u8 {
    (gain_db / 45.0 * 21.0).round().clamp(0.0, 21.0) as u8
}

#[derive(Default)]
struct StreamStats {
    transfers: AtomicU64,
    /// Samples the driver dropped at the USB/device level (its own counter).
    usb_dropped: AtomicU64,
    /// Transfers dropped because our consumer channel was full.
    chan_dropped: AtomicU64,
    /// Transfers that arrived essentially all-zero (a device stall).
    near_zero: AtomicU64,
}

/// A single-channel RX capture from an Airspy R2/Mini.
pub struct AirspyIqSource {
    // Stops streaming and releases the interface on drop.
    _dev: Device,
    pump: SamplePump,
    sample_rate: f64,
    center_freq_hz: u64,
    stats: Arc<StreamStats>,
    reads: u64,
}

impl AirspyIqSource {
    /// Open (optionally by serial), tune, and start streaming.
    ///
    /// `gain_db` of 0..45 maps onto the linearity composite gain; hardware
    /// AGC (LNA+mixer, VGA mid-scale) when omitted.
    pub fn open(
        serial: Option<u64>,
        sample_rate: f64,
        center_freq_hz: u64,
        gain_db: Option<f64>,
        bias_t: bool,
    ) -> Result<Self, SdrError> {
        if !(FREQ_MIN_HZ..=FREQ_MAX_HZ).contains(&center_freq_hz) {
            return Err(SdrError::Config(format!(
                "airspy tunes 24 MHz .. 1.75 GHz; {center_freq_hz} Hz is outside"
            )));
        }
        let mut dev = open_device(serial).map_err(|e| {
            SdrError::Device(format!(
                "airspy open: {e} (no Airspy R2/Mini connected{}?)",
                if serial.is_some() { " with that serial" } else { "" }
            ))
        })?;

        dev.set_sample_type(libairspy::commands::SampleType::Float32Iq)
            .map_err(|e| dev_err("set_sample_type", e))?;

        let rate = sample_rate.round() as u32;
        if dev.set_samplerate(rate).is_err() {
            // Verified on a Mini running the final firmware (v1.0.0-rc10-6):
            // off-list rates are refused even though the driver forwards
            // them, so the advertised list is the only safe guidance.
            return Err(SdrError::Config(format!(
                "device refused {rate} S/s; use one of its supported rates: {}",
                dev.samplerates().iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
            )));
        }

        match gain_db {
            Some(g) => dev
                .set_linearity_gain(linearity_index(g))
                .map_err(|e| dev_err("set_linearity_gain", e))?,
            None => {
                dev.set_lna_agc(true).map_err(|e| dev_err("set_lna_agc", e))?;
                dev.set_mixer_agc(true).map_err(|e| dev_err("set_mixer_agc", e))?;
                dev.set_vga_gain(13).map_err(|e| dev_err("set_vga_gain", e))?;
            }
        }
        if bias_t {
            dev.set_rf_bias(true).map_err(|e| dev_err("set_rf_bias", e))?;
        }
        dev.set_freq(center_freq_hz as u32).map_err(|e| dev_err("set_freq", e))?;

        let (tx, pump) = sample_channel();
        let stats = Arc::new(StreamStats::default());
        let cb_stats = stats.clone();
        dev.start_rx(move |t: Transfer<'_>| {
            let Samples::Float32(iq) = t.samples else { return true };
            if iq.len() < 2 {
                return true;
            }
            cb_stats.transfers.fetch_add(1, Ordering::Relaxed);
            if t.dropped_samples > 0 {
                cb_stats.usb_dropped.fetch_add(t.dropped_samples, Ordering::Relaxed);
            }
            let samples: Vec<Complex<f32>> =
                iq.chunks_exact(2).map(|p| Complex::new(p[0], p[1])).collect();
            let head_pwr: f32 = samples.iter().take(64).map(|s| s.norm_sqr()).sum();
            if head_pwr < 1e-9 {
                cb_stats.near_zero.fetch_add(1, Ordering::Relaxed);
            }
            // Drop the transfer if the consumer is behind; stalling the
            // driver thread only moves the overflow into the device.
            if tx.try_send(samples).is_err() {
                cb_stats.chan_dropped.fetch_add(1, Ordering::Relaxed);
            }
            true
        })
        .map_err(|e| dev_err("start_rx", e))?;

        Ok(Self { _dev: dev, pump, sample_rate, center_freq_hz, stats, reads: 0 })
    }
}

impl IqSource for AirspyIqSource {
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn center_freq_hz(&self) -> u64 {
        self.center_freq_hz
    }

    fn read(&mut self, buf: &mut [Complex<f32>]) -> Result<usize, SdrError> {
        self.reads += 1;
        // Stream-health watchdog: warn (periodically) only if the device or
        // pipeline is actually losing samples — silent when healthy.
        if self.reads % 500 == 0 {
            let s = &self.stats;
            let usb = s.usb_dropped.load(Ordering::Relaxed);
            let chan = s.chan_dropped.load(Ordering::Relaxed);
            let nz = s.near_zero.load(Ordering::Relaxed);
            if usb > 0 || chan > 0 || nz > 0 {
                eprintln!(
                    "airspy stream WARNING: usb_dropped={usb}, chan_dropped={chan}, near_zero={nz} \
                     over {} transfers",
                    s.transfers.load(Ordering::Relaxed)
                );
            }
        }
        self.pump.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_and_enumeration_work_without_hardware() {
        let _ = lib_version();
        // No device attached in CI: must still return cleanly. A host with
        // no usable USB stack (some containers) reports an error instead of
        // an empty list; either way it must not panic.
        let _ = enumerate();
    }

    #[test]
    fn linearity_gain_mapping() {
        assert_eq!(linearity_index(0.0), 0);
        assert_eq!(linearity_index(45.0), 21);
        assert_eq!(linearity_index(100.0), 21);
        assert_eq!(linearity_index(-3.0), 0);
        assert_eq!(linearity_index(21.0), 10);
    }
}
