# xng-sdr provenance

## SoapySDR backend (`soapy` feature)

Uses the `soapysdr` Rust crate (Apache-2.0/MIT bindings to libSoapySDR).
Original glue code.

## Native Airspy R2/Mini backend (`airspy` feature)

Uses the **libairspy-rs** crate (BSD-3-Clause AND MIT; a pure-Rust port of
libairspy on `nusb`, with real-to-IQ converters bit-exact against the C
library). No C library is linked. Original glue code. Facts taken from
airspyone_host (BSD 3-clause):

- airspy.h: gain ranges (LNA/mixer/VGA 0..15, linearity/sensitivity 0..21),
  tuner range 24 MHz..1.75 GHz.
- airspy.c (`airspy_set_samplerate`): values not in the advertised list are
  forwarded to the firmware in kHz — arbitrary rates work on firmware
  >= 1.0.7, so the backend attempts the requested rate and only reports the
  advertised list when the device refuses.
## Native Airspy HF+ backend (`airspyhf` feature)

Direct FFI against **libairspyhf** (BSD 3-clause). The `extern "C"`
declarations in `src/airspyhf.rs` are transcribed from the public header
(`airspyhf.h`); no C source code is ported. Facts taken from it:

- airspyhf.h: transfer struct layout (samples are calibrated float32 IQ),
  AGC/attenuator (0..48 dB, 6 dB steps)/preamp (+6 dB) controls.
  `airspyhf_set_bias_tee` is deliberately not bound: it appeared in 1.8 and
  the widely-packaged 1.6.x lacks the symbol (and HF+ hardware has no bias
  tee).

Streaming design, both backends (driver callback thread -> bounded channel -> `IqSource::read`,
dropping transfers when the consumer lags) is original; the drop-don't-block
choice follows from USB callback semantics, not from any reference
implementation.

## IQ file sources

Original code.
