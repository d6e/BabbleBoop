//! Converts a recording into the WAV file that is uploaded for
//! transcription: mono, 16 kHz, 16 bit PCM.
//!
//! Whisper resamples all input to 16 kHz (Radford et al. 2022, "Robust
//! Speech Recognition via Large-Scale Weak Supervision", section 2.2), and
//! speech carries little above 8 kHz, so a higher rate or more channels
//! mostly make the upload larger. One minute is about 1.9 MB in this
//! format, against 23 MB as 32 bit float stereo at 48 kHz.

use crate::types::CapturedAudio;
use hound::WavWriter;
use std::f64::consts::PI;
use std::io::Cursor;

/// Sample rate of the uploaded WAV.
pub const UPLOAD_SAMPLE_RATE: u32 = 16_000;

/// Filter taps on each side of the centre, per input sample per output
/// sample. More taps give a steeper low-pass filter.
const TAPS_PER_RATIO: f64 = 16.0;

/// Low-pass cutoff as a fraction of the output Nyquist frequency. The
/// filter rolls off around it, so the margin keeps most content just below
/// the output Nyquist frequency out of the aliased band.
const CUTOFF: f64 = 0.85;

/// Encode the recording as a mono 16 kHz 16 bit WAV file.
pub fn encode_upload_wav(audio: &CapturedAudio) -> Result<Vec<u8>, hound::Error> {
    let mono = downmix(&audio.samples, audio.channels);
    let samples = resample(&mono, audio.sample_rate, UPLOAD_SAMPLE_RATE);

    let mut wav_buffer = Vec::new();
    let mut writer = WavWriter::new(
        Cursor::new(&mut wav_buffer),
        hound::WavSpec {
            channels: 1,
            sample_rate: UPLOAD_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for &sample in &samples {
        writer.write_sample(to_i16(sample))?;
    }
    writer.finalize()?;
    Ok(wav_buffer)
}

/// Convert to 16 bit, clipping values outside [-1, 1] instead of letting
/// them wrap around.
fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

/// Average the channels of each frame. An incomplete last frame is dropped.
fn downmix(samples: &[f32], channels: u16) -> Vec<f32> {
    let channels = usize::from(channels.max(1));
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample mono audio. When the rate goes down, a windowed sinc low-pass
/// filter first removes the frequencies that the output rate cannot hold,
/// so they do not fold back into the speech band. Linear interpolation
/// between the filtered samples then gives the values at the output times.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    // Input samples per output sample
    let step = f64::from(from_rate) / f64::from(to_rate);
    let output_len = (input.len() as f64 / step).round() as usize;
    let kernel = (step > 1.0).then(|| low_pass_kernel(step));
    let value_at = |index: usize| -> f64 {
        match &kernel {
            Some(kernel) => filtered_sample(input, kernel, index),
            None => f64::from(input[index]),
        }
    };

    let last = input.len() - 1;
    (0..output_len)
        .map(|k| {
            let position = k as f64 * step;
            let index = (position.floor() as usize).min(last);
            let fraction = position - index as f64;
            let current = value_at(index);
            let value = if fraction > 0.0 && index < last {
                current + (value_at(index + 1) - current) * fraction
            } else {
                current
            };
            value as f32
        })
        .collect()
}

/// Blackman windowed sinc low-pass filter for downsampling by `step`,
/// normalised to unity gain at 0 Hz.
fn low_pass_kernel(step: f64) -> Vec<f64> {
    let half_width = (TAPS_PER_RATIO * step).ceil() as usize;
    let len = 2 * half_width + 1;
    // Cutoff in cycles per input sample
    let cutoff = CUTOFF * 0.5 / step;
    let mut kernel: Vec<f64> = (0..len)
        .map(|n| {
            let x = n as f64 - half_width as f64;
            let sinc = if x == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * x).sin() / (PI * x)
            };
            let phase = 2.0 * PI * n as f64 / (len - 1) as f64;
            let window = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            sinc * window
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    for tap in &mut kernel {
        *tap /= sum;
    }
    kernel
}

/// Output of the filter at `index`, treating samples outside the input
/// as silence.
fn filtered_sample(input: &[f32], kernel: &[f64], index: usize) -> f64 {
    let half_width = kernel.len() / 2;
    let start = index.saturating_sub(half_width);
    let end = (index + half_width + 1).min(input.len());
    input
        .get(start..end)
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(offset, &sample)| {
            // Tap for the sample at start + offset; the kernel is symmetric
            let tap = start + offset + half_width - index;
            kernel[tap] * f64::from(sample)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavReader;

    fn sine(frequency: f32, sample_rate: u32, seconds: f32, amplitude: f32) -> Vec<f32> {
        let len = (sample_rate as f32 * seconds) as usize;
        (0..len)
            .map(|n| {
                let t = n as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect()
    }

    /// Repeat each sample for every channel.
    fn interleave(mono: &[f32], channels: usize) -> Vec<f32> {
        mono.iter()
            .flat_map(|&s| std::iter::repeat_n(s, channels))
            .collect()
    }

    fn decode(wav: &[u8]) -> (hound::WavSpec, Vec<i16>) {
        let mut reader = WavReader::new(Cursor::new(wav)).unwrap();
        let samples = reader.samples::<i16>().map(Result::unwrap).collect();
        (reader.spec(), samples)
    }

    fn encode(samples: Vec<f32>, channels: u16, sample_rate: u32) -> (hound::WavSpec, Vec<i16>) {
        let audio = CapturedAudio {
            samples,
            channels,
            sample_rate,
        };
        decode(&encode_upload_wav(&audio).unwrap())
    }

    fn zero_crossings(samples: &[i16]) -> usize {
        samples
            .windows(2)
            .filter(|pair| (pair[0] < 0) != (pair[1] < 0))
            .count()
    }

    fn rms(samples: &[i16]) -> f64 {
        let sum: f64 = samples
            .iter()
            .map(|&s| (f64::from(s) / f64::from(i16::MAX)).powi(2))
            .sum();
        (sum / samples.len() as f64).sqrt()
    }

    #[test]
    fn test_upload_is_mono_16khz_16bit_pcm() {
        let (spec, _) = encode(vec![0.0; 960], 2, 48_000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, UPLOAD_SAMPLE_RATE);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    }

    #[test]
    fn test_48khz_stereo_sine_keeps_its_duration_and_frequency() {
        let tone = sine(440.0, 48_000, 1.0, 0.5);
        let (_, samples) = encode(interleave(&tone, 2), 2, 48_000);

        assert!(samples.len().abs_diff(16_000) <= 1, "{}", samples.len());
        // 440 Hz crosses zero 880 times a second
        let crossings = zero_crossings(&samples);
        assert!(crossings.abs_diff(880) <= 2, "{} zero crossings", crossings);
        // The level of a tone in the speech band does not change
        let expected_rms = 0.5 / 2f64.sqrt();
        assert!((rms(&samples) - expected_rms).abs() < 0.01);
    }

    #[test]
    fn test_44_1khz_sine_keeps_its_duration_and_frequency() {
        let tone = sine(1000.0, 44_100, 1.5, 0.5);
        let (_, samples) = encode(tone, 1, 44_100);

        assert!(samples.len().abs_diff(24_000) <= 1, "{}", samples.len());
        let crossings = zero_crossings(&samples);
        assert!(
            crossings.abs_diff(3000) <= 2,
            "{} zero crossings",
            crossings
        );
    }

    #[test]
    fn test_resampled_sine_matches_the_ideal_samples() {
        // 44.1 kHz to 16 kHz: the output times fall between input samples
        let resampled = resample(&sine(1000.0, 44_100, 0.5, 0.5), 44_100, UPLOAD_SAMPLE_RATE);
        let ideal = sine(1000.0, UPLOAD_SAMPLE_RATE, 0.5, 0.5);
        assert_eq!(resampled.len(), ideal.len());
        // The filter treats samples before the start and after the end as
        // silence, so skip the edges.
        let edge = 100;
        let error = resampled[edge..ideal.len() - edge]
            .iter()
            .zip(&ideal[edge..ideal.len() - edge])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 0.01, "max error {}", error);
    }

    #[test]
    fn test_frequencies_above_8khz_do_not_fold_into_the_speech_band() {
        // Without a low-pass filter, 12 kHz sampled at 16 kHz is a 4 kHz tone
        // at full level.
        let tone = sine(12_000.0, 48_000, 0.5, 0.5);
        let (_, samples) = encode(tone, 1, 48_000);
        assert!(rms(&samples) < 0.01, "rms {}", rms(&samples));
    }

    #[test]
    fn test_channels_are_averaged_for_odd_channel_counts() {
        // Three channels at the upload rate, so only the downmix changes them
        let frames = [[0.3, 0.6, 0.0], [-0.9, 0.0, 0.0]];
        let (_, samples) = encode(frames.concat(), 3, UPLOAD_SAMPLE_RATE);
        assert_eq!(samples, vec![to_i16(0.3), to_i16(-0.3)]);
    }

    #[test]
    fn test_16khz_mono_is_not_resampled() {
        let tone = sine(440.0, UPLOAD_SAMPLE_RATE, 0.1, 0.5);
        let (_, samples) = encode(tone.clone(), 1, UPLOAD_SAMPLE_RATE);
        let expected: Vec<i16> = tone.iter().map(|&s| to_i16(s)).collect();
        assert_eq!(samples, expected);
    }

    #[test]
    fn test_samples_outside_the_range_clip_instead_of_wrapping() {
        let (_, samples) = encode(vec![1.5, -2.0, 1.0, -1.0, 0.5], 1, UPLOAD_SAMPLE_RATE);
        assert_eq!(
            samples,
            vec![i16::MAX, -i16::MAX, i16::MAX, -i16::MAX, 16384]
        );
    }

    #[test]
    fn test_empty_recording_gives_an_empty_wav() {
        let (_, samples) = encode(Vec::new(), 2, 48_000);
        assert!(samples.is_empty());
    }
}
