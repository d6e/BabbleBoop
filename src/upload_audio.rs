//! Converts a recording into the WAV file that is uploaded for
//! transcription: mono, 16 kHz, 16 bit PCM.
//!
//! Whisper resamples all input to 16 kHz (Radford et al. 2022, "Robust
//! Speech Recognition via Large-Scale Weak Supervision", section 2.2), and
//! speech carries little above 8 kHz, so a higher rate or more channels
//! mostly make the upload larger. One minute is about 1.9 MB in this
//! format, against 23 MB as 32 bit float stereo at 48 kHz.

use crate::resample::resample;
use crate::types::CapturedAudio;
use hound::WavWriter;
use std::io::Cursor;

/// Sample rate of the uploaded WAV.
pub const UPLOAD_SAMPLE_RATE: u32 = 16_000;

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
