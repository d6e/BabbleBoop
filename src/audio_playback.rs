use crate::resample::resample;
use crate::types::CapturedAudio;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use std::error::Error;

/// The default output device and the stream format it plays.
pub struct AudioOutput {
    device: cpal::Device,
    config: cpal::StreamConfig,
    sample_format: SampleFormat,
}

impl AudioOutput {
    /// Open the default output device in its default format.
    pub fn open_default() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let device = cpal::default_host()
            .default_output_device()
            .ok_or("No output device available")?;
        let supported_config = device.default_output_config()?;
        Ok(Self {
            device,
            sample_format: supported_config.sample_format(),
            config: supported_config.into(),
        })
    }

    pub fn channels(&self) -> u16 {
        self.config.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate.0
    }

    /// Play interleaved samples in the channels and rate of this output
    /// (see `convert_for_output`). Returns a Stream that must be kept alive
    /// until playback is complete.
    pub fn play(&self, samples: Vec<f32>) -> Result<Stream, Box<dyn Error + Send + Sync>> {
        let stream = match self.sample_format {
            SampleFormat::F32 => build_output_stream::<f32>(&self.device, &self.config, samples)?,
            SampleFormat::I16 => build_output_stream::<i16>(&self.device, &self.config, samples)?,
            format => return Err(format!("Unsupported sample format: {:?}", format).into()),
        };
        stream.play()?;
        Ok(stream)
    }
}

/// Convert a recording to interleaved samples with `channels` channels at
/// `sample_rate`, so that it plays at its own speed and pitch. Output
/// channel n plays recording channel n, wrapping around when the output
/// has more channels. Audio with no channels, or an output with none,
/// gives no samples.
pub fn convert_for_output(audio: &CapturedAudio, channels: u16, sample_rate: u32) -> Vec<f32> {
    let (in_channels, out_channels) = (usize::from(audio.channels), usize::from(channels));
    // Only the recording channels that the output plays. None if either
    // has no channels, so a 0 channel count is never a divisor.
    let resampled: Vec<Vec<f32>> = (0..in_channels.min(out_channels))
        .map(|ch| {
            let channel: Vec<f32> = audio
                .samples
                .chunks_exact(in_channels)
                .map(|frame| frame[ch])
                .collect();
            resample(&channel, audio.sample_rate, sample_rate)
        })
        .collect();
    let frames = resampled.first().map_or(0, Vec::len);
    (0..frames)
        .flat_map(|frame| {
            let resampled = &resampled;
            (0..out_channels).map(move |ch| resampled[ch % resampled.len()][frame])
        })
        .collect()
}

/// Sample formats the output stream accepts.
trait OutputSample: cpal::SizedSample {
    fn from_f32(sample: f32) -> Self;
}

impl OutputSample for f32 {
    fn from_f32(sample: f32) -> Self {
        sample
    }
}

impl OutputSample for i16 {
    fn from_f32(sample: f32) -> Self {
        (sample * i16::MAX as f32) as i16
    }
}

fn build_output_stream<T: OutputSample>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Vec<f32>,
) -> Result<Stream, cpal::BuildStreamError> {
    let mut position = 0;
    device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            write_samples(output, &samples, &mut position);
        },
        |err| eprintln!("Playback error: {}", err),
        None,
    )
}

/// Fill `output` with the samples from `position` on, and with silence
/// after the last one.
fn write_samples<T: OutputSample>(output: &mut [T], samples: &[f32], position: &mut usize) {
    let remaining = samples.get(*position..).unwrap_or_default();
    for (index, out) in output.iter_mut().enumerate() {
        *out = T::from_f32(remaining.get(index).copied().unwrap_or(0.0));
    }
    *position += output.len().min(remaining.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio(samples: &[f32], channels: u16) -> CapturedAudio {
        CapturedAudio {
            samples: samples.to_vec(),
            channels,
            sample_rate: 48_000,
        }
    }

    #[test]
    fn test_mono_recording_plays_on_every_output_channel() {
        let converted = convert_for_output(&audio(&[0.1, 0.2], 1), 2, 48_000);
        assert_eq!(converted, vec![0.1, 0.1, 0.2, 0.2]);
    }

    #[test]
    fn test_output_channels_wrap_around_the_recording_channels() {
        let stereo = audio(&[0.1, 0.2, 0.3, 0.4], 2);
        assert_eq!(convert_for_output(&stereo, 2, 48_000), stereo.samples);
        assert_eq!(
            convert_for_output(&stereo, 3, 48_000),
            vec![0.1, 0.2, 0.1, 0.3, 0.4, 0.3]
        );
        assert_eq!(convert_for_output(&stereo, 1, 48_000), vec![0.1, 0.3]);
    }

    fn sine(frequency: f32, sample_rate: u32, seconds: f32) -> Vec<f32> {
        let len = (sample_rate as f32 * seconds) as usize;
        (0..len)
            .map(|n| {
                let t = n as f32 / sample_rate as f32;
                0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect()
    }

    fn zero_crossings(samples: impl Iterator<Item = f32>) -> usize {
        let signs: Vec<bool> = samples.map(|s| s < 0.0).collect();
        signs.windows(2).filter(|pair| pair[0] != pair[1]).count()
    }

    /// Play a 440 Hz stereo tone of 1 s recorded at `from_rate` on a
    /// stereo output at `to_rate`, and check that it still lasts 1 s at
    /// 440 Hz on both channels.
    fn assert_tone_keeps_its_duration_and_pitch(from_rate: u32, to_rate: u32) {
        let tone = sine(440.0, from_rate, 1.0);
        let recording = CapturedAudio {
            samples: tone.iter().flat_map(|&s| [s, s]).collect(),
            channels: 2,
            sample_rate: from_rate,
        };
        let played = convert_for_output(&recording, 2, to_rate);

        let frames = played.len() / 2;
        assert!(
            frames.abs_diff(to_rate as usize) <= 1,
            "{} frames at {} Hz",
            frames,
            to_rate
        );
        for channel in 0..2 {
            // 440 Hz crosses zero 880 times a second
            let crossings = zero_crossings(played.iter().skip(channel).step_by(2).copied());
            assert!(crossings.abs_diff(880) <= 2, "{} zero crossings", crossings);
        }
    }

    #[test]
    fn test_playback_on_a_44_1khz_output_keeps_the_pitch_of_48khz_audio() {
        assert_tone_keeps_its_duration_and_pitch(48_000, 44_100);
    }

    #[test]
    fn test_playback_on_a_48khz_output_keeps_the_pitch_of_44_1khz_audio() {
        assert_tone_keeps_its_duration_and_pitch(44_100, 48_000);
    }

    #[test]
    fn test_audio_without_channels_plays_nothing() {
        assert!(convert_for_output(&audio(&[0.1, 0.2], 0), 2, 48_000).is_empty());
        assert!(convert_for_output(&audio(&[0.1, 0.2], 2), 0, 48_000).is_empty());
    }

    #[test]
    fn test_playback_goes_on_across_callbacks_and_ends_in_silence() {
        let samples = [0.1, 0.2, 0.3, 0.4, 0.5];
        let mut position = 0;
        let mut first = [9.0f32; 3];
        write_samples(&mut first, &samples, &mut position);
        let mut second = [9.0f32; 3];
        write_samples(&mut second, &samples, &mut position);
        let mut third = [9.0f32; 3];
        write_samples(&mut third, &samples, &mut position);

        assert_eq!(first, [0.1, 0.2, 0.3]);
        assert_eq!(second, [0.4, 0.5, 0.0]);
        assert_eq!(third, [0.0; 3]);
    }

    #[test]
    fn test_16_bit_output_scales_the_samples() {
        let mut position = 0;
        let mut output = [0i16; 3];
        write_samples(&mut output, &[1.0, -0.5, 0.0], &mut position);
        assert_eq!(output, [i16::MAX, -(i16::MAX / 2), 0]);
    }
}
