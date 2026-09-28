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

/// Convert a recording to interleaved samples with `channels` channels.
/// Output channel n plays recording channel n, wrapping around when the
/// output has more channels.
pub fn convert_for_output(audio: &CapturedAudio, channels: u16) -> Vec<f32> {
    let in_channels = usize::from(audio.channels);
    audio
        .samples
        .chunks_exact(in_channels)
        .flat_map(|frame| (0..usize::from(channels)).map(move |ch| frame[ch % in_channels]))
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
        let converted = convert_for_output(&audio(&[0.1, 0.2], 1), 2);
        assert_eq!(converted, vec![0.1, 0.1, 0.2, 0.2]);
    }

    #[test]
    fn test_output_channels_wrap_around_the_recording_channels() {
        let stereo = audio(&[0.1, 0.2, 0.3, 0.4], 2);
        assert_eq!(convert_for_output(&stereo, 2), stereo.samples);
        assert_eq!(
            convert_for_output(&stereo, 3),
            vec![0.1, 0.2, 0.1, 0.3, 0.4, 0.3]
        );
        assert_eq!(convert_for_output(&stereo, 1), vec![0.1, 0.3]);
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
