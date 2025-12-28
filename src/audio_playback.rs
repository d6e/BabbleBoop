use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use std::error::Error;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Plays a WAV buffer through the default output device.
/// Returns a Stream that must be kept alive until playback is complete.
/// The `is_playing` flag will be set to false when playback finishes.
pub fn play_wav_buffer(
    wav_data: Vec<u8>,
    is_playing: Arc<AtomicBool>,
) -> Result<Stream, Box<dyn Error + Send + Sync>> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or("No output device available")?;
    let supported_config = device.default_output_config()?;
    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();

    println!(
        "  Output device: {} channels, {} Hz, {:?}",
        config.channels, config.sample_rate.0, sample_format
    );

    // Parse WAV header to get samples
    let reader = hound::WavReader::new(Cursor::new(&wav_data))?;
    let spec = reader.spec();
    let wav_channels = spec.channels as usize;

    println!(
        "  WAV source: {} channels, {} Hz",
        spec.channels, spec.sample_rate
    );

    // Convert samples to f32
    let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
        reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect()
    } else {
        reader
            .into_samples::<i16>()
            .filter_map(|s| s.ok())
            .map(|s| s as f32 / i16::MAX as f32)
            .collect()
    };

    println!("  Loaded {} samples for playback", samples.len());

    let samples = Arc::new(samples);
    let position = Arc::new(AtomicUsize::new(0));
    let output_channels = config.channels as usize;

    is_playing.store(true, Ordering::SeqCst);

    let stream = match sample_format {
        SampleFormat::F32 => {
            let samples_clone = Arc::clone(&samples);
            let position_clone = Arc::clone(&position);
            let is_playing_clone = Arc::clone(&is_playing);

            device.build_output_stream(
                &config,
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    write_samples(
                        output,
                        &samples_clone,
                        &position_clone,
                        &is_playing_clone,
                        wav_channels,
                        output_channels,
                    );
                },
                |err| eprintln!("Playback error: {}", err),
                None,
            )?
        }
        SampleFormat::I16 => {
            let samples_clone = Arc::clone(&samples);
            let position_clone = Arc::clone(&position);
            let is_playing_clone = Arc::clone(&is_playing);

            device.build_output_stream(
                &config,
                move |output: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    let mut temp = vec![0.0f32; output.len()];
                    write_samples(
                        &mut temp,
                        &samples_clone,
                        &position_clone,
                        &is_playing_clone,
                        wav_channels,
                        output_channels,
                    );
                    for (out, sample) in output.iter_mut().zip(temp.iter()) {
                        *out = (*sample * i16::MAX as f32) as i16;
                    }
                },
                |err| eprintln!("Playback error: {}", err),
                None,
            )?
        }
        _ => return Err(format!("Unsupported sample format: {:?}", sample_format).into()),
    };

    stream.play()?;
    println!("  Playback started");
    Ok(stream)
}

fn write_samples(
    output: &mut [f32],
    samples: &Arc<Vec<f32>>,
    position: &Arc<AtomicUsize>,
    is_playing: &Arc<AtomicBool>,
    wav_channels: usize,
    output_channels: usize,
) {
    let mut pos = position.load(Ordering::Relaxed);
    let samples_len = samples.len();

    for frame in output.chunks_mut(output_channels) {
        if pos < samples_len {
            // Handle channel conversion
            for (ch, out_sample) in frame.iter_mut().enumerate() {
                // Map output channel to input channel (wrap around if needed)
                let in_ch = ch % wav_channels;
                let sample_idx = pos + in_ch;
                if sample_idx < samples_len {
                    *out_sample = samples[sample_idx];
                } else {
                    *out_sample = 0.0;
                }
            }
            pos += wav_channels;
        } else {
            for out_sample in frame.iter_mut() {
                *out_sample = 0.0;
            }
        }
    }

    position.store(pos, Ordering::Relaxed);

    // Signal playback complete when we've played all samples
    if pos >= samples_len {
        is_playing.store(false, Ordering::SeqCst);
    }
}
