use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
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

    // Parse WAV header to get samples
    let reader = hound::WavReader::new(Cursor::new(&wav_data))?;
    let spec = reader.spec();

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

    let samples = Arc::new(samples);
    let position = Arc::new(AtomicUsize::new(0));

    let samples_clone = Arc::clone(&samples);
    let position_clone = Arc::clone(&position);
    let is_playing_clone = Arc::clone(&is_playing);

    is_playing.store(true, Ordering::SeqCst);

    let stream = device.build_output_stream(
        &supported_config.into(),
        move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut pos = position_clone.load(Ordering::Relaxed);
            let samples_len = samples_clone.len();

            for sample in output.iter_mut() {
                if pos < samples_len {
                    *sample = samples_clone[pos];
                    pos += 1;
                } else {
                    *sample = 0.0;
                }
            }

            position_clone.store(pos, Ordering::Relaxed);

            // Signal playback complete when we've played all samples
            if pos >= samples_len {
                is_playing_clone.store(false, Ordering::SeqCst);
            }
        },
        |err| eprintln!("Playback error: {}", err),
        None,
    )?;

    stream.play()?;
    Ok(stream)
}
