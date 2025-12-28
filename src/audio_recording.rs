use crate::app_state::AudioParams;
use crate::types::AudioEvent;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use hound::WavWriter;
use std::error::Error;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
}

fn encode_wav_buffer(samples: &[f32], channels: usize, sample_rate: f32) -> Option<Vec<u8>> {
    let mut wav_buffer = Vec::new();
    let mut writer = WavWriter::new(
        Cursor::new(&mut wav_buffer),
        hound::WavSpec {
            channels: channels as u16,
            sample_rate: sample_rate as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .ok()?;

    for &sample in samples.iter() {
        if writer.write_sample(sample).is_err() {
            eprintln!("Error writing audio sample");
            return None;
        }
    }

    if writer.finalize().is_err() {
        eprintln!("Error finalizing WAV buffer");
        return None;
    }

    Some(wav_buffer)
}

struct NoiseGate {
    params: Arc<AudioParams>,
    last_active: std::time::Instant,
    is_active: bool,
}

impl NoiseGate {
    fn new(params: Arc<AudioParams>) -> Self {
        NoiseGate {
            params,
            last_active: std::time::Instant::now(),
            is_active: false,
        }
    }

    fn process(&mut self, samples: &[f32]) -> bool {
        // Read threshold and hold_time from atomics for hot reload support
        let threshold = self.params.get_noise_gate_threshold();
        let hold_time = self.params.get_noise_gate_hold_time();
        let max_amplitude = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        if max_amplitude > threshold {
            self.last_active = std::time::Instant::now();
            self.is_active = true;
        } else if self.is_active && self.last_active.elapsed().as_secs_f32() > hold_time {
            self.is_active = false;
        }

        self.is_active
    }
}

fn build_input_stream_f32(
    device: &cpal::Device,
    device_config: cpal::SupportedStreamConfig,
    audio_params: Arc<AudioParams>,
    audio_level: Arc<AtomicU32>,
    test_mode_active: Arc<AtomicBool>,
    test_recording_buffer: Arc<Mutex<Vec<f32>>>,
    tx: mpsc::Sender<AudioEvent>,
    channels: usize,
    sample_rate: f32,
) -> Result<Stream, Box<dyn Error>> {
    let audio_data = Arc::new(Mutex::new(Vec::new()));
    let audio_data_clone = Arc::clone(&audio_data);

    let params_clone = Arc::clone(&audio_params);
    let mut noise_gate = NoiseGate::new(params_clone);

    let mut is_recording = false;
    let mut silent_frames = 0;

    let err_fn = |err| eprintln!("An error occurred on the audio stream: {}", err);

    let stream = device.build_input_stream(
        &device_config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            process_audio_data(
                data,
                &audio_data_clone,
                &mut noise_gate,
                &mut is_recording,
                &mut silent_frames,
                &audio_params,
                &audio_level,
                &test_mode_active,
                &test_recording_buffer,
                &tx,
                channels,
                sample_rate,
            );
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

fn build_input_stream_i16(
    device: &cpal::Device,
    device_config: cpal::SupportedStreamConfig,
    audio_params: Arc<AudioParams>,
    audio_level: Arc<AtomicU32>,
    test_mode_active: Arc<AtomicBool>,
    test_recording_buffer: Arc<Mutex<Vec<f32>>>,
    tx: mpsc::Sender<AudioEvent>,
    channels: usize,
    sample_rate: f32,
) -> Result<Stream, Box<dyn Error>> {
    let audio_data = Arc::new(Mutex::new(Vec::new()));
    let audio_data_clone = Arc::clone(&audio_data);

    let params_clone = Arc::clone(&audio_params);
    let mut noise_gate = NoiseGate::new(params_clone);

    let mut is_recording = false;
    let mut silent_frames = 0;

    let err_fn = |err| eprintln!("An error occurred on the audio stream: {}", err);

    let stream = device.build_input_stream(
        &device_config.into(),
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            let f32_data: Vec<f32> = data.iter().map(|&s| i16_to_f32(s)).collect();
            process_audio_data(
                &f32_data,
                &audio_data_clone,
                &mut noise_gate,
                &mut is_recording,
                &mut silent_frames,
                &audio_params,
                &audio_level,
                &test_mode_active,
                &test_recording_buffer,
                &tx,
                channels,
                sample_rate,
            );
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

#[allow(clippy::too_many_arguments)]
fn process_audio_data(
    data: &[f32],
    audio_data: &Arc<Mutex<Vec<f32>>>,
    noise_gate: &mut NoiseGate,
    is_recording: &mut bool,
    silent_frames: &mut u32,
    audio_params: &Arc<AudioParams>,
    audio_level: &Arc<AtomicU32>,
    test_mode_active: &Arc<AtomicBool>,
    test_recording_buffer: &Arc<Mutex<Vec<f32>>>,
    tx: &mpsc::Sender<AudioEvent>,
    channels: usize,
    sample_rate: f32,
) {
    // Calculate and store the current audio level for the GUI level meter
    let max_amplitude = data.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    audio_level.store(max_amplitude.to_bits(), Ordering::Relaxed);

    // If test mode is active, write raw samples to the test buffer
    if test_mode_active.load(Ordering::Relaxed) {
        if let Ok(mut buffer) = test_recording_buffer.lock() {
            buffer.extend_from_slice(data);
        }
    }

    // Read silence_threshold from atomics for hot reload support
    let silence_threshold = audio_params.get_silence_threshold();

    if noise_gate.process(data) {
        let mut buffer = audio_data.lock().unwrap();

        if !*is_recording {
            *is_recording = true;
            println!("Sound detected. Starting recording...");
            if let Err(e) = tx.try_send(AudioEvent::StartRecording) {
                eprintln!("Warning: Failed to send StartRecording event: {}", e);
            }
        }

        buffer.extend_from_slice(data);
        *silent_frames = 0;
    } else if *is_recording {
        *silent_frames += 1;

        if *silent_frames >= silence_threshold {
            *is_recording = false;
            *silent_frames = 0;

            let mut buffer = audio_data.lock().unwrap();
            if !buffer.is_empty() {
                println!("Silence detected. Stopping recording and processing audio...");
                if let Some(wav_buffer) = encode_wav_buffer(&buffer, channels, sample_rate) {
                    if let Err(e) = tx.try_send(AudioEvent::AudioData(wav_buffer)) {
                        eprintln!("Warning: Failed to send AudioData event: {}", e);
                    }
                }
                buffer.clear();
            }

            if let Err(e) = tx.try_send(AudioEvent::StopRecording) {
                eprintln!("Warning: Failed to send StopRecording event: {}", e);
            }
        } else {
            // Keep recording during short pauses
            let mut buffer = audio_data.lock().unwrap();
            buffer.extend_from_slice(data);
        }
    }
}

/// Information about the audio stream configuration
pub struct AudioStreamInfo {
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn start_audio_recording(
    audio_params: Arc<AudioParams>,
    audio_level: Arc<AtomicU32>,
    test_mode_active: Arc<AtomicBool>,
    test_recording_buffer: Arc<Mutex<Vec<f32>>>,
    tx: mpsc::Sender<AudioEvent>,
) -> Result<(Stream, AudioStreamInfo), Box<dyn Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let device_config = device.default_input_config()?;

    let sample_rate = device_config.sample_rate().0 as f32;
    let channels = device_config.channels() as usize;
    let sample_format = device_config.sample_format();

    let stream_info = AudioStreamInfo {
        sample_rate: sample_rate as u32,
        channels: channels as u16,
    };

    let stream: Stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream_f32(
            &device,
            device_config,
            audio_params,
            audio_level,
            test_mode_active,
            test_recording_buffer,
            tx,
            channels,
            sample_rate,
        )?,
        cpal::SampleFormat::I16 => build_input_stream_i16(
            &device,
            device_config,
            audio_params,
            audio_level,
            test_mode_active,
            test_recording_buffer,
            tx,
            channels,
            sample_rate,
        )?,
        _ => return Err(format!("Unsupported sample format: {:?}", sample_format).into()),
    };

    stream.play()?;

    Ok((stream, stream_info))
}
