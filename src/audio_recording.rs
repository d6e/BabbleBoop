use crate::app_state::{AppState, AudioParams, Logger};
use crate::recorder::{peak_level, Recorder, RecorderEvent, RecorderSettings, RecorderStatus};
use crate::types::AudioEvent;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use hound::WavWriter;
use std::error::Error;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

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

/// State that the audio callback shares with the GUI and the processing loop.
pub struct SharedAudioState {
    pub audio_params: Arc<AudioParams>,
    /// Peak level of the last buffer, for the level meter
    pub audio_level: Arc<AtomicU32>,
    pub test_mode_active: Arc<AtomicBool>,
    pub test_recording_buffer: Arc<Mutex<Vec<f32>>>,
    pub is_recording: Arc<AtomicBool>,
    pub silent_frames: Arc<AtomicU32>,
    pub noise_gate_active: Arc<AtomicBool>,
    pub noise_gate_hold_remaining: Arc<AtomicU32>,
    pub recording_duration: Arc<AtomicU32>,
}

impl SharedAudioState {
    pub fn new(app_state: &AppState) -> Self {
        Self {
            audio_params: Arc::clone(&app_state.audio_params),
            audio_level: Arc::clone(&app_state.current_audio_level),
            test_mode_active: Arc::clone(&app_state.test_mode_active),
            test_recording_buffer: Arc::clone(&app_state.test_recording_buffer),
            is_recording: Arc::clone(&app_state.is_recording),
            silent_frames: Arc::clone(&app_state.silent_frames),
            noise_gate_active: Arc::clone(&app_state.noise_gate_active),
            noise_gate_hold_remaining: Arc::clone(&app_state.noise_gate_hold_remaining),
            recording_duration: Arc::clone(&app_state.recording_duration),
        }
    }

    fn recorder_settings(&self) -> RecorderSettings {
        RecorderSettings {
            noise_gate_threshold: self.audio_params.get_noise_gate_threshold(),
            noise_gate_hold_time: self.audio_params.get_noise_gate_hold_time(),
            silence_threshold: self.audio_params.get_silence_threshold(),
        }
    }

    fn publish(&self, status: &RecorderStatus) {
        self.noise_gate_active
            .store(status.gate_open, Ordering::Relaxed);
        self.noise_gate_hold_remaining
            .store(status.hold_remaining.to_bits(), Ordering::Relaxed);
        self.is_recording
            .store(status.is_recording, Ordering::Relaxed);
        self.silent_frames
            .store(status.silent_frames, Ordering::Relaxed);
        self.recording_duration
            .store(status.recording_duration.to_bits(), Ordering::Relaxed);
    }
}

/// Handles the input buffers of one stream, converted to f32.
struct InputHandler {
    shared: SharedAudioState,
    recorder: Recorder,
    tx: mpsc::Sender<AudioEvent>,
    logger: Logger,
    channels: usize,
    sample_rate: f32,
}

impl InputHandler {
    fn process(&mut self, data: &[f32]) {
        self.shared
            .audio_level
            .store(peak_level(data).to_bits(), Ordering::Relaxed);

        // If test mode is active, write raw samples to the test buffer and skip normal processing
        if self.shared.test_mode_active.load(Ordering::Relaxed) {
            if let Ok(mut buffer) = self.shared.test_recording_buffer.lock() {
                buffer.extend_from_slice(data);
            }
            return;
        }

        let settings = self.shared.recorder_settings();
        let now = Instant::now();
        let Self {
            recorder,
            tx,
            logger,
            channels,
            sample_rate,
            ..
        } = self;
        recorder.process(data, &settings, now, |event| match event {
            RecorderEvent::Started => {
                logger.info("Sound detected, recording...");
                if let Err(e) = tx.try_send(AudioEvent::StartRecording) {
                    logger.error(format!("Failed to send StartRecording: {}", e));
                }
            }
            RecorderEvent::Ended(samples) => {
                if !samples.is_empty() {
                    logger.info("Silence detected, processing...");
                    if let Some(wav_buffer) = encode_wav_buffer(&samples, *channels, *sample_rate) {
                        if let Err(e) = tx.try_send(AudioEvent::AudioData(wav_buffer)) {
                            logger.error(format!("Failed to send AudioData: {}", e));
                        }
                    }
                }
                let _ = tx.try_send(AudioEvent::StopRecording);
            }
        });
        self.shared.publish(&self.recorder.status(now));
    }
}

/// Sample formats the input stream accepts.
trait InputSample: cpal::SizedSample {
    fn to_f32(self) -> f32;
}

impl InputSample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl InputSample for i16 {
    fn to_f32(self) -> f32 {
        self as f32 / i16::MAX as f32
    }
}

fn build_input_stream<T: InputSample>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut handler: InputHandler,
) -> Result<Stream, cpal::BuildStreamError> {
    let mut samples = Vec::new();
    let err_fn = |err| eprintln!("An error occurred on the audio stream: {}", err);
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            samples.clear();
            samples.extend(data.iter().map(|&s| s.to_f32()));
            handler.process(&samples);
        },
        err_fn,
        None,
    )
}

/// Information about the audio stream configuration
pub struct AudioStreamInfo {
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn start_audio_recording(
    shared: SharedAudioState,
    tx: mpsc::Sender<AudioEvent>,
    logger: Logger,
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

    let handler = InputHandler {
        shared,
        recorder: Recorder::new(Instant::now()),
        tx,
        logger,
        channels,
        sample_rate,
    };
    let config = device_config.config();
    let stream: Stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(&device, &config, handler)?,
        cpal::SampleFormat::I16 => build_input_stream::<i16>(&device, &config, handler)?,
        _ => return Err(format!("Unsupported sample format: {:?}", sample_format).into()),
    };

    stream.play()?;

    Ok((stream, stream_info))
}
