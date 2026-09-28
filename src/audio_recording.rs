use crate::app_state::{AppState, AudioParams};
use crate::recorder::{peak_level, Recorder, RecorderEvent, RecorderSettings, RecorderStatus};
use crate::types::{AudioEvent, CapturedAudio};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use std::error::Error;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;

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

/// Queues events for the processing loop without blocking.
struct EventQueue {
    tx: mpsc::Sender<AudioEvent>,
    /// Events that did not fit in the channel and are not reported yet
    dropped: u32,
}

impl EventQueue {
    /// Queue an event, or count it as dropped if the channel is full.
    fn send(&mut self, event: AudioEvent) {
        if self.tx.try_send(event).is_err() {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    /// Report the dropped events if the channel has room again.
    fn report_dropped(&mut self) {
        if self.dropped > 0
            && self
                .tx
                .try_send(AudioEvent::EventsDropped(self.dropped))
                .is_ok()
        {
            self.dropped = 0;
        }
    }
}

/// Handles the input buffers of one stream, converted to f32.
///
/// This runs on the audio thread, which can be real time. It only updates
/// atomics, copies samples and queues events with `try_send`. Logging and
/// encoding happen on the processing side when it receives the events.
struct InputHandler {
    shared: SharedAudioState,
    recorder: Recorder,
    events: EventQueue,
    channels: u16,
    sample_rate: u32,
}

impl InputHandler {
    fn new(
        shared: SharedAudioState,
        tx: mpsc::Sender<AudioEvent>,
        channels: u16,
        sample_rate: u32,
        now: Instant,
    ) -> Self {
        Self {
            shared,
            recorder: Recorder::new(now),
            events: EventQueue { tx, dropped: 0 },
            channels,
            sample_rate,
        }
    }

    fn process(&mut self, data: &[f32], now: Instant) {
        self.events.report_dropped();
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
        let Self {
            recorder,
            events,
            channels,
            sample_rate,
            ..
        } = self;
        recorder.process(data, &settings, now, |event| match event {
            RecorderEvent::Started => events.send(AudioEvent::StartRecording),
            RecorderEvent::Ended(samples) => {
                if !samples.is_empty() {
                    events.send(AudioEvent::AudioData(CapturedAudio {
                        samples,
                        channels: *channels,
                        sample_rate: *sample_rate,
                    }));
                }
                events.send(AudioEvent::StopRecording);
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
            handler.process(&samples, Instant::now());
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
) -> Result<(Stream, AudioStreamInfo), Box<dyn Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let device_config = device.default_input_config()?;

    let sample_format = device_config.sample_format();
    let stream_info = AudioStreamInfo {
        sample_rate: device_config.sample_rate().0,
        channels: device_config.channels(),
    };

    let handler = InputHandler::new(
        shared,
        tx,
        stream_info.channels,
        stream_info.sample_rate,
        Instant::now(),
    );
    let config = device_config.config();
    let stream: Stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(&device, &config, handler)?,
        cpal::SampleFormat::I16 => build_input_stream::<i16>(&device, &config, handler)?,
        _ => return Err(format!("Unsupported sample format: {:?}", sample_format).into()),
    };

    stream.play()?;

    Ok((stream, stream_info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::AppCommand;
    use crate::config::{AudioConfig, Config};
    use crate::types::CapturedAudio;
    use std::time::Duration;

    const LOUD: [f32; 4] = [0.0, 0.5, -0.5, 0.0];
    const QUIET: [f32; 4] = [0.0, 0.05, -0.05, 0.0];

    struct Setup {
        app_state: AppState,
        handler: InputHandler,
        rx: mpsc::Receiver<AudioEvent>,
        now: Instant,
        // Keep the command and log channels open
        _channels: (
            mpsc::Receiver<AppCommand>,
            mpsc::Receiver<crate::app_state::LogEntry>,
        ),
    }

    impl Setup {
        /// A stereo 48 kHz handler whose recording ends after two silent
        /// buffers, with events queued in a channel of `capacity`.
        fn new(capacity: usize) -> Self {
            let (cmd_tx, cmd_rx) = mpsc::channel(1);
            let (log_tx, log_rx) = mpsc::channel(10);
            let app_state = AppState::new(Config::default(), cmd_tx, log_tx);
            app_state.audio_params.update(&AudioConfig {
                silence_threshold: 2,
                noise_gate_threshold: 0.1,
                noise_gate_hold_time: 0.0,
                min_transcription_duration: 0.0,
            });
            let (tx, rx) = mpsc::channel(capacity);
            let now = Instant::now();
            let handler = InputHandler::new(SharedAudioState::new(&app_state), tx, 2, 48_000, now);
            Setup {
                app_state,
                handler,
                rx,
                now,
                _channels: (cmd_rx, log_rx),
            }
        }

        fn feed(&mut self, data: &[f32]) {
            self.now += Duration::from_millis(10);
            self.handler.process(data, self.now);
        }

        fn events(&mut self) -> Vec<AudioEvent> {
            std::iter::from_fn(|| self.rx.try_recv().ok()).collect()
        }
    }

    #[test]
    fn test_a_recording_reaches_the_processing_side_as_raw_samples() {
        let mut s = Setup::new(10);
        s.feed(&LOUD);
        s.feed(&QUIET);
        s.feed(&QUIET);
        assert_eq!(
            s.events(),
            vec![
                AudioEvent::StartRecording,
                AudioEvent::AudioData(CapturedAudio {
                    samples: [LOUD, QUIET].concat(),
                    channels: 2,
                    sample_rate: 48_000,
                }),
                AudioEvent::StopRecording,
            ]
        );
    }

    #[test]
    fn test_recording_state_is_published_for_the_gui() {
        let mut s = Setup::new(10);
        s.feed(&LOUD);
        assert!(s.app_state.is_recording.load(Ordering::Relaxed));
        assert!(s.app_state.noise_gate_active.load(Ordering::Relaxed));
        s.feed(&QUIET);
        assert_eq!(s.app_state.silent_frames.load(Ordering::Relaxed), 1);
        assert!(!s.app_state.noise_gate_active.load(Ordering::Relaxed));
        let level = f32::from_bits(s.app_state.current_audio_level.load(Ordering::Relaxed));
        assert_eq!(level, 0.05);
    }

    #[test]
    fn test_test_mode_diverts_the_samples_and_sends_nothing() {
        let mut s = Setup::new(10);
        s.app_state.test_mode_active.store(true, Ordering::Relaxed);
        s.feed(&LOUD);
        s.feed(&QUIET);
        assert!(s.events().is_empty());
        assert!(!s.app_state.is_recording.load(Ordering::Relaxed));
        let buffer = s.app_state.test_recording_buffer.lock().unwrap().clone();
        assert_eq!(buffer, [LOUD, QUIET].concat());
        let level = f32::from_bits(s.app_state.current_audio_level.load(Ordering::Relaxed));
        assert_eq!(level, 0.05);
    }

    #[test]
    fn test_events_lost_on_a_full_queue_are_reported_once_there_is_room() {
        let mut s = Setup::new(1);
        s.feed(&LOUD); // StartRecording fills the queue
        s.feed(&QUIET);
        s.feed(&QUIET); // AudioData and StopRecording do not fit
        assert_eq!(s.events(), vec![AudioEvent::StartRecording]);
        s.feed(&QUIET);
        assert_eq!(s.events(), vec![AudioEvent::EventsDropped(2)]);
        s.feed(&QUIET);
        assert!(s.events().is_empty());
    }
}
