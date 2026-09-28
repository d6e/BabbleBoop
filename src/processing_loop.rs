//! Parts of the processing loop in `main.rs` that can be tested without an
//! audio device.

use crate::app_state::{AppState, Logger};
use crate::audio_playback::convert_for_output;
use crate::config::Config;
use crate::models;
use crate::price_estimator::PriceEstimator;
use crate::rate_limiter::RateLimiter;
use crate::recorder::MAX_RECORDING;
use crate::recording_manager::RecordingManager;
use crate::shutdown::Shutdown;
use crate::types::{AudioEvent, CapturedAudio};
use crate::typing_indicator::TypingIndicator;
use crate::upload_audio::encode_upload_wav;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

/// Directory for saved recordings when `keep_audio_files` is on.
const RECORDINGS_DIR: &str = "recordings";

/// Longest test recording. The Stop button in the GUI ends it earlier.
pub const TEST_RECORDING_LIMIT: Duration = Duration::from_secs(30);

/// Number of interleaved samples in `TEST_RECORDING_LIMIT`.
fn test_recording_samples(channels: u16, sample_rate: u32) -> usize {
    TEST_RECORDING_LIMIT.as_secs() as usize * sample_rate as usize * usize::from(channels)
}

/// The test microphone recording. While it runs, the audio callback copies
/// the input into the test buffer instead of the recorder. The Stop button
/// in the GUI or `TEST_RECORDING_LIMIT` ends it.
pub struct TestRecording {
    active: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    channels: u16,
    sample_rate: u32,
    /// When the running test recording reaches the limit
    deadline: Option<Instant>,
}

impl TestRecording {
    /// Test recordings of the input stream with this format.
    pub fn new(app_state: &AppState, channels: u16, sample_rate: u32) -> Self {
        Self {
            active: Arc::clone(&app_state.test_mode_active),
            buffer: Arc::clone(&app_state.test_recording_buffer),
            channels,
            sample_rate,
            deadline: None,
        }
    }

    /// Start a test recording. A test recording that runs starts again.
    pub fn start(&mut self) {
        // The callback adds samples only up to this capacity
        let reserved = Vec::with_capacity(test_recording_samples(self.channels, self.sample_rate));
        let previous = std::mem::replace(&mut *self.lock_buffer(), reserved);
        drop(previous);
        self.deadline = Some(Instant::now() + TEST_RECORDING_LIMIT);
        self.active.store(true, Ordering::SeqCst);
    }

    /// Stop the test recording and return what it recorded. Returns `None`
    /// if no test recording runs.
    pub fn stop(&mut self) -> Option<CapturedAudio> {
        self.deadline.take()?;
        self.active.store(false, Ordering::SeqCst);
        // Leaves a buffer with no capacity, so the callback adds nothing
        // if it still sees test mode on.
        let samples = std::mem::take(&mut *self.lock_buffer());
        Some(CapturedAudio {
            samples,
            channels: self.channels,
            sample_rate: self.sample_rate,
        })
    }

    /// Resolves when the running test recording reaches the limit. Never
    /// resolves while no test recording runs.
    pub async fn limit_reached(&self) {
        match self.deadline {
            Some(deadline) => tokio::time::sleep_until(deadline).await,
            None => std::future::pending().await,
        }
    }

    /// The callback only copies samples while it holds the lock, so a
    /// poisoned lock still holds a valid buffer.
    fn lock_buffer(&self) -> MutexGuard<'_, Vec<f32>> {
        self.buffer.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Processing loop state that depends on the settings.
pub struct ProcessingServices {
    pub rate_limiter: RateLimiter,
    pub price_estimator: PriceEstimator,
    pub recording_manager: Option<RecordingManager>,
}

impl ProcessingServices {
    pub fn new(config: &Config, logger: &Logger) -> Self {
        log_model_warnings(config, logger);
        Self {
            rate_limiter: RateLimiter::new(config.rate_limit.requests_per_minute),
            price_estimator: PriceEstimator::new(
                &config.openai.model,
                &config.openai.transcription_model,
            ),
            recording_manager: recording_manager(config),
        }
    }

    /// Apply settings saved in the GUI.
    pub fn apply_config(&mut self, config: &Config, logger: &Logger) {
        // Keep the requests already counted; a new limiter would reset them.
        self.rate_limiter
            .set_max_requests(config.rate_limit.requests_per_minute);
        self.price_estimator
            .set_models(&config.openai.model, &config.openai.transcription_model);
        self.recording_manager = recording_manager(config);
        log_model_warnings(config, logger);
    }
}

/// Tell the user when a model is scheduled to shut down, or when the cost
/// display cannot be accurate. The GUI accepts any model name.
fn log_model_warnings(config: &Config, logger: &Logger) {
    let (model, transcription_model) = (&config.openai.model, &config.openai.transcription_model);
    for warning in models::shutdown_warnings(model, transcription_model)
        .into_iter()
        .chain(PriceEstimator::unknown_pricing(model, transcription_model))
    {
        logger.info(warning);
    }
}

fn recording_manager(config: &Config) -> Option<RecordingManager> {
    config
        .keep_audio_files
        .then(|| RecordingManager::new(PathBuf::from(RECORDINGS_DIR), config.max_audio_files))
}

/// Handle the translation toggle from the GUI.
///
/// Disabling turns the typing indicator off. While translation is off the
/// loop ignores audio events, so the StopRecording of an utterance that
/// started before would not turn it off. The GUI stores `enabled` before it
/// sends this command, so no StartRecording handled after this can turn the
/// indicator on again.
pub async fn apply_enabled(enabled: bool, typing_indicator: &TypingIndicator, logger: &Logger) {
    if enabled {
        logger.info("Translation enabled");
    } else {
        typing_indicator.stop_typing().await;
        logger.info("Translation disabled");
    }
}

/// The events from the audio callback, as the processing loop receives
/// them.
///
/// An input error or the end of the input also ends the recording that
/// runs: a StopRecording follows, so the loop turns the typing indicator
/// off. The callback does not send one, as it may not run again:
/// - After a panic, `PanicGuard` in `audio_recording.rs` runs nothing of
///   the callback.
/// - On WASAPI, cpal 0.15.3 ends the stream thread after the first stream
///   error (`src/host/wasapi/stream.rs` lines 343 to 360 and 388 to 390),
///   and the channel closes.
/// - On CoreAudio, cpal pauses the stream when the device is disconnected
///   and then reports the error (`src/host/coreaudio/macos/mod.rs` lines
///   465 to 468).
/// - On ALSA, cpal reports an error and polls the device again
///   (`src/host/alsa/mod.rs` lines 586 to 596), so while the error does
///   not clear, cpal calls the error callback and not the data callback.
///
/// If the input goes on after a stream error, the indicator is off for the
/// rest of the recording, or until a part of it is processed.
pub struct AudioEvents {
    /// `None` after every sender is gone
    rx: Option<mpsc::Receiver<AudioEvent>>,
    /// An input error was returned, and the StopRecording that follows it
    /// was not
    stop_pending: bool,
    logger: Logger,
}

impl AudioEvents {
    pub fn new(rx: mpsc::Receiver<AudioEvent>, logger: Logger) -> Self {
        Self {
            rx: Some(rx),
            stop_pending: false,
            logger,
        }
    }

    /// The next event, after its line in the activity log. When every
    /// sender is gone, logs an error once, returns StopRecording and then
    /// never resolves. The senders go when the audio thread of cpal ends,
    /// which on WASAPI follows the first stream error, possibly before its
    /// report fits in the channel.
    ///
    /// Cancel safe: `mpsc::Receiver::recv` is (tokio 1.48.0,
    /// `src/sync/mpsc/bounded.rs` lines 199 to 204), and this future
    /// changes the state only in the poll that returns the event.
    pub async fn recv(&mut self) -> AudioEvent {
        if std::mem::take(&mut self.stop_pending) {
            return AudioEvent::StopRecording;
        }
        if let Some(rx) = &mut self.rx {
            if let Some(event) = rx.recv().await {
                log_audio_event(&event, &self.logger);
                self.stop_pending = matches!(event, AudioEvent::InputError(_));
                return event;
            }
            self.rx = None;
            self.logger
                .error("Audio input stopped. Restart BabbleBoop to record again.");
            return AudioEvent::StopRecording;
        }
        std::future::pending().await
    }
}

/// Write the activity log line for an event from the audio callback. The
/// callback runs on the audio thread and does not log itself.
pub fn log_audio_event(event: &AudioEvent, logger: &Logger) {
    match event {
        AudioEvent::StartRecording => logger.info("Sound detected, recording..."),
        AudioEvent::AudioPart(_) => logger.info(format!(
            "Recording reached {} s, processing it while recording goes on...",
            MAX_RECORDING.as_secs()
        )),
        AudioEvent::AudioData(..) => logger.info("Silence detected, processing..."),
        AudioEvent::StopRecording => {}
        AudioEvent::EventsDropped(count) => logger.error(format!(
            "Lost {} audio events because the processing queue was full",
            count
        )),
        AudioEvent::InputError(message) => logger.error(message.as_str()),
    }
}

/// Encode a recording for upload on a blocking thread. The processing loop
/// runs in `block_on` on the processing thread, so encoding in the loop
/// would stop the loop, and its check for shutdown, until the encoding ends.
pub async fn encode_for_upload(audio: CapturedAudio) -> Result<Vec<u8>, String> {
    match tokio::task::spawn_blocking(move || encode_upload_wav(&audio)).await {
        Ok(Ok(wav)) => Ok(wav),
        Ok(Err(e)) => Err(format!("cannot encode the recording: {}", e)),
        Err(e) => Err(format!("encoding the recording failed: {}", e)),
    }
}

/// Longest wait for the audio input stream to start. Opening a device
/// usually takes less than a second, but a driver can take a few seconds
/// (for example a Bluetooth headset that changes to its microphone
/// profile). A driver that never returns gives this error instead of an
/// application that records nothing and shows no reason.
pub const AUDIO_START_TIMEOUT: Duration = Duration::from_secs(15);

/// Wait until the audio thread reports whether the input stream started.
/// Returns the stream format, or `None` if shutdown is requested first.
/// The error is the startup error for the activity log.
///
/// The wait does not block the runtime, and it ends on shutdown, so a
/// driver that never returns cannot stop the application from closing.
pub async fn wait_for_audio_start<T>(
    started: oneshot::Receiver<Result<T, String>>,
    shutdown: &Shutdown,
    timeout: Duration,
) -> Result<Option<T>, String> {
    match shutdown
        .run_until(tokio::time::timeout(timeout, started))
        .await
    {
        None => Ok(None),
        Some(Ok(Ok(Ok(info)))) => Ok(Some(info)),
        Some(Ok(Ok(Err(e)))) => Err(format!("cannot start audio input: {}", e)),
        // The audio thread panicked; its own log line gives the reason
        Some(Ok(Err(_))) => Err("cannot start audio input: the audio thread stopped".to_string()),
        Some(Err(_)) => Err(format!(
            "cannot start audio input: timed out after {} s",
            timeout.as_secs()
        )),
    }
}

/// Keep the audio input stream until shutdown is requested or the
/// processing thread ends. The processing thread also ends when it stops
/// waiting for the stream to start, so no stream records that nothing
/// reads.
pub fn hold_audio_stream<S>(stream: S, app_state: &AppState) {
    while !app_state.is_shutdown_requested() && !app_state.is_processing_stopped() {
        std::thread::sleep(Duration::from_millis(100));
    }
    drop(stream);
}

/// Convert a test recording for the output device on a blocking thread.
/// Resampling 30 s of 48 kHz stereo to 44.1 kHz takes about 0.1 s in a
/// release build and 3 s in a debug build.
pub async fn convert_for_playback(
    audio: CapturedAudio,
    channels: u16,
    sample_rate: u32,
) -> Result<Vec<f32>, String> {
    tokio::task::spawn_blocking(move || convert_for_output(&audio, channels, sample_rate))
        .await
        .map_err(|e| format!("converting the test recording failed: {}", e))
}
