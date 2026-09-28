//! Parts of the processing loop in `main.rs` that can be tested without an
//! audio device.

use crate::app_state::Logger;
use crate::config::Config;
use crate::models;
use crate::price_estimator::PriceEstimator;
use crate::rate_limiter::RateLimiter;
use crate::recorder::MAX_RECORDING;
use crate::recording_manager::RecordingManager;
use crate::types::{AudioEvent, CapturedAudio};
use crate::typing_indicator::TypingIndicator;
use crate::upload_audio::encode_upload_wav;
use std::path::PathBuf;
use std::time::Duration;

/// Directory for saved recordings when `keep_audio_files` is on.
const RECORDINGS_DIR: &str = "recordings";

/// Longest test recording. The Stop button in the GUI ends it earlier.
pub const TEST_RECORDING_LIMIT: Duration = Duration::from_secs(30);

/// Number of interleaved samples in `TEST_RECORDING_LIMIT`.
pub fn test_recording_samples(channels: u16, sample_rate: u32) -> usize {
    TEST_RECORDING_LIMIT.as_secs() as usize * sample_rate as usize * usize::from(channels)
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

/// Write the activity log line for an event from the audio callback. The
/// callback runs on the audio thread and does not log itself.
pub fn log_audio_event(event: &AudioEvent, logger: &Logger) {
    match event {
        AudioEvent::StartRecording => logger.info("Sound detected, recording..."),
        AudioEvent::AudioPart(_) => logger.info(format!(
            "Recording reached {} s, processing it while recording goes on...",
            MAX_RECORDING.as_secs()
        )),
        AudioEvent::AudioData(_) => logger.info("Silence detected, processing..."),
        AudioEvent::StopRecording => {}
        AudioEvent::EventsDropped(count) => logger.error(format!(
            "Lost {} audio events because the processing queue was full",
            count
        )),
        AudioEvent::InputError(message) => logger.error(message.as_str()),
    }
}

/// Encode a recording for upload on a blocking thread, so that a long
/// recording does not hold a runtime worker and shutdown does not wait for
/// it.
pub async fn encode_for_upload(audio: CapturedAudio) -> Result<Vec<u8>, String> {
    match tokio::task::spawn_blocking(move || encode_upload_wav(&audio)).await {
        Ok(Ok(wav)) => Ok(wav),
        Ok(Err(e)) => Err(format!("cannot encode the recording: {}", e)),
        Err(e) => Err(format!("encoding the recording failed: {}", e)),
    }
}
