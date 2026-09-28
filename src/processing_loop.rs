//! Parts of the processing loop in `main.rs` that can be tested without an
//! audio device.

use crate::config::Config;
use crate::price_estimator::PriceEstimator;
use crate::rate_limiter::RateLimiter;
use crate::recording_manager::RecordingManager;
use std::path::PathBuf;

/// Directory for saved recordings when `keep_audio_files` is on.
const RECORDINGS_DIR: &str = "recordings";

/// Processing loop state that depends on the settings.
pub struct ProcessingServices {
    pub rate_limiter: RateLimiter,
    pub price_estimator: PriceEstimator,
    pub recording_manager: Option<RecordingManager>,
}

impl ProcessingServices {
    pub fn new(config: &Config) -> Self {
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
    pub fn apply_config(&mut self, config: &Config) {
        // Keep the requests already counted; a new limiter would reset them.
        self.rate_limiter
            .set_max_requests(config.rate_limit.requests_per_minute);
        self.price_estimator
            .set_models(&config.openai.model, &config.openai.transcription_model);
        self.recording_manager = recording_manager(config);
    }
}

fn recording_manager(config: &Config) -> Option<RecordingManager> {
    config
        .keep_audio_files
        .then(|| RecordingManager::new(PathBuf::from(RECORDINGS_DIR), config.max_audio_files))
}
