//! The API pipeline for one utterance: transcribe, cost, save the debug
//! recording, translate, and deliver to the VRChat chatbox. `Pipeline`
//! owns what this needs across utterances: the OpenAI client, the
//! chatbox, the typing indicator, and the `ProcessingServices` that
//! depend on the settings (rate limiter, price estimator, recording
//! manager). `Config` comes with each call, since the processing loop
//! replaces it only between utterances.

use crate::api_client::OpenAi;
use crate::app_state::{AppState, Logger};
use crate::chatbox::Chatbox;
use crate::config::Config;
use crate::data_dir::DataDir;
use crate::models;
use crate::price_estimator::PriceEstimator;
use crate::rate_limiter::RateLimiter;
use crate::recording_manager::RecordingManager;
use crate::transcription::transcribe_audio;
use crate::translation::{ask_chatgpt, ChatGptRequest, Translation};
use crate::types::Extent;
use crate::typing_indicator::TypingIndicator;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;

/// The API pipeline: the OpenAI client, the chatbox and typing indicator
/// it sends to, the settings dependent services, and the state it shares
/// with the GUI.
pub struct Pipeline {
    api: OpenAi,
    pub(crate) chatbox: Chatbox,
    pub(crate) typing_indicator: TypingIndicator,
    pub(crate) services: ProcessingServices,
    app_state: Arc<AppState>,
}

impl Pipeline {
    /// A pipeline that sends its requests through `api`, its OSC messages
    /// through `chatbox` and `typing_indicator`, and keeps the total cost
    /// and the recordings of `app_state` through `services`.
    pub fn new(
        app_state: Arc<AppState>,
        api: OpenAi,
        chatbox: Chatbox,
        typing_indicator: TypingIndicator,
        services: ProcessingServices,
    ) -> Self {
        Self {
            api,
            chatbox,
            typing_indicator,
            services,
            app_state,
        }
    }

    /// Send from `socket` from now on, for both the chatbox and the
    /// typing indicator.
    pub fn set_socket(&mut self, socket: Arc<UdpSocket>) {
        self.typing_indicator.set_socket(Arc::clone(&socket));
        self.chatbox.set_socket(socket);
    }

    /// Run one utterance through the pipeline: transcribe `upload`, add
    /// its cost, save it if debug recording is on, translate it, and
    /// deliver the translation to the chatbox. `audio_duration` is the
    /// duration of the captured recording `upload` was encoded from.
    pub async fn process(
        &mut self,
        upload: Vec<u8>,
        audio_duration: Duration,
        extent: Extent,
        config: &Config,
    ) -> Result<(), Box<dyn Error>> {
        let min_duration = min_transcription_duration(config.audio.min_transcription_duration);
        if extent == Extent::Whole && audio_duration < min_duration {
            self.app_state.logger.info(format!(
                "Audio too short ({:.2}s < {:.2}s), skipping",
                audio_duration.as_secs_f32(),
                min_duration.as_secs_f32()
            ));
            self.typing_indicator.stop_typing(config).await;
            return Ok(());
        }

        let text = transcribe_audio(
            &self.api,
            upload.clone(),
            &config.openai,
            &mut self.services.rate_limiter,
        )
        .await?;
        let Some(transcription) = self
            .accept_transcription(text, audio_duration, config)
            .await
        else {
            return Ok(());
        };

        // Save the audio recording if debug mode is enabled. A failed save
        // is logged by the manager itself and does not stop the
        // translation.
        if let Some(manager) = self.services.recording_manager.as_mut() {
            manager
                .save_recording(upload, &transcription, &self.app_state.logger)
                .await;
        }

        let request = ChatGptRequest::translation(
            &config.openai.model,
            &config.translation.target_language,
            &transcription,
        );

        let translation = ask_chatgpt(
            &self.api,
            &request,
            &config.openai,
            &mut self.services.rate_limiter,
        )
        .await?;
        self.deliver_translation(translation, &transcription, config)
            .await
    }

    /// Add the cost of the transcription request to the total and return
    /// the text to translate. The cost is added before the later steps,
    /// so it stays in the total when one of them fails. The estimate
    /// depends only on the audio duration, so an empty or blank text
    /// costs as much as speech. For such a text, an error message goes to
    /// the activity log, the typing indicator at the address in `config`
    /// turns off, and the function returns `None`.
    pub(crate) async fn accept_transcription(
        &mut self,
        text: String,
        audio_duration: Duration,
        config: &Config,
    ) -> Option<String> {
        let cost = self
            .services
            .price_estimator
            .estimate_transcription_cost(audio_duration);
        self.services
            .price_estimator
            .add_cost(cost, &self.app_state.logger);
        self.app_state
            .set_total_cost(self.services.price_estimator.total_cost);

        if text.trim().is_empty() {
            self.app_state
                .logger
                .error("The transcription is empty, so nothing was translated");
            self.typing_indicator.stop_typing(config).await;
            return None;
        }
        self.app_state
            .logger
            .info(format!("Transcription: {}", text));
        Some(text)
    }

    /// Add the cost of the translation request to the total and send the
    /// translation to the chatbox. A response without a translation, such
    /// as a refusal, costs its tokens too; it goes to the activity log,
    /// not to the chatbox. The translation waits for the display time of
    /// the previous message, and is not sent if translation is off after
    /// that wait. A message that has started to go out is sent to its
    /// last chunk.
    pub(crate) async fn deliver_translation(
        &mut self,
        translation: Translation,
        transcription: &str,
        config: &Config,
    ) -> Result<(), Box<dyn Error>> {
        let translation_cost = self
            .services
            .price_estimator
            .estimate_translation_cost(translation.tokens);
        self.services
            .price_estimator
            .add_cost(translation_cost, &self.app_state.logger);
        self.app_state
            .set_total_cost(self.services.price_estimator.total_cost);

        let text = match translation.text {
            Ok(text) => text,
            Err(no_translation) => {
                self.app_state.logger.error(no_translation.to_string());
                self.typing_indicator.stop_typing(config).await;
                return Ok(());
            }
        };
        self.app_state
            .logger
            .success(format!("Translation: {}", text));

        let mut final_response = text;
        if config.translation.include_original_message {
            final_response = final_response + "\n" + transcription;
        }
        // Translation can be switched off while this message waits for the
        // previous one or for the API. The processing loop checks the
        // toggle only when it takes the audio event, so check it again
        // here.
        self.chatbox.wait_for_display(config).await;
        if !self.app_state.enabled.load(Ordering::Relaxed) {
            self.app_state
                .logger
                .info("Translation is off, so the translation was not sent to the chatbox");
            self.typing_indicator.stop_typing(config).await;
            return Ok(());
        }
        self.chatbox.send(&final_response, config).await?;

        self.typing_indicator.stop_typing(config).await;

        Ok(())
    }
}

/// Processing loop state that depends on the settings.
pub struct ProcessingServices {
    pub rate_limiter: RateLimiter,
    pub price_estimator: PriceEstimator,
    pub recording_manager: Option<RecordingManager>,
    /// Where the recording manager saves recordings
    recordings_dir: PathBuf,
}

impl ProcessingServices {
    /// Services that keep the total cost and the recordings in `data_dir`.
    pub fn new(config: &Config, data_dir: &DataDir, logger: &Logger) -> Self {
        log_model_warnings(config, logger);
        let recordings_dir = data_dir.recordings_dir();
        Self {
            rate_limiter: RateLimiter::new(config.rate_limit.requests_per_minute),
            price_estimator: PriceEstimator::new(
                data_dir.cost_file(),
                &config.openai.model,
                &config.openai.transcription_model,
            ),
            recording_manager: recording_manager(config, &recordings_dir),
            recordings_dir,
        }
    }

    /// Apply settings saved in the GUI.
    pub fn apply_config(&mut self, config: &Config, logger: &Logger) {
        // Keep the requests already counted; a new limiter would reset them.
        self.rate_limiter
            .set_max_requests(config.rate_limit.requests_per_minute);
        self.price_estimator
            .set_models(&config.openai.model, &config.openai.transcription_model);
        self.recording_manager = recording_manager(config, &self.recordings_dir);
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

fn recording_manager(config: &Config, recordings_dir: &Path) -> Option<RecordingManager> {
    config
        .keep_audio_files
        .then(|| RecordingManager::new(recordings_dir.to_path_buf(), config.max_audio_files))
}

/// The shortest recording to transcribe, from the minimum in seconds in
/// the config. config.toml can hold any float there, and
/// `Duration::from_secs_f32` panics if its argument is negative, not finite
/// or too large for `Duration` (see its documentation). A negative or NaN
/// minimum means no minimum. A minimum too large for `Duration`, such as
/// `inf`, skips every whole recording. The minimum does not apply to the
/// parts of a recording that reached the length limit: `Pipeline::process`
/// checks it only when `extent == Extent::Whole`, so the minimum never
/// stops those parts.
fn min_transcription_duration(seconds: f32) -> Duration {
    match Duration::try_from_secs_f32(seconds) {
        Ok(duration) => duration,
        Err(_) if seconds > 0.0 => Duration::MAX,
        Err(_) => Duration::ZERO,
    }
}
