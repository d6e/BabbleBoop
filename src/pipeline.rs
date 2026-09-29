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
    chatbox: Chatbox,
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
    async fn accept_transcription(
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
    async fn deliver_translation(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::LogLevel;
    use crate::test_support::{
        chat_completion_body, chat_completion_body_with, missing_data_dir, price_estimator,
        LogCapture, SHORT_USAGE, TOKENS,
    };
    use tokio::net::UdpSocket;

    // ===========================================================================
    // Test: A response without a translation is logged, not sent
    // ===========================================================================

    /// What the processing loop did with a response from the API.
    #[derive(Debug)]
    struct Delivery {
        /// Level and text of the activity log entries.
        log: Vec<(LogLevel, String)>,
        /// The OSC messages that reached the chatbox address.
        chatbox: Vec<rosc::OscMessage>,
        /// The total cost after the response, from zero.
        total_cost: f64,
    }

    /// A chatbox on a local UDP socket, the activity log, and a total cost
    /// that starts at zero, in a data folder of its own named after the
    /// test. The pipeline sends its requests to an address that nothing
    /// answers; the tests that use this fixture call `accept_transcription`
    /// or `deliver_translation` directly, so no request goes out.
    struct DeliveryFixture {
        config: Config,
        chatbox: UdpSocket,
        pipeline: Pipeline,
        app_state: Arc<AppState>,
        log_rx: tokio::sync::mpsc::Receiver<crate::app_state::LogEntry>,
        cost_file: std::path::PathBuf,
    }

    impl DeliveryFixture {
        async fn new(test_name: &str) -> Self {
            use crate::typing_indicator::TypingIndicator;

            let chatbox = UdpSocket::bind("127.0.0.1:0").await.unwrap();
            let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
            let mut config = Config::default();
            config.osc.address = "127.0.0.1".to_string();
            config.osc.output_port = chatbox.local_addr().unwrap().port();
            config.osc.display_time = 0;
            let (app_state, _cmd_rx, log_rx) = crate::test_support::app_state_with_channels();
            let typing_indicator = TypingIndicator::new(socket.clone(), app_state.logger.clone());
            let data_dir_path = std::env::temp_dir().join(format!(
                "babble_boop_{}_{}_delivery_fixture",
                test_name,
                std::process::id()
            ));
            std::fs::create_dir_all(&data_dir_path).unwrap();
            let data_dir = DataDir::new(data_dir_path);
            let cost_file = data_dir.cost_file();
            if cost_file.exists() {
                std::fs::remove_file(&cost_file).unwrap();
            }
            let services = ProcessingServices::new(&config, &data_dir, &app_state.logger);
            let api = OpenAi::new("http://127.0.0.1:1/v1").unwrap();
            let pipeline = Pipeline::new(
                Arc::clone(&app_state),
                api,
                Chatbox::new(socket),
                typing_indicator,
                services,
            );
            DeliveryFixture {
                config,
                chatbox,
                pipeline,
                app_state,
                log_rx,
                cost_file,
            }
        }

        /// Log an error as the processing loop logs an error of
        /// `Pipeline::process`, then collect what reached the activity log
        /// and the chatbox.
        async fn finish(mut self, result: Result<(), Box<dyn Error>>) -> Delivery {
            if let Err(e) = result {
                crate::processing_loop::log_processing_error(&*e, &self.app_state.logger);
            }
            if let Some(data_dir) = self.cost_file.parent() {
                std::fs::remove_dir_all(data_dir).unwrap();
            }
            assert_eq!(
                self.app_state.get_total_cost(),
                self.pipeline.services.price_estimator.total_cost
            );

            let mut chatbox_messages = Vec::new();
            let mut buf = [0u8; 1024];
            while let Ok(received) =
                tokio::time::timeout(Duration::from_millis(200), self.chatbox.recv_from(&mut buf))
                    .await
            {
                let (len, _) = received.unwrap();
                match rosc::decoder::decode_udp(&buf[..len]).unwrap().1 {
                    rosc::OscPacket::Message(message) => chatbox_messages.push(message),
                    bundle => panic!("unexpected OSC bundle {:?}", bundle),
                }
            }
            Delivery {
                log: std::iter::from_fn(|| self.log_rx.try_recv().ok())
                    .map(|entry| (entry.level, entry.message))
                    .collect(),
                chatbox: chatbox_messages,
                total_cost: self.pipeline.services.price_estimator.total_cost,
            }
        }
    }

    /// Give the Chat Completions response `body` to `parse_response` and
    /// `deliver_translation` in a `DeliveryFixture`.
    async fn deliver_response(test_name: &str, body: &str) -> Delivery {
        let mut fixture = DeliveryFixture::new(test_name).await;
        let request = ChatGptRequest::translation(&fixture.config.openai.model, "French", "Hello");

        let result = match request.parse_response(body) {
            Ok(translation) => {
                fixture
                    .pipeline
                    .deliver_translation(translation, "Hello", &fixture.config)
                    .await
            }
            Err(e) => Err(e),
        };
        fixture.finish(result).await
    }

    /// The only OSC message that a response without a translation sends:
    /// the typing indicator off.
    fn typing_off() -> rosc::OscMessage {
        rosc::OscMessage {
            addr: "/chatbox/typing".to_string(),
            args: vec![rosc::OscType::Bool(false)],
        }
    }

    /// The cost of a response that reports `SHORT_USAGE`.
    fn short_usage_cost() -> f64 {
        use crate::price_estimator::TokenCounts;

        let config = Config::default();
        price_estimator(&config.openai.model, &config.openai.transcription_model)
            .estimate_translation_cost(TokenCounts {
                input: 58,
                output: 12,
            })
    }

    /// A response without a translation logs `message` as an error, sends
    /// nothing to the chatbox, turns the typing indicator off, and adds
    /// its reported tokens to the total cost.
    fn assert_not_translated(delivery: Delivery, message: &str) {
        assert_eq!(
            delivery.log,
            [(LogLevel::Error, message.to_string())],
            "{:?}",
            delivery
        );
        assert_eq!(delivery.chatbox, [typing_off()], "{:?}", delivery);
        assert!(short_usage_cost() > 0.0);
        assert_eq!(delivery.total_cost, short_usage_cost(), "{:?}", delivery);
    }

    #[tokio::test]
    async fn test_translation_is_sent_to_the_chatbox_and_costed() {
        let body = chat_completion_body("Bonjour", Some(SHORT_USAGE));

        let delivery = deliver_response("translated", &body).await;

        assert_eq!(
            delivery.log,
            [(LogLevel::Success, "Translation: Bonjour".to_string())]
        );
        assert_eq!(
            delivery.chatbox,
            [
                rosc::OscMessage {
                    addr: "/chatbox/input".to_string(),
                    args: vec![
                        rosc::OscType::String("Bonjour".to_string()),
                        rosc::OscType::Bool(true),
                        rosc::OscType::Bool(true),
                    ],
                },
                typing_off(),
            ]
        );
        assert_eq!(delivery.total_cost, short_usage_cost());
    }

    #[tokio::test]
    async fn test_refusal_is_logged_and_not_sent() {
        let body = chat_completion_body_with(
            "null",
            r#""I'm sorry, but I can't help with that.""#,
            "stop",
            Some(SHORT_USAGE),
        );

        assert_not_translated(
            deliver_response("refusal", &body).await,
            "The model refused to translate: I'm sorry, but I can't help with that.",
        );
    }

    #[tokio::test]
    async fn test_content_filtered_response_is_logged_and_not_sent() {
        let body = chat_completion_body_with("null", "null", "content_filter", Some(SHORT_USAGE));

        assert_not_translated(
            deliver_response("content_filter", &body).await,
            "The content filter removed the translation, \
             so nothing was sent (finish_reason: content_filter)",
        );
    }

    #[tokio::test]
    async fn test_partly_filtered_translation_is_logged_and_not_sent() {
        let body =
            chat_completion_body_with(r#""Bonjour""#, "null", "content_filter", Some(SHORT_USAGE));

        assert_not_translated(
            deliver_response("content_filter_partial", &body).await,
            "The content filter removed part of the translation, \
             so it was not sent (finish_reason: content_filter)",
        );
    }

    #[tokio::test]
    async fn test_refusal_comes_before_the_content_filter() {
        let body = chat_completion_body_with(
            r#""Bonjour""#,
            r#""I'm sorry, but I can't help with that.""#,
            "content_filter",
            Some(SHORT_USAGE),
        );

        assert_not_translated(
            deliver_response("content_filter_refusal", &body).await,
            "The model refused to translate: I'm sorry, but I can't help with that.",
        );
    }

    #[tokio::test]
    async fn test_empty_response_is_logged_and_not_sent() {
        for (name, content) in [("empty_null", "null"), ("empty_string", r#""  ""#)] {
            let body = chat_completion_body_with(content, "null", "stop", Some(SHORT_USAGE));

            assert_not_translated(
                deliver_response(name, &body).await,
                "The model returned no translation (finish_reason: stop)",
            );
        }
    }

    #[tokio::test]
    async fn test_response_without_choices_is_logged_and_not_sent() {
        let body = format!(
            r#"{{
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1790000000,
                "model": "gpt-5.6-sol-2026-08-14",
                "choices": [],
                "usage": {}
            }}"#,
            SHORT_USAGE
        );

        assert_not_translated(
            deliver_response("no_choices", &body).await,
            "The model returned no translation (the response has no choices)",
        );
    }

    #[tokio::test]
    async fn test_cut_off_translation_is_logged_and_not_sent() {
        let body = chat_completion_body_with(
            r#""Bonjour, bonjour, bonjour, bonjour""#,
            "null",
            "length",
            Some(SHORT_USAGE),
        );

        assert_not_translated(
            deliver_response("length", &body).await,
            "The translation reached the output token limit and was cut off, \
             so it was not sent (finish_reason: length)",
        );
    }

    // ===========================================================================
    // Test: A transcription response is costed
    // ===========================================================================

    /// The body of a transcription response with `text_json` as its text.
    fn transcription_body(text_json: &str) -> String {
        format!(
            r#"{{
                "text": {},
                "usage": {{
                    "type": "tokens",
                    "input_tokens": 14,
                    "input_token_details": {{"text_tokens": 0, "audio_tokens": 14}},
                    "output_tokens": 2,
                    "total_tokens": 16
                }}
            }}"#,
            text_json
        )
    }

    /// Give the transcription response `body` for one second of audio to
    /// `parse_transcription` and `accept_transcription` in a
    /// `DeliveryFixture`. Returns the text to translate, if any, and what
    /// happened.
    async fn accept_transcription_response(
        test_name: &str,
        body: &str,
    ) -> (Option<String>, Delivery) {
        use crate::transcription::parse_transcription;

        let mut fixture = DeliveryFixture::new(test_name).await;
        let (text, result) = match parse_transcription(body) {
            Ok(text) => (
                fixture
                    .pipeline
                    .accept_transcription(text, Duration::from_secs(1), &fixture.config)
                    .await,
                Ok(()),
            ),
            Err(e) => (None, Err(e)),
        };
        (text, fixture.finish(result).await)
    }

    /// The cost of transcribing one second of audio.
    fn one_second_transcription_cost() -> f64 {
        let config = Config::default();
        price_estimator(&config.openai.model, &config.openai.transcription_model)
            .estimate_transcription_cost(Duration::from_secs(1))
    }

    #[tokio::test]
    async fn test_transcription_is_costed_and_translated() {
        let (text, delivery) =
            accept_transcription_response("transcribed", &transcription_body(r#""Hello""#)).await;

        assert_eq!(text, Some("Hello".to_string()));
        assert_eq!(
            delivery.log,
            [(LogLevel::Info, "Transcription: Hello".to_string())]
        );
        assert!(delivery.chatbox.is_empty(), "{:?}", delivery);
        assert_eq!(delivery.total_cost, one_second_transcription_cost());
    }

    /// A response with an empty or blank text costs as much as any other
    /// response for the same audio, but there is nothing to translate.
    #[tokio::test]
    async fn test_empty_transcription_is_costed_and_not_translated() {
        for (name, text_json) in [
            ("transcription_empty", r#""""#),
            ("transcription_blank", r#"" \n ""#),
        ] {
            let (text, delivery) =
                accept_transcription_response(name, &transcription_body(text_json)).await;

            assert_eq!(text, None, "{}", name);
            assert_eq!(
                delivery.log,
                [(
                    LogLevel::Error,
                    "The transcription is empty, so nothing was translated".to_string()
                )],
                "{}",
                name
            );
            assert_eq!(delivery.chatbox, [typing_off()], "{}", name);
            assert!(one_second_transcription_cost() > 0.0);
            assert_eq!(
                delivery.total_cost,
                one_second_transcription_cost(),
                "{}",
                name
            );
        }
    }

    /// A body with a null text is an error of `Pipeline::process`, and there
    /// is no transcription to cost.
    #[tokio::test]
    async fn test_transcription_body_with_null_text_is_an_error() {
        let (text, delivery) =
            accept_transcription_response("transcription_null", &transcription_body("null")).await;

        assert_eq!(text, None);
        assert!(
            matches!(
                delivery.log.as_slice(),
                [(LogLevel::Error, message)] if message.starts_with("Error: ")
            ),
            "{:?}",
            delivery
        );
        assert!(delivery.chatbox.is_empty(), "{:?}", delivery);
        assert_eq!(delivery.total_cost, 0.0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_response_without_translation_does_not_wait_for_the_chatbox() {
        let mut fixture = DeliveryFixture::new("refusal_after_message").await;
        fixture.config.osc.display_time = 10_000;
        fixture
            .pipeline
            .chatbox
            .send("first", &fixture.config)
            .await
            .unwrap();
        let body = chat_completion_body_with(
            "null",
            r#""I can't help with that.""#,
            "stop",
            Some(SHORT_USAGE),
        );
        let translation =
            ChatGptRequest::translation(&fixture.config.openai.model, "French", "Hello")
                .parse_response(&body)
                .unwrap();

        let start = tokio::time::Instant::now();
        let result = fixture
            .pipeline
            .deliver_translation(translation, "Hello", &fixture.config)
            .await;

        assert!(start.elapsed().is_zero(), "waited {:?}", start.elapsed());
        let delivery = fixture.finish(result).await;
        assert_eq!(delivery.log.first().unwrap().0, LogLevel::Error);
        assert_eq!(delivery.chatbox.len(), 2, "{:?}", delivery);
        assert_eq!(delivery.chatbox[1], typing_off());
    }

    // ===========================================================================
    // Test: Translation switched off before the chatbox send
    // ===========================================================================

    /// The first chunk of `text` as the chatbox receives it.
    fn chatbox_input(text: &str) -> rosc::OscMessage {
        rosc::OscMessage {
            addr: "/chatbox/input".to_string(),
            args: vec![
                rosc::OscType::String(text.to_string()),
                rosc::OscType::Bool(true),
                rosc::OscType::Bool(true),
            ],
        }
    }

    /// A `DeliveryFixture` whose chatbox shows "first" for 10 s from now.
    async fn fixture_showing_first(test_name: &str) -> DeliveryFixture {
        let mut fixture = DeliveryFixture::new(test_name).await;
        fixture.config.osc.display_time = 10_000;
        fixture
            .pipeline
            .chatbox
            .send("first", &fixture.config)
            .await
            .unwrap();
        fixture
    }

    /// Set the translation toggle to `enabled` after `delay`, as the GUI
    /// does: it stores the value at once, and the processing loop handles
    /// the `SetEnabled` command only after `Pipeline::process` returns.
    fn toggle_after(app_state: &Arc<AppState>, delay: Duration, enabled: bool) {
        let app_state = Arc::clone(app_state);
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            app_state.enabled.store(enabled, Ordering::Relaxed);
        });
    }

    /// Give the translation "Bonjour" of "Hello" to `deliver_translation`
    /// in `fixture`.
    async fn deliver_bonjour(fixture: &mut DeliveryFixture) -> Result<(), Box<dyn Error>> {
        let body = chat_completion_body("Bonjour", Some(SHORT_USAGE));
        let translation =
            ChatGptRequest::translation(&fixture.config.openai.model, "French", "Hello")
                .parse_response(&body)
                .unwrap();
        fixture
            .pipeline
            .deliver_translation(translation, "Hello", &fixture.config)
            .await
    }

    /// The log lines of a translation that was not sent because
    /// translation was switched off.
    fn not_sent_log() -> Vec<(LogLevel, String)> {
        vec![
            (LogLevel::Success, "Translation: Bonjour".to_string()),
            (
                LogLevel::Info,
                "Translation is off, so the translation was not sent to the chatbox".to_string(),
            ),
        ]
    }

    #[tokio::test(start_paused = true)]
    async fn test_translation_switched_off_during_the_display_wait_is_not_sent() {
        let mut fixture = fixture_showing_first("off_during_display_wait").await;
        toggle_after(&fixture.app_state, Duration::from_secs(1), false);

        let result = deliver_bonjour(&mut fixture).await;

        let delivery = fixture.finish(result).await;
        assert_eq!(
            delivery.chatbox,
            [chatbox_input("first"), typing_off()],
            "{:?}",
            delivery
        );
        assert_eq!(delivery.log, not_sent_log());
        assert_eq!(delivery.total_cost, short_usage_cost());
    }

    #[tokio::test(start_paused = true)]
    async fn test_translation_switched_off_during_the_api_calls_is_not_sent() {
        let mut fixture = DeliveryFixture::new("off_during_api_calls").await;
        // No display wait: the toggle came while the requests ran.
        fixture.app_state.enabled.store(false, Ordering::Relaxed);

        let result = deliver_bonjour(&mut fixture).await;

        let delivery = fixture.finish(result).await;
        assert_eq!(delivery.chatbox, [typing_off()], "{:?}", delivery);
        assert_eq!(delivery.log, not_sent_log());
        assert_eq!(delivery.total_cost, short_usage_cost());
    }

    #[tokio::test(start_paused = true)]
    async fn test_translation_switched_off_and_on_during_the_display_wait_is_sent() {
        let mut fixture = fixture_showing_first("off_and_on_during_display_wait").await;
        toggle_after(&fixture.app_state, Duration::from_secs(1), false);
        toggle_after(&fixture.app_state, Duration::from_secs(2), true);

        let start = tokio::time::Instant::now();
        let result = deliver_bonjour(&mut fixture).await;

        assert_eq!(start.elapsed(), Duration::from_secs(10));
        let delivery = fixture.finish(result).await;
        assert_eq!(
            delivery.chatbox,
            [
                chatbox_input("first"),
                chatbox_input("Bonjour"),
                typing_off()
            ],
            "{:?}",
            delivery
        );
        assert_eq!(
            delivery.log,
            [(LogLevel::Success, "Translation: Bonjour".to_string())]
        );
    }

    // ===========================================================================
    // Test: Saved settings reach the processing loop
    // ===========================================================================

    #[test]
    fn test_config_update_applies_new_model_prices_and_keeps_total() {
        let mut config = Config::default();
        config.openai.model = "gpt-4o-mini".to_string();
        config.openai.transcription_model = "whisper-1".to_string();
        let mut services =
            ProcessingServices::new(&config, &missing_data_dir(), &LogCapture::new().logger());
        services.price_estimator.total_cost = 1.25;

        config.openai.model = "gpt-4o".to_string();
        config.openai.transcription_model = "gpt-4o-mini-transcribe".to_string();
        services.apply_config(&config, &LogCapture::new().logger());

        let expected = price_estimator("gpt-4o", "gpt-4o-mini-transcribe");
        let old = price_estimator("gpt-4o-mini", "whisper-1");
        let minute = Duration::from_secs(60);
        let estimator = &services.price_estimator;
        assert_ne!(
            expected.estimate_translation_cost(TOKENS),
            old.estimate_translation_cost(TOKENS)
        );
        assert_ne!(
            expected.estimate_transcription_cost(minute),
            old.estimate_transcription_cost(minute)
        );
        assert_eq!(
            estimator.estimate_translation_cost(TOKENS),
            expected.estimate_translation_cost(TOKENS)
        );
        assert_eq!(
            estimator.estimate_transcription_cost(minute),
            expected.estimate_transcription_cost(minute)
        );
        // The running total stays in memory; it is not reloaded from disk.
        assert_eq!(estimator.total_cost, 1.25);
    }

    #[tokio::test]
    async fn test_cost_and_recordings_are_saved_in_the_data_folder() {
        let dir =
            std::env::temp_dir().join(format!("babble_boop_data_folder_{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("total_cost.txt"), "1.5").unwrap();
        let data_dir = DataDir::new(dir.clone());
        let config = Config {
            keep_audio_files: true,
            ..Config::default()
        };
        let logger = LogCapture::new().logger();

        let mut services = ProcessingServices::new(&config, &data_dir, &logger);
        let loaded_cost = services.price_estimator.total_cost;
        services.price_estimator.add_cost(0.5, &logger);
        services
            .recording_manager
            .as_mut()
            .expect("keep_audio_files is on")
            .save_recording(vec![0u8; 4], "first", &logger)
            .await;
        // Saved settings make a new recording manager
        services.apply_config(&config, &logger);
        services
            .recording_manager
            .as_mut()
            .expect("keep_audio_files is on")
            .save_recording(vec![0u8; 4], "second", &logger)
            .await;
        let saved_cost = std::fs::read_to_string(dir.join("total_cost.txt")).ok();
        let mut recordings: Vec<String> = std::fs::read_dir(dir.join("recordings"))
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        recordings.sort();
        // Clean up before asserting, so a failure does not leave files behind
        std::fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded_cost, 1.5);
        assert_eq!(saved_cost.as_deref(), Some("2"));
        assert_eq!(recordings.len(), 2, "{:?}", recordings);
        assert!(recordings[0].ends_with("_first.wav"), "{:?}", recordings);
        assert!(recordings[1].ends_with("_second.wav"), "{:?}", recordings);
    }

    /// Activity log entries written by `body`.
    fn entries_logged_by(body: impl FnOnce(&Logger)) -> Vec<crate::app_state::LogEntry> {
        let mut log = LogCapture::new();
        body(&log.logger());
        log.full_entries()
    }

    #[test]
    fn test_unknown_model_pricing_is_logged() {
        let mut config = Config::default();
        config.openai.model = "my-finetuned-model".to_string();
        let entries = entries_logged_by(|logger| {
            ProcessingServices::new(&config, &missing_data_dir(), logger);
        });
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert!(entries[0].message.contains("'my-finetuned-model'"));

        let mut services = ProcessingServices::new(
            &Config::default(),
            &missing_data_dir(),
            &LogCapture::new().logger(),
        );
        config.openai.model = "gpt-4o".to_string();
        config.openai.transcription_model = "my-transcriber".to_string();
        let entries = entries_logged_by(|logger| services.apply_config(&config, logger));
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert!(entries[0].message.contains("'my-transcriber'"));
    }

    #[test]
    fn test_model_shutdown_date_is_logged_at_start_and_on_save() {
        let mut config = Config::default();
        config.openai.model = "gpt-3.5-turbo".to_string();
        let entries = entries_logged_by(|logger| {
            ProcessingServices::new(&config, &missing_data_dir(), logger);
        });
        assert_eq!(entries.len(), 1, "{:?}", entries);
        for text in ["'gpt-3.5-turbo'", "2026-10-23", "gpt-5.6-terra"] {
            assert!(entries[0].message.contains(text), "{:?}", entries[0]);
        }

        let mut services = ProcessingServices::new(
            &Config::default(),
            &missing_data_dir(),
            &LogCapture::new().logger(),
        );
        config.openai.model = Config::default().openai.model;
        config.openai.transcription_model = "whisper-1".to_string();
        let entries = entries_logged_by(|logger| services.apply_config(&config, logger));
        assert_eq!(entries.len(), 1, "{:?}", entries);
        for text in ["'whisper-1'", "2027-02-26", "gpt-transcribe"] {
            assert!(entries[0].message.contains(text), "{:?}", entries[0]);
        }
    }

    #[test]
    fn test_suggested_models_are_not_scheduled_to_shut_down() {
        for model in crate::models::suggested_chat_models() {
            for transcription_model in crate::models::suggested_transcription_models() {
                let mut config = Config::default();
                config.openai.model = model.to_string();
                config.openai.transcription_model = transcription_model.to_string();
                let entries = entries_logged_by(|logger| {
                    ProcessingServices::new(&config, &missing_data_dir(), logger);
                });
                assert!(entries.is_empty(), "{:?}", entries);
            }
        }
    }

    #[test]
    fn test_known_model_pricing_is_not_logged() {
        let config = Config::default();
        let entries = entries_logged_by(|logger| {
            ProcessingServices::new(&config, &missing_data_dir(), logger)
                .apply_config(&config, logger);
        });
        assert!(entries.is_empty(), "{:?}", entries);
    }

    /// Time one more rate limiter wait after using up a budget of two
    /// requests and saving settings with `requests_per_minute` set to `limit`.
    async fn wait_after_saving_limit(limit: usize) -> Duration {
        let mut config = Config::default();
        config.rate_limit.requests_per_minute = 2;
        let mut services =
            ProcessingServices::new(&config, &missing_data_dir(), &LogCapture::new().logger());
        services.rate_limiter.wait().await;
        services.rate_limiter.wait().await;

        config.rate_limit.requests_per_minute = limit;
        services.apply_config(&config, &LogCapture::new().logger());

        let start = tokio::time::Instant::now();
        services.rate_limiter.wait().await;
        start.elapsed()
    }

    #[tokio::test(start_paused = true)]
    async fn test_saving_settings_keeps_rate_limit_budget() {
        // The budget is used up, so the next request waits for the window.
        assert!(wait_after_saving_limit(2).await >= Duration::from_secs(59));
    }

    #[tokio::test(start_paused = true)]
    async fn test_raised_rate_limit_applies_at_once() {
        assert_eq!(wait_after_saving_limit(3).await, Duration::ZERO);
    }

    // ===========================================================================
    // Test: Any minimum transcription duration in config.toml is safe
    // ===========================================================================

    /// Run `Pipeline::process` on one second of audio with `min_seconds` as
    /// the minimum transcription duration.
    async fn check_one_second_against_minimum(
        min_seconds: f32,
    ) -> crate::test_support::MinimumCheck {
        use crate::types::CapturedAudio;

        let second = CapturedAudio {
            samples: vec![0.25; 16_000],
            channels: 1,
            sample_rate: 16_000,
        };
        crate::test_support::check_against_minimum(second, Extent::Whole, min_seconds).await
    }

    #[tokio::test]
    async fn test_negative_minimum_duration_means_no_minimum() {
        use crate::test_support::MinimumCheck;

        for min_seconds in [-1.0, -f32::MIN_POSITIVE, f32::NEG_INFINITY] {
            assert_eq!(
                (
                    min_seconds,
                    check_one_second_against_minimum(min_seconds).await
                ),
                (min_seconds, MinimumCheck::Transcribed)
            );
        }
    }

    #[tokio::test]
    async fn test_nan_minimum_duration_means_no_minimum() {
        use crate::test_support::MinimumCheck;

        assert_eq!(
            check_one_second_against_minimum(f32::NAN).await,
            MinimumCheck::Transcribed
        );
    }

    #[tokio::test]
    async fn test_infinite_minimum_duration_skips_the_recording() {
        use crate::test_support::MinimumCheck;

        assert_eq!(
            check_one_second_against_minimum(f32::INFINITY).await,
            MinimumCheck::Skipped
        );
    }

    #[tokio::test]
    async fn test_minimum_duration_too_large_for_duration_skips_the_recording() {
        use crate::test_support::MinimumCheck;

        // Duration holds at most u64::MAX seconds, about 1.8e19
        for min_seconds in [1e20, f32::MAX] {
            assert_eq!(
                (
                    min_seconds,
                    check_one_second_against_minimum(min_seconds).await
                ),
                (min_seconds, MinimumCheck::Skipped)
            );
        }
    }

    #[tokio::test]
    async fn test_valid_minimum_duration_is_kept() {
        use crate::test_support::MinimumCheck;

        for (min_seconds, expected) in [
            (0.0, MinimumCheck::Transcribed),
            (0.5, MinimumCheck::Transcribed),
            (1.0, MinimumCheck::Transcribed),
            (2.0, MinimumCheck::Skipped),
        ] {
            assert_eq!(
                (
                    min_seconds,
                    check_one_second_against_minimum(min_seconds).await
                ),
                (min_seconds, expected)
            );
        }
    }

    #[tokio::test]
    async fn test_minimum_duration_does_not_apply_to_a_part_of_a_long_recording() {
        use crate::test_support::MinimumCheck;
        use crate::types::CapturedAudio;

        // A part holds 30 s unless it is the last part; a config.toml
        // minimum can be longer than that.
        let second = CapturedAudio {
            samples: vec![0.25; 16_000],
            channels: 1,
            sample_rate: 16_000,
        };
        assert_eq!(
            crate::test_support::check_against_minimum(second, Extent::Part, 2.0).await,
            MinimumCheck::Transcribed
        );
    }
}
