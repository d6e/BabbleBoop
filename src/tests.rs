//! Tests for BabbleBoop
//!
//! These tests verify the fixes for issues identified during code review.

#[cfg(test)]
mod regression_tests {
    use crate::config::{
        AudioConfig, Config, OpenAiConfig, OscConfig, RateLimitConfig, ThemeMode, TranslationConfig,
    };
    use crate::price_estimator::PriceEstimator;
    use crate::rate_limiter::RateLimiter;

    // ===========================================================================
    // Regression test: Model pricing now covers common models
    // ===========================================================================

    #[test]
    fn test_common_models_have_pricing() {
        // These models should now have proper pricing
        let models_with_pricing = vec![
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ];

        for model in models_with_pricing {
            let estimator = PriceEstimator::new(model, "whisper-1");
            // Verify we can estimate non-zero costs
            let cost = estimator.estimate_translation_cost(1000, 500);
            assert!(cost > 0.0, "Model {} should have non-zero pricing", model);
        }
    }

    #[test]
    fn test_unknown_model_uses_default_pricing() {
        // Unknown models should use conservative default pricing (gpt-4o-mini rates)
        let estimator = PriceEstimator::new("unknown-model-xyz", "whisper-1");
        let cost = estimator.estimate_translation_cost(1000, 500);
        // Should use gpt-4o-mini pricing as fallback, not zero
        assert!(cost > 0.0, "Unknown models should use default pricing");
    }

    // ===========================================================================
    // Regression test: Rate limiter behavior
    // ===========================================================================

    #[tokio::test]
    async fn test_rate_limiter_tracks_requests() {
        let mut limiter = RateLimiter::new(2);

        // First two requests should not block
        limiter.wait().await;
        limiter.wait().await;

        // Rate limiter should now be at capacity
        // (would block if we called wait again within the same minute)
    }

    // ===========================================================================
    // Test: Translation error response structure
    // ===========================================================================

    mod translation_tests {
        #[derive(serde::Deserialize)]
        struct ChatGptChoice {
            message: ChatGptMessage,
        }

        #[derive(serde::Deserialize)]
        struct ChatGptMessage {
            content: String,
        }

        #[derive(serde::Deserialize)]
        struct ChatGptResponse {
            choices: Vec<ChatGptChoice>,
        }

        #[test]
        fn test_empty_choices_handled_with_iterator() {
            // This test verifies the fix: using .into_iter().next() instead of [0]
            let json = r#"{"choices": []}"#;
            let response: ChatGptResponse = serde_json::from_str(json).unwrap();

            // Using iterator pattern (the fix) - returns None instead of panicking
            let result = response.choices.into_iter().next();
            assert!(result.is_none(), "Empty choices should return None");
        }

        #[test]
        fn test_valid_response_parsed_correctly() {
            let json = r#"{"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}"#;
            let response: ChatGptResponse = serde_json::from_str(json).unwrap();

            let choice = response.choices.into_iter().next();
            assert!(choice.is_some());
            assert_eq!(choice.unwrap().message.content, "Hello");
        }
    }

    // ===========================================================================
    // Test: WAV encoding helper
    // ===========================================================================

    #[test]
    fn test_wav_encoding() {
        use hound::{WavReader, WavSpec, WavWriter};
        use std::io::Cursor;

        // Create sample audio data
        let samples: Vec<f32> = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let channels = 1;
        let sample_rate = 44100;

        // Encode to WAV
        let mut wav_buffer = Vec::new();
        {
            let mut writer = WavWriter::new(
                Cursor::new(&mut wav_buffer),
                WavSpec {
                    channels: channels as u16,
                    sample_rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();

            for &sample in &samples {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }

        // Verify we can read it back
        let reader = WavReader::new(Cursor::new(&wav_buffer)).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 44100);
    }

    // ===========================================================================
    // Test: I16 to F32 conversion
    // ===========================================================================

    #[test]
    fn test_i16_to_f32_conversion() {
        fn i16_to_f32(sample: i16) -> f32 {
            sample as f32 / i16::MAX as f32
        }

        // Test boundary values
        assert!((i16_to_f32(0) - 0.0).abs() < 0.0001);
        assert!((i16_to_f32(i16::MAX) - 1.0).abs() < 0.0001);
        assert!((i16_to_f32(i16::MIN) - (-1.0)).abs() < 0.01);

        // Test mid-range value
        let mid = i16::MAX / 2;
        assert!((i16_to_f32(mid) - 0.5).abs() < 0.01);
    }

    // ===========================================================================
    // Test: Config load/save round-trip
    // ===========================================================================

    #[test]
    fn test_config_round_trip() {
        use std::fs;
        use std::path::PathBuf;

        let config = Config {
            osc: OscConfig {
                address: "127.0.0.1".to_string(),
                input_port: 9001,
                output_port: 9000,
                max_message_chunks: 9,
                display_time: 3000,
            },
            openai: OpenAiConfig {
                api_key: "test-api-key".to_string(),
                model: "gpt-4o-mini".to_string(),
                transcription_model: "whisper-1".to_string(),
            },
            translation: TranslationConfig {
                target_language: "Japanese".to_string(),
                include_original_message: false,
            },
            audio: AudioConfig {
                silence_threshold: 100,
                noise_gate_threshold: 0.3,
                noise_gate_hold_time: 0.20,
                min_transcription_duration: 1.0,
            },
            rate_limit: RateLimitConfig {
                requests_per_minute: 50,
            },
            // Fields with a serde default use non default values, so a field
            // that fails to save cannot pass by falling back to its default.
            keep_audio_files: true,
            max_audio_files: 25,
            theme: ThemeMode::Light,
        };

        // Create a temp file path
        let temp_path = PathBuf::from("test_config_roundtrip.toml");

        // Save config
        config.save(&temp_path).expect("Failed to save config");

        // Load it back
        let loaded = Config::load(&temp_path).expect("Failed to load config");

        // Clean up before asserting, so a failure does not leave the file behind
        fs::remove_file(&temp_path).ok();

        // Compare the whole struct, so every field is checked
        assert_eq!(loaded, config);
    }

    #[test]
    fn test_config_migration_from_old_format() {
        // Old config format (v0.3.1) used "debug" instead of "keep_audio_files",
        // didn't have "max_audio_files" or "transcription_model",
        // and had "passthrough_enabled" and "passthrough_port" in [osc]
        // Note: In TOML, top-level keys must come before any [section] declarations
        let old_config_toml = r#"
debug = true

[osc]
address = "127.0.0.1"
input_port = 9001
output_port = 9000
max_message_chunks = 9
display_time = 3000
passthrough_enabled = false
passthrough_port = 9002

[openai]
api_key = "test-key"
model = "gpt-4o-mini"

[translation]
target_language = "Japanese"
include_original_message = false

[audio]
silence_threshold = 100
noise_gate_threshold = 0.3
noise_gate_hold_time = 0.20
min_transcription_duration = 1.0

[rate_limit]
requests_per_minute = 50
"#;

        let config: Config =
            toml::from_str(old_config_toml).expect("Failed to parse old config format");

        // "debug = true" should be read as keep_audio_files = true
        assert!(
            config.keep_audio_files,
            "debug should be aliased to keep_audio_files"
        );
        // max_audio_files should default to 10
        assert_eq!(
            config.max_audio_files, 10,
            "max_audio_files should default to 10"
        );
        // transcription_model should default to whisper-1
        assert_eq!(
            config.openai.transcription_model, "whisper-1",
            "transcription_model should default to whisper-1"
        );
        // Removed fields (passthrough_enabled, passthrough_port) should be silently ignored
    }

    // ===========================================================================
    // Test: API error messages are truncated by characters, not bytes
    // ===========================================================================

    /// Sends `message` as an OpenAI style JSON error through `Logger::error_api`
    /// and returns the text shown in the activity log.
    fn displayed_api_error(message: &str) -> String {
        use crate::app_state::Logger;

        let (log_tx, mut log_rx) = tokio::sync::mpsc::channel(1);
        let body = serde_json::json!({ "error": { "message": message, "code": "other" } });
        Logger::new(log_tx, Default::default()).error_api(format!("API error: {}", body));
        log_rx
            .try_recv()
            .expect("error_api sends one log entry")
            .message
    }

    #[test]
    fn test_api_error_truncation_ascii() {
        let long = "a".repeat(130);
        assert_eq!(
            displayed_api_error(&long),
            format!("{}...", "a".repeat(117))
        );

        let short = "a".repeat(120);
        assert_eq!(displayed_api_error(&short), short);
    }

    #[test]
    fn test_api_error_truncation_multibyte() {
        // Byte 117 is inside the two byte 'é'.
        let accented = format!("{}é{}", "a".repeat(116), "b".repeat(20));
        assert_eq!(
            displayed_api_error(&accented),
            format!("{}é...", "a".repeat(116))
        );

        // 121 characters of three bytes each.
        let cjk = "語".repeat(121);
        assert_eq!(
            displayed_api_error(&cjk),
            format!("{}...", "語".repeat(117))
        );

        // 120 characters is short enough to show in full, even at 360 bytes.
        let cjk_short = "語".repeat(120);
        assert_eq!(displayed_api_error(&cjk_short), cjk_short);
    }

    // ===========================================================================
    // Test: Recording file names cut the transcription by characters
    // ===========================================================================

    /// Saves one recording with `transcription` into a fresh directory and
    /// returns the file name without the timestamp prefix.
    async fn saved_recording_name(test_name: &str, transcription: &str) -> String {
        use crate::recording_manager::RecordingManager;
        use std::fs;

        let dir =
            std::env::temp_dir().join(format!("babble_boop_{}_{}", test_name, std::process::id()));
        fs::remove_dir_all(&dir).ok();

        let result = RecordingManager::new(dir.clone(), 10)
            .save_recording(vec![0u8; 4], transcription)
            .await;
        let names: Vec<String> = fs::read_dir(&dir)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();

        // Clean up before asserting, so a failure does not leave files behind
        fs::remove_dir_all(&dir).ok();

        result.expect("save_recording should succeed");
        assert_eq!(names.len(), 1, "expected one recording, found {:?}", names);
        let (_timestamp, rest) = names[0]
            .split_once('_')
            .expect("file name has a timestamp prefix");
        rest.to_string()
    }

    #[tokio::test]
    async fn test_recording_name_ascii() {
        // The first 50 characters are kept, so " cut" is dropped.
        let transcription = "Hello, World! This is a test of the recording name cut";
        assert_eq!(
            saved_recording_name("ascii", transcription).await,
            "hello-world-this-is-a-test-of-the-recording-name.wav"
        );
    }

    #[tokio::test]
    async fn test_recording_name_multibyte() {
        // Byte 50 is inside the two byte 'é'.
        let accented = format!("{}é{}", "a".repeat(49), "b".repeat(20));
        assert_eq!(
            saved_recording_name("accented", &accented).await,
            format!("{}é.wav", "a".repeat(49))
        );

        // 60 characters of three bytes each; byte 50 is inside the 17th.
        let cjk = "日本語".repeat(20);
        assert_eq!(
            saved_recording_name("cjk", &cjk).await,
            format!("{}.wav", cjk.chars().take(50).collect::<String>())
        );
    }

    // ===========================================================================
    // Test: API client timeouts
    // ===========================================================================

    #[tokio::test]
    async fn test_api_client_times_out_when_server_never_responds() {
        use crate::api_client::client_with_timeouts;
        use std::time::Duration;
        use tokio::net::TcpListener;

        // Accept the connection and keep it open without sending a response.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            std::future::pending::<()>().await;
        });

        let client =
            client_with_timeouts(Duration::from_secs(5), Duration::from_millis(200)).unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            client.get(format!("http://{}/", addr)).send(),
        )
        .await;
        server.abort();

        let error = result
            .expect("request was not stopped by the client timeout")
            .expect_err("server never responds");
        assert!(
            error.is_timeout(),
            "expected a timeout error, got {}",
            error
        );
    }

    // ===========================================================================
    // Test: Shutdown interrupts in flight work
    // ===========================================================================

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_interrupts_chatbox_display_pause() {
        use crate::chatbox::send_to_chatbox;
        use crate::shutdown::Shutdown;
        use std::time::Duration;
        use tokio::net::UdpSocket;
        use tokio::time::{sleep, Instant};

        let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = receiver.local_addr().unwrap().port();
        config.osc.display_time = 30_000;
        config.osc.max_message_chunks = 10;
        // Three chunks, so the chatbox pauses 90 s in total.
        let message = "a".repeat(300);

        let shutdown = Shutdown::new();
        let requester = shutdown.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(1)).await;
            requester.request();
        });

        let start = Instant::now();
        let result = shutdown
            .run_until(send_to_chatbox(&message, &config, &socket))
            .await;

        assert!(result.is_none(), "shutdown did not stop the chatbox send");
        assert_eq!(start.elapsed(), Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_requested_before_wait_is_seen() {
        use crate::shutdown::Shutdown;
        use std::time::Duration;

        let shutdown = Shutdown::new();
        shutdown.request();

        assert!(shutdown.is_requested());
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            shutdown.run_until(std::future::pending::<()>()),
        )
        .await;
        assert_eq!(result, Ok(None));
    }

    #[tokio::test]
    async fn test_run_until_returns_output_without_shutdown() {
        use crate::shutdown::Shutdown;

        let shutdown = Shutdown::new();
        assert_eq!(shutdown.run_until(async { 7 }).await, Some(7));
        assert!(!shutdown.is_requested());
    }

    // ===========================================================================
    // Test: Translation request shape
    // ===========================================================================

    #[test]
    fn test_translation_request_sends_speech_as_user_message() {
        use crate::translation::ChatGptRequest;
        use serde_json::json;

        let spoken = "What time is it?";
        let request = ChatGptRequest::translation("gpt-4o-mini", "Japanese", spoken);
        let body = serde_json::to_value(&request).unwrap();

        assert_eq!(body["model"], "gpt-4o-mini");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        let instructions = messages[0]["content"].as_str().unwrap();
        assert!(instructions.contains("Japanese"));
        assert!(!instructions.contains(spoken));
        assert_eq!(messages[1], json!({"role": "user", "content": spoken}));
    }

    #[test]
    fn test_translation_request_token_estimate_counts_all_messages() {
        use crate::translation::ChatGptRequest;

        let silent = ChatGptRequest::translation("gpt-4o-mini", "Japanese", "");
        let spoken = ChatGptRequest::translation("gpt-4o-mini", "Japanese", &"a".repeat(400));

        // The instructions count, and so does the speech at 4 bytes a token.
        assert!(silent.approx_input_tokens() > 0);
        assert_eq!(
            spoken.approx_input_tokens() - silent.approx_input_tokens(),
            100
        );
    }

    // ===========================================================================
    // Test: Saved settings reach the processing loop
    // ===========================================================================

    #[test]
    fn test_config_update_applies_new_model_prices_and_keeps_total() {
        use crate::processing_loop::ProcessingServices;
        use std::time::Duration;

        let mut config = Config::default();
        config.openai.model = "gpt-4o-mini".to_string();
        config.openai.transcription_model = "whisper-1".to_string();
        let mut services = ProcessingServices::new(&config, &test_logger());
        services.price_estimator.total_cost = 1.25;

        config.openai.model = "gpt-4o".to_string();
        config.openai.transcription_model = "gpt-4o-mini-transcribe".to_string();
        services.apply_config(&config, &test_logger());

        let expected = PriceEstimator::new("gpt-4o", "gpt-4o-mini-transcribe");
        let old = PriceEstimator::new("gpt-4o-mini", "whisper-1");
        let minute = Duration::from_secs(60);
        let estimator = &services.price_estimator;
        assert_ne!(
            expected.estimate_translation_cost(1000, 500),
            old.estimate_translation_cost(1000, 500)
        );
        assert_ne!(
            expected.estimate_transcription_cost(minute),
            old.estimate_transcription_cost(minute)
        );
        assert_eq!(
            estimator.estimate_translation_cost(1000, 500),
            expected.estimate_translation_cost(1000, 500)
        );
        assert_eq!(
            estimator.estimate_transcription_cost(minute),
            expected.estimate_transcription_cost(minute)
        );
        // The running total stays in memory; it is not reloaded from disk.
        assert_eq!(estimator.total_cost, 1.25);
    }

    /// Logger whose entries nobody reads.
    fn test_logger() -> crate::app_state::Logger {
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        crate::app_state::Logger::new(log_tx, Default::default())
    }

    /// Activity log entries written by `body`.
    fn entries_logged_by(
        body: impl FnOnce(&crate::app_state::Logger),
    ) -> Vec<crate::app_state::LogEntry> {
        let (log_tx, mut log_rx) = tokio::sync::mpsc::channel(10);
        body(&crate::app_state::Logger::new(log_tx, Default::default()));
        std::iter::from_fn(|| log_rx.try_recv().ok()).collect()
    }

    #[test]
    fn test_unknown_model_pricing_is_logged() {
        use crate::processing_loop::ProcessingServices;

        let mut config = Config::default();
        config.openai.model = "my-finetuned-model".to_string();
        let entries = entries_logged_by(|logger| {
            ProcessingServices::new(&config, logger);
        });
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert!(entries[0].message.contains("'my-finetuned-model'"));

        let mut services = ProcessingServices::new(&Config::default(), &test_logger());
        config.openai.model = "gpt-4o".to_string();
        config.openai.transcription_model = "my-transcriber".to_string();
        let entries = entries_logged_by(|logger| services.apply_config(&config, logger));
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert!(entries[0].message.contains("'my-transcriber'"));
    }

    #[test]
    fn test_known_model_pricing_is_not_logged() {
        use crate::processing_loop::ProcessingServices;

        let config = Config::default();
        let entries = entries_logged_by(|logger| {
            ProcessingServices::new(&config, logger).apply_config(&config, logger);
        });
        assert!(entries.is_empty(), "{:?}", entries);
    }

    /// Time one more rate limiter wait after using up a budget of two
    /// requests and saving settings with `requests_per_minute` set to `limit`.
    async fn wait_after_saving_limit(limit: usize) -> std::time::Duration {
        use crate::processing_loop::ProcessingServices;
        use tokio::time::Instant;

        let mut config = Config::default();
        config.rate_limit.requests_per_minute = 2;
        let mut services = ProcessingServices::new(&config, &test_logger());
        services.rate_limiter.wait().await;
        services.rate_limiter.wait().await;

        config.rate_limit.requests_per_minute = limit;
        services.apply_config(&config, &test_logger());

        let start = Instant::now();
        services.rate_limiter.wait().await;
        start.elapsed()
    }

    #[tokio::test(start_paused = true)]
    async fn test_saving_settings_keeps_rate_limit_budget() {
        use std::time::Duration;

        // The budget is used up, so the next request waits for the window.
        assert!(wait_after_saving_limit(2).await >= Duration::from_secs(59));
    }

    #[tokio::test(start_paused = true)]
    async fn test_raised_rate_limit_applies_at_once() {
        use std::time::Duration;

        assert_eq!(wait_after_saving_limit(3).await, Duration::ZERO);
    }

    // ===========================================================================
    // Test: Processing thread failures reach the activity log
    // ===========================================================================

    fn logged_entries(
        body: impl FnOnce() -> Result<(), String>,
    ) -> Vec<crate::app_state::LogEntry> {
        entries_logged_by(|logger| {
            crate::app_state::run_logging_failure(logger, "Processing", body)
        })
    }

    #[test]
    fn test_processing_error_is_logged() {
        use crate::app_state::LogLevel;

        let entries = logged_entries(|| Err("Address already in use".to_string()));

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].level, LogLevel::Error);
        assert_eq!(
            entries[0].message,
            "Processing stopped: Address already in use"
        );
    }

    #[test]
    fn test_processing_panic_is_logged() {
        use crate::app_state::LogLevel;

        // A message formatted at run time is a String payload, a literal is
        // a &str.
        let what = String::from("poisoned");
        let entries = logged_entries(move || panic!("lock {}", what));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].level, LogLevel::Error);
        assert_eq!(entries[0].message, "Processing crashed: lock poisoned");

        let entries = logged_entries(|| panic!("no runtime"));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].message, "Processing crashed: no runtime");
    }

    #[test]
    fn test_processing_normal_exit_is_not_logged() {
        assert!(logged_entries(|| Ok(())).is_empty());
    }

    // ===========================================================================
    // Test: Changes made on other threads wake the GUI
    // ===========================================================================

    /// Whether `change` asks the egui context attached to a `GuiWaker` for a
    /// repaint. The GUI does not repaint on its own, so a change that does
    /// not ask stays hidden until the next mouse or keyboard input.
    fn wakes_gui(change: impl FnOnce(&crate::app_state::GuiWaker)) -> bool {
        use crate::app_state::GuiWaker;
        use eframe::egui;

        let ctx = egui::Context::default();
        let waker = GuiWaker::default();
        waker.attach(ctx.clone());
        assert!(!ctx.has_requested_repaint());
        change(&waker);
        ctx.has_requested_repaint()
    }

    #[test]
    fn test_log_entry_wakes_gui() {
        use crate::app_state::Logger;

        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        assert!(wakes_gui(|waker| Logger::new(
            log_tx.clone(),
            waker.clone()
        )
        .info("Transcription: hello")));
        assert!(wakes_gui(|waker| Logger::new(
            log_tx.clone(),
            waker.clone()
        )
        .success("Translation: hallo")));
        assert!(wakes_gui(|waker| Logger::new(
            log_tx.clone(),
            waker.clone()
        )
        .error("Processing stopped")));
        assert!(wakes_gui(|waker| Logger::new(
            log_tx.clone(),
            waker.clone()
        )
        .error_api("API error")));
    }

    #[test]
    fn test_cost_update_wakes_gui() {
        use crate::app_state::AppState;
        use eframe::egui;

        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        let app_state = AppState::new(Config::default(), cmd_tx, log_tx);
        let ctx = egui::Context::default();
        app_state.gui_waker.attach(ctx.clone());

        app_state.set_total_cost(0.25);

        assert!(ctx.has_requested_repaint());
    }

    // ===========================================================================
    // Test: Disabling translation clears the typing indicator
    // ===========================================================================

    #[tokio::test]
    async fn test_disabling_translation_turns_typing_indicator_off() {
        use crate::app_state::Logger;
        use crate::processing_loop::apply_enabled;
        use crate::typing_indicator::TypingIndicator;
        use rosc::{OscMessage, OscPacket, OscType};
        use std::sync::{Arc, RwLock};
        use std::time::Duration;
        use tokio::net::UdpSocket;

        let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = receiver.local_addr().unwrap().port();
        let indicator = TypingIndicator::new(socket, Arc::new(RwLock::new(config)));
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);

        apply_enabled(false, &indicator, &Logger::new(log_tx, Default::default())).await;

        let mut buf = [0u8; 256];
        let (len, _) = tokio::time::timeout(Duration::from_secs(1), receiver.recv_from(&mut buf))
            .await
            .expect("no typing indicator message was sent")
            .unwrap();
        let (_, packet) = rosc::decoder::decode_udp(&buf[..len]).unwrap();
        assert_eq!(
            packet,
            OscPacket::Message(OscMessage {
                addr: "/chatbox/typing".to_string(),
                args: vec![OscType::Bool(false)],
            })
        );
    }
}

#[cfg(test)]
mod gui_tests {
    use crate::app_state::{AppState, LogEntry, LogLevel};
    use crate::config::Config;
    use crate::gui::BabbleBoopApp;
    use eframe::egui;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn test_app_with_state() -> (BabbleBoopApp, Arc<AppState>) {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(Config::default(), cmd_tx, log_tx));
        (
            BabbleBoopApp::new(Arc::clone(&app_state), log_rx),
            app_state,
        )
    }

    fn test_app() -> (BabbleBoopApp, tokio::sync::mpsc::Sender<LogEntry>) {
        let (app, app_state) = test_app_with_state();
        (app, app_state.log_tx.clone())
    }

    fn raw_input() -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 600.0),
            )),
            ..Default::default()
        }
    }

    /// Run a few frames, so that the first frame layout passes are over,
    /// and return how long the GUI asks to wait before the next frame.
    /// `Duration::MAX` means it waits for input or a wake up.
    fn repaint_delay_after_frames(
        ctx: &egui::Context,
        mut run_ui: impl FnMut(&egui::Context),
    ) -> Duration {
        let mut output = ctx.run(raw_input(), &mut run_ui);
        for _ in 0..4 {
            output = ctx.run(raw_input(), &mut run_ui);
        }
        output.viewport_output[&egui::ViewportId::ROOT].repaint_delay
    }

    #[test]
    fn test_idle_gui_with_log_entries_does_not_repaint() {
        let (mut app, log_tx) = test_app();
        log_tx
            .try_send(LogEntry {
                timestamp: Instant::now(),
                message: "Starting audio recording...".to_string(),
                level: LogLevel::Info,
            })
            .unwrap();

        let delay = repaint_delay_after_frames(&egui::Context::default(), |ctx| app.ui(ctx));

        assert_eq!(delay, Duration::MAX);
    }

    #[test]
    fn test_status_message_repaints_when_it_expires() {
        let (mut app, _log_tx) = test_app();
        app.set_status_info("Settings saved successfully");

        let delay = repaint_delay_after_frames(&egui::Context::default(), |ctx| app.ui(ctx));

        assert!(delay > Duration::from_secs(2), "delay {:?}", delay);
        assert!(delay <= Duration::from_secs(3), "delay {:?}", delay);
    }

    #[test]
    fn test_audio_settings_keep_meters_live() {
        let (mut app, _log_tx) = test_app();

        let delay = repaint_delay_after_frames(&egui::Context::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        });

        assert!(delay <= Duration::from_millis(50), "delay {:?}", delay);
    }

    #[test]
    fn test_audio_settings_do_not_repaint_while_minimized() {
        let (mut app, _log_tx) = test_app();
        let ctx = egui::Context::default();
        let minimized = || {
            let mut input = raw_input();
            input
                .viewports
                .entry(egui::ViewportId::ROOT)
                .or_default()
                .minimized = Some(true);
            input
        };
        let mut run_ui = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        };

        let mut output = ctx.run(minimized(), &mut run_ui);
        for _ in 0..4 {
            output = ctx.run(minimized(), &mut run_ui);
        }

        assert_eq!(
            output.viewport_output[&egui::ViewportId::ROOT].repaint_delay,
            Duration::MAX
        );
    }

    /// Text of all labels painted in `output`. Labels outside the visible
    /// part of a scroll area are not painted.
    fn painted_text(output: &egui::FullOutput) -> Vec<String> {
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) => Some(text.galley.text().to_string()),
                _ => None,
            })
            .collect()
    }

    /// Run frames the way eframe does: one, then one more for each frame
    /// that asks for a repaint at once. Returns the last frame's output.
    fn run_until_idle(ctx: &egui::Context, app: &mut BabbleBoopApp) -> egui::FullOutput {
        for _ in 0..10 {
            let output = ctx.run(raw_input(), |ctx| app.ui(ctx));
            if output.viewport_output[&egui::ViewportId::ROOT].repaint_delay != Duration::ZERO {
                return output;
            }
        }
        panic!("the GUI still repaints continuously after 10 frames");
    }

    #[test]
    fn test_new_log_entry_is_visible_when_gui_goes_idle() {
        let (mut app, app_state) = test_app_with_state();
        let ctx = egui::Context::default();
        app_state.gui_waker.attach(ctx.clone());
        // More entries than the log shows, so the log scrolls.
        for i in 0..20 {
            app_state.logger.info(format!("entry {}", i));
        }
        run_until_idle(&ctx, &mut app);

        // The GUI is idle. The processing thread logs one more line.
        app_state.logger.info("Translation: hallo");
        assert!(ctx.has_requested_repaint());
        let output = run_until_idle(&ctx, &mut app);

        assert!(
            painted_text(&output).contains(&"Translation: hallo".to_string()),
            "{:?}",
            painted_text(&output)
        );
    }

    // ===========================================================================
    // Test: Meters and the toggle follow the theme
    // ===========================================================================

    fn themed_app(theme: crate::config::ThemeMode) -> (BabbleBoopApp, Arc<AppState>) {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let config = Config {
            theme,
            ..Config::default()
        };
        let app_state = Arc::new(AppState::new(config, cmd_tx, log_tx));
        (
            BabbleBoopApp::new(Arc::clone(&app_state), log_rx),
            app_state,
        )
    }

    /// Filled rectangles of exactly `width` x `height` points in `output`.
    fn painted_rect_fills(
        output: &egui::FullOutput,
        width: f32,
        height: f32,
    ) -> Vec<egui::Color32> {
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect)
                    if rect.rect.width() == width && rect.rect.height() == height =>
                {
                    Some(rect.fill)
                }
                _ => None,
            })
            .collect()
    }

    fn is_light(color: egui::Color32) -> bool {
        (u32::from(color.r()) + u32::from(color.g()) + u32::from(color.b())) / 3 > 128
    }

    #[test]
    fn test_meter_tracks_follow_the_theme() {
        use crate::config::ThemeMode;
        use std::sync::atomic::Ordering;

        for theme in [ThemeMode::Dark, ThemeMode::Light] {
            let (mut app, app_state) = themed_app(theme);
            // Recording shows the silence and duration meters. Level,
            // silence and duration are zero, so only the tracks are painted.
            app_state.is_recording.store(true, Ordering::Relaxed);
            let output = egui::Context::default().run(raw_input(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
            });

            let window_is_light = is_light(crate::theme::get_visuals(theme).panel_fill);
            let level_track = painted_rect_fills(&output, 200.0, 16.0);
            let small_tracks = painted_rect_fills(&output, 200.0, 10.0);
            // Gate, silence and duration.
            assert_eq!(small_tracks.len(), 3, "{:?}", theme);
            for track in level_track.iter().chain(&small_tracks) {
                assert_eq!(
                    is_light(*track),
                    window_is_light,
                    "{:?} theme meter track {:?}",
                    theme,
                    track
                );
            }
        }
    }

    #[test]
    fn test_toggle_off_track_follows_the_theme() {
        use crate::config::ThemeMode;
        use std::sync::atomic::Ordering;

        for theme in [ThemeMode::Dark, ThemeMode::Light] {
            let (mut app, app_state) = themed_app(theme);
            app_state.enabled.store(false, Ordering::Relaxed);
            let output = egui::Context::default().run(raw_input(), |ctx| app.ui(ctx));

            let window_is_light = is_light(crate::theme::get_visuals(theme).panel_fill);
            let toggle = painted_rect_fills(&output, 36.0, 20.0);
            assert_eq!(toggle.len(), 1, "{:?}", theme);
            assert_eq!(is_light(toggle[0]), window_is_light, "{:?} theme", theme);
        }
    }
}
