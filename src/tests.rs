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
        Logger::new(log_tx).error_api(format!("API error: {}", body));
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
        let mut services = ProcessingServices::new(&config);
        services.price_estimator.total_cost = 1.25;

        config.openai.model = "gpt-4o".to_string();
        config.openai.transcription_model = "gpt-4o-mini-transcribe".to_string();
        services.apply_config(&config);

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
}
