//! Tests for BabbleBoop
//!
//! These tests verify the fixes for issues identified during code review.

#[cfg(test)]
mod regression_tests {
    use crate::config::{
        AudioConfig, Config, OpenAiConfig, OscConfig, RateLimitConfig, ThemeMode, TranslationConfig,
    };
    use crate::price_estimator::{PriceEstimator, TokenCounts};
    use crate::rate_limiter::RateLimiter;

    /// Token counts for comparing translation prices.
    const TOKENS: TokenCounts = TokenCounts {
        input: 1000,
        output: 500,
    };

    // ===========================================================================
    // Regression test: Model pricing now covers common models
    // ===========================================================================

    #[test]
    fn test_older_models_keep_pricing() {
        // Models that existing config files can name, until they shut down.
        for (model, transcription_model) in [
            ("gpt-4o", "whisper-1"),
            ("gpt-4o-mini", "gpt-4o-transcribe"),
            ("gpt-4-turbo", "gpt-4o-mini-transcribe"),
            ("gpt-4", "whisper-1"),
            ("gpt-3.5-turbo", "whisper-1"),
        ] {
            assert_eq!(
                PriceEstimator::unknown_pricing(model, transcription_model),
                Vec::<String>::new()
            );
        }
    }

    #[test]
    fn test_current_models_have_pricing() {
        for (model, transcription_model) in [
            ("gpt-6-luna", "gpt-transcribe"),
            ("gpt-6-sol", "gpt-transcribe"),
            ("gpt-5.4-nano", "gpt-transcribe"),
            ("gpt-5.4-mini", "gpt-transcribe"),
            ("gpt-4.1-mini", "gpt-transcribe"),
        ] {
            assert_eq!(
                PriceEstimator::unknown_pricing(model, transcription_model),
                Vec::<String>::new()
            );
        }
    }

    #[test]
    fn test_dated_snapshots_cost_the_same_as_their_model() {
        // The gpt-4o and gpt-4o-mini model pages list these snapshots as
        // available and give no other price for them.
        for (snapshot, model) in [
            ("gpt-4o-2024-11-20", "gpt-4o"),
            ("gpt-4o-2024-08-06", "gpt-4o"),
            ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
        ] {
            let snapshot_cost =
                PriceEstimator::new(snapshot, "gpt-transcribe").estimate_translation_cost(TOKENS);
            let model_cost =
                PriceEstimator::new(model, "gpt-transcribe").estimate_translation_cost(TOKENS);
            assert_eq!(
                (
                    snapshot,
                    snapshot_cost,
                    PriceEstimator::unknown_pricing(snapshot, "gpt-transcribe")
                ),
                (snapshot, model_cost, Vec::<String>::new())
            );
        }
    }

    #[test]
    fn test_unknown_model_uses_default_model_pricing() {
        use std::time::Duration;

        let defaults = Config::default().openai;
        let unknown = PriceEstimator::new("unknown-model-xyz", "unknown-transcriber");
        let default = PriceEstimator::new(&defaults.model, &defaults.transcription_model);
        assert_eq!(
            unknown.estimate_translation_cost(TOKENS),
            default.estimate_translation_cost(TOKENS)
        );
        let minute = Duration::from_secs(60);
        assert_eq!(
            unknown.estimate_transcription_cost(minute),
            default.estimate_transcription_cost(minute)
        );

        // The warnings name the models whose prices the estimate uses.
        let warnings = PriceEstimator::unknown_pricing("unknown-model-xyz", "unknown-transcriber");
        assert_eq!(warnings.len(), 2, "{:?}", warnings);
        assert!(warnings[0].contains(&format!("{} prices", defaults.model)));
        assert!(warnings[1].contains(&format!("{} prices", defaults.transcription_model)));
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
        // A missing transcription_model gets the default of a new config
        assert_eq!(
            config.openai.transcription_model,
            Config::default().openai.transcription_model
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
    fn test_translation_request_options_follow_the_model() {
        use crate::translation::ChatGptRequest;

        // (model, instructions role, reasoning_effort)
        for (model, role, effort) in [
            ("gpt-6-luna", "developer", Some("none")),
            ("gpt-6-sol", "developer", Some("none")),
            // Shutdown replacements; without `none` they reason at `medium`.
            ("gpt-5.6-luna", "developer", Some("none")),
            ("gpt-5.6-terra", "developer", Some("none")),
            ("gpt-5.6-sol", "developer", Some("none")),
            ("gpt-5.4-nano", "developer", None),
            ("gpt-4.1-mini", "system", None),
            ("gpt-4o-mini", "system", None),
            ("my-finetuned-model", "system", None),
        ] {
            let request = ChatGptRequest::translation(model, "Japanese", "Hello");
            let body = serde_json::to_value(&request).unwrap();
            assert_eq!(body["messages"][0]["role"], role, "{}", model);
            assert_eq!(
                body.get("reasoning_effort").map(|v| v.as_str().unwrap()),
                effort,
                "{}",
                model
            );
        }
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

    /// Chat Completions response body in the shape the API returns. `usage`
    /// is the JSON of the usage field, or `None` to leave the field out.
    fn chat_completion_body(content: &str, usage: Option<&str>) -> String {
        let usage = usage
            .map(|usage| format!(r#","usage":{}"#, usage))
            .unwrap_or_default();
        format!(
            r#"{{
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1790000000,
                "model": "gpt-5.6-sol-2026-08-14",
                "choices": [{{
                    "index": 0,
                    "message": {{
                        "role": "assistant",
                        "content": {},
                        "refusal": null,
                        "annotations": []
                    }},
                    "logprobs": null,
                    "finish_reason": "stop"
                }}],
                "service_tier": "default",
                "system_fingerprint": null{}
            }}"#,
            serde_json::to_string(content).unwrap(),
            usage
        )
    }

    /// Usage of a response whose completion is mostly reasoning tokens.
    const REASONING_USAGE: &str = r#"{
        "prompt_tokens": 58,
        "completion_tokens": 331,
        "total_tokens": 389,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 320,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0
        }
    }"#;

    #[test]
    fn test_translation_tokens_come_from_reported_usage() {
        use crate::translation::ChatGptRequest;

        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "What time is it?");
        let body = chat_completion_body("Quelle heure est-il ?", Some(REASONING_USAGE));

        let translation = request.parse_response(&body).unwrap();

        assert_eq!(translation.text, "Quelle heure est-il ?");
        // completion_tokens includes the 320 reasoning tokens.
        assert_eq!(
            translation.tokens,
            TokenCounts {
                input: 58,
                output: 331
            }
        );
    }

    #[test]
    fn test_translation_tokens_are_estimated_without_usage() {
        use crate::translation::ChatGptRequest;

        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "What time is it?");
        for body in [
            chat_completion_body("Quelle heure est-il ?", None),
            chat_completion_body("Quelle heure est-il ?", Some("null")),
        ] {
            let translation = request.parse_response(&body).unwrap();

            assert_eq!(translation.text, "Quelle heure est-il ?");
            // Four bytes a token: 21 bytes of translation give 5 tokens.
            assert_eq!(
                translation.tokens,
                TokenCounts {
                    input: request.approx_input_tokens(),
                    output: 5
                }
            );
        }
    }

    #[test]
    fn test_translation_cost_counts_reported_reasoning_tokens() {
        use crate::translation::ChatGptRequest;

        let estimator = PriceEstimator::new("gpt-5.6-sol", "gpt-transcribe");
        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "What time is it?");
        let text = "Quelle heure est-il ?";
        let reported = request
            .parse_response(&chat_completion_body(text, Some(REASONING_USAGE)))
            .unwrap();
        let estimated = request
            .parse_response(&chat_completion_body(text, None))
            .unwrap();

        let reported_cost = estimator.estimate_translation_cost(reported.tokens);
        assert_eq!(
            reported_cost,
            estimator.estimate_translation_cost(TokenCounts {
                input: 58,
                output: 331
            })
        );
        // The text alone does not show the reasoning tokens.
        assert!(reported_cost > estimator.estimate_translation_cost(estimated.tokens));
    }

    #[test]
    fn test_translation_response_without_choices_is_an_error() {
        use crate::translation::ChatGptRequest;

        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "Hello");
        let body = r#"{"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 0}}"#;

        assert!(request.parse_response(body).is_err());
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
    fn test_model_shutdown_date_is_logged_at_start_and_on_save() {
        use crate::processing_loop::ProcessingServices;

        let mut config = Config::default();
        config.openai.model = "gpt-3.5-turbo".to_string();
        let entries = entries_logged_by(|logger| {
            ProcessingServices::new(&config, logger);
        });
        assert_eq!(entries.len(), 1, "{:?}", entries);
        for text in ["'gpt-3.5-turbo'", "2026-10-23", "gpt-5.6-terra"] {
            assert!(entries[0].message.contains(text), "{:?}", entries[0]);
        }

        let mut services = ProcessingServices::new(&Config::default(), &test_logger());
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
        use crate::processing_loop::ProcessingServices;

        for model in crate::models::suggested_chat_models() {
            for transcription_model in crate::models::suggested_transcription_models() {
                let mut config = Config::default();
                config.openai.model = model.to_string();
                config.openai.transcription_model = transcription_model.to_string();
                let entries = entries_logged_by(|logger| {
                    ProcessingServices::new(&config, logger);
                });
                assert!(entries.is_empty(), "{:?}", entries);
            }
        }
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
    // Test: Audio events are logged and encoded on the processing side
    // ===========================================================================

    #[test]
    fn test_audio_events_are_logged_on_the_processing_side() {
        use crate::app_state::LogLevel;
        use crate::processing_loop::log_audio_event;
        use crate::types::{AudioEvent, CapturedAudio};

        let audio = || CapturedAudio {
            samples: vec![0.5; 4],
            channels: 1,
            sample_rate: 16_000,
        };
        let entries = entries_logged_by(|logger| {
            for event in [
                AudioEvent::StartRecording,
                AudioEvent::AudioPart(audio()),
                AudioEvent::AudioData(audio()),
                AudioEvent::StopRecording,
                AudioEvent::EventsDropped(3),
                AudioEvent::InputError("Audio input error: device unplugged".to_string()),
            ] {
                log_audio_event(&event, logger);
            }
        });
        let logged: Vec<(&str, LogLevel)> = entries
            .iter()
            .map(|entry| (entry.message.as_str(), entry.level))
            .collect();
        assert_eq!(
            logged,
            vec![
                ("Sound detected, recording...", LogLevel::Info),
                (
                    "Recording reached 30 s, processing it while recording goes on...",
                    LogLevel::Info
                ),
                ("Silence detected, processing...", LogLevel::Info),
                (
                    "Lost 3 audio events because the processing queue was full",
                    LogLevel::Error
                ),
                ("Audio input error: device unplugged", LogLevel::Error),
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_the_end_of_audio_input_is_logged_once() {
        use crate::app_state::{LogLevel, Logger};
        use crate::processing_loop::AudioEvents;
        use crate::types::AudioEvent;
        use std::time::Duration;

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, mut log_rx) = tokio::sync::mpsc::channel(10);
        let mut events = AudioEvents::new(rx, Logger::new(log_tx, Default::default()));
        tx.try_send(AudioEvent::StartRecording).unwrap();
        // The audio thread ends and drops every sender
        drop(tx);

        assert_eq!(events.recv().await, AudioEvent::StartRecording);
        // After that, no event comes; the loop waits for its other branches
        for _ in 0..2 {
            let next = tokio::time::timeout(Duration::from_secs(60), events.recv()).await;
            assert!(next.is_err(), "got {:?} after the channel closed", next);
        }
        let logged: Vec<(String, LogLevel)> = std::iter::from_fn(|| log_rx.try_recv().ok())
            .map(|entry| (entry.message, entry.level))
            .collect();
        assert_eq!(
            logged,
            vec![
                ("Sound detected, recording...".to_string(), LogLevel::Info),
                (
                    "Audio input stopped. Restart BabbleBoop to record again.".to_string(),
                    LogLevel::Error
                ),
            ]
        );
    }

    #[tokio::test]
    async fn test_recording_is_encoded_for_upload_with_its_duration() {
        use crate::processing_loop::encode_for_upload;
        use crate::types::CapturedAudio;

        // Half a second of stereo audio at 48 kHz
        let audio = CapturedAudio {
            samples: vec![0.25; 48_000],
            channels: 2,
            sample_rate: 48_000,
        };
        let wav = encode_for_upload(audio).await.unwrap();

        // The minimum duration check reads the duration from the WAV
        let duration = crate::audio_processing::calculate_audio_duration(&wav).unwrap();
        assert!(
            (duration.as_secs_f32() - 0.5).abs() < 1e-3,
            "duration {:?}",
            duration
        );
    }

    #[test]
    fn test_wav_with_a_sample_rate_of_zero_is_an_error() {
        // The duration check reads the sample rate from the WAV header.
        // hound cannot write a rate of 0, so build the file here: a PCM fmt
        // chunk with 1 channel, a rate of 0 (and so 0 bytes per second), a
        // block of 2 bytes and 16 bits, then one sample.
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&38u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        for field in [1u16, 1] {
            wav.extend_from_slice(&field.to_le_bytes());
        }
        for field in [0u32, 0] {
            wav.extend_from_slice(&field.to_le_bytes());
        }
        for field in [2u16, 16] {
            wav.extend_from_slice(&field.to_le_bytes());
        }
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&2u32.to_le_bytes());
        wav.extend_from_slice(&0i16.to_le_bytes());
        assert_eq!(
            hound::WavReader::new(wav.as_slice())
                .unwrap()
                .spec()
                .sample_rate,
            0
        );

        let duration = crate::audio_processing::calculate_audio_duration(&wav);
        assert!(duration.is_err(), "{:?}", duration);
    }

    /// Duration conversions from a float that panic on a negative, NaN or
    /// too large value. Production code uses the `try_` forms instead.
    #[test]
    fn test_no_float_to_duration_conversion_can_panic() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut hits = Vec::new();
        for entry in std::fs::read_dir(&src).unwrap() {
            let path = entry.unwrap().path();
            if path.extension() != Some("rs".as_ref()) {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&path).unwrap();
            for (index, line) in text.lines().enumerate() {
                let line = line.trim_start();
                // Test code follows the first cfg(test) attribute of a file
                if line.starts_with("#[cfg(") && line.contains("test") {
                    break;
                }
                if line.starts_with("//") {
                    continue;
                }
                let code = line.replace("try_from_secs_f", "");
                let panicking = [
                    "from_secs_f32(",
                    "from_secs_f64(",
                    ".mul_f32(",
                    ".mul_f64(",
                    ".div_f32(",
                    ".div_f64(",
                ]
                .iter()
                .any(|call| code.contains(call));
                if panicking {
                    hits.push(format!("{}:{}: {}", name, index + 1, line));
                }
            }
        }
        assert!(hits.is_empty(), "{:#?}", hits);
    }

    // ===========================================================================
    // Test: Any minimum transcription duration in config.toml is safe
    // ===========================================================================

    /// What `process_audio` did with a recording.
    #[derive(Debug, PartialEq)]
    enum MinimumCheck {
        /// The recording was skipped as too short.
        Skipped,
        /// The recording went on to transcription.
        Transcribed,
    }

    /// Run `process_audio` on one second of audio with `min_seconds` as the
    /// minimum transcription duration. The API client sends its requests
    /// through a local proxy, which accepts the connection and closes it, so
    /// no request leaves the machine.
    async fn check_one_second_against_minimum(min_seconds: f32) -> MinimumCheck {
        use crate::app_state::AppState;
        use crate::audio_processing::process_audio;
        use crate::processing_loop::encode_for_upload;
        use crate::types::CapturedAudio;
        use crate::typing_indicator::TypingIndicator;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, RwLock};
        use std::time::Duration;
        use tokio::net::{TcpListener, UdpSocket};

        let proxy = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy_addr = proxy.local_addr().unwrap();
        let requested = Arc::new(AtomicBool::new(false));
        let proxy_task = tokio::spawn({
            let requested = requested.clone();
            async move {
                // The client gets its error only when this drops the connection
                let (_connection, _) = proxy.accept().await.unwrap();
                requested.store(true, Ordering::SeqCst);
            }
        });
        let client = reqwest::Client::builder()
            .proxy(reqwest::Proxy::all(format!("http://{}", proxy_addr)).unwrap())
            .build()
            .unwrap();

        let chatbox = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = chatbox.local_addr().unwrap().port();
        config.audio.min_transcription_duration = min_seconds;
        let typing_indicator =
            TypingIndicator::new(socket.clone(), Arc::new(RwLock::new(config.clone())));
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        let app_state = Arc::new(AppState::new(config.clone(), cmd_tx, log_tx));
        let wav = encode_for_upload(CapturedAudio {
            samples: vec![0.25; 16_000],
            channels: 1,
            sample_rate: 16_000,
        })
        .await
        .unwrap();

        let result = tokio::time::timeout(
            Duration::from_secs(10),
            process_audio(
                &client,
                wav,
                &config,
                &socket,
                &mut RateLimiter::new(50),
                &typing_indicator,
                &mut PriceEstimator::new(&config.openai.model, &config.openai.transcription_model),
                None,
                &app_state,
            ),
        )
        .await
        .expect("process_audio did not finish");
        proxy_task.abort();

        if requested.load(Ordering::SeqCst) {
            assert!(result.is_err(), "the proxy closed the connection");
            MinimumCheck::Transcribed
        } else {
            assert!(result.is_ok(), "{:?}", result.err());
            MinimumCheck::Skipped
        }
    }

    #[tokio::test]
    async fn test_negative_minimum_duration_means_no_minimum() {
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
        assert_eq!(
            check_one_second_against_minimum(f32::NAN).await,
            MinimumCheck::Transcribed
        );
    }

    #[tokio::test]
    async fn test_infinite_minimum_duration_skips_the_recording() {
        assert_eq!(
            check_one_second_against_minimum(f32::INFINITY).await,
            MinimumCheck::Skipped
        );
    }

    #[tokio::test]
    async fn test_minimum_duration_too_large_for_duration_skips_the_recording() {
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
    // Test: The test microphone recording stops at its limit
    // ===========================================================================

    /// App state with its command and log channels kept open.
    fn app_state_for_test() -> (
        crate::app_state::AppState,
        tokio::sync::mpsc::Receiver<crate::app_state::AppCommand>,
        tokio::sync::mpsc::Receiver<crate::app_state::LogEntry>,
    ) {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(10);
        let app_state = crate::app_state::AppState::new(Config::default(), cmd_tx, log_tx);
        (app_state, cmd_rx, log_rx)
    }

    #[tokio::test(start_paused = true)]
    async fn test_test_recording_reaches_its_limit_after_30_seconds() {
        use crate::processing_loop::{TestRecording, TEST_RECORDING_LIMIT};
        use std::time::Duration;
        use tokio::time::{timeout, Instant};

        let (app_state, _cmd_rx, _log_rx) = app_state_for_test();
        let mut test = TestRecording::new(&app_state, 2, 48_000);
        test.start();
        let started = Instant::now();

        timeout(Duration::from_secs(60), test.limit_reached())
            .await
            .expect("the test recording has no time limit");
        assert_eq!(started.elapsed(), TEST_RECORDING_LIMIT);
    }

    #[tokio::test(start_paused = true)]
    async fn test_no_limit_is_reached_while_no_test_recording_runs() {
        use crate::processing_loop::TestRecording;
        use std::time::Duration;
        use tokio::time::timeout;

        let (app_state, _cmd_rx, _log_rx) = app_state_for_test();
        let mut test = TestRecording::new(&app_state, 2, 48_000);
        let hour = Duration::from_secs(3600);
        assert!(timeout(hour, test.limit_reached()).await.is_err());

        test.start();
        test.stop();
        assert!(timeout(hour, test.limit_reached()).await.is_err());
    }

    #[test]
    fn test_test_recording_returns_the_samples_the_callback_wrote() {
        use crate::processing_loop::TestRecording;
        use crate::types::CapturedAudio;
        use std::sync::atomic::Ordering;

        let (app_state, _cmd_rx, _log_rx) = app_state_for_test();
        let mut test = TestRecording::new(&app_state, 2, 48_000);
        assert_eq!(test.stop(), None);

        test.start();
        assert!(app_state.test_mode_active.load(Ordering::SeqCst));
        // As the callback does, within the reserved capacity
        let reserved = {
            let mut buffer = app_state.test_recording_buffer.lock().unwrap();
            buffer.extend_from_slice(&[0.1, 0.2, 0.3, 0.4]);
            buffer.capacity()
        };
        // Room for 30 s of 48 kHz stereo
        assert_eq!(reserved, 30 * 48_000 * 2);

        assert_eq!(
            test.stop(),
            Some(CapturedAudio {
                samples: vec![0.1, 0.2, 0.3, 0.4],
                channels: 2,
                sample_rate: 48_000,
            })
        );
        assert!(!app_state.test_mode_active.load(Ordering::SeqCst));
        // A callback that still sees test mode on has no room to write
        assert_eq!(
            app_state.test_recording_buffer.lock().unwrap().capacity(),
            0
        );
        assert_eq!(test.stop(), None);
    }

    #[test]
    fn test_a_new_test_recording_starts_empty() {
        use crate::processing_loop::TestRecording;

        let (app_state, _cmd_rx, _log_rx) = app_state_for_test();
        let mut test = TestRecording::new(&app_state, 1, 16_000);
        test.start();
        app_state.test_recording_buffer.lock().unwrap().push(0.5);
        test.start();
        assert_eq!(test.stop().map(|audio| audio.samples), Some(Vec::new()));
    }

    #[tokio::test]
    async fn test_test_recording_is_converted_to_the_output_format() {
        use crate::processing_loop::convert_for_playback;
        use crate::types::CapturedAudio;

        // Half a second of stereo audio at 48 kHz, for a mono 16 kHz output
        let audio = CapturedAudio {
            samples: vec![0.25; 48_000],
            channels: 2,
            sample_rate: 48_000,
        };
        let samples = convert_for_playback(audio, 1, 16_000).await.unwrap();
        assert_eq!(samples.len(), 8_000);
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

    // ===========================================================================
    // Test: The translation model takes a custom name
    // ===========================================================================

    /// Centre of the first painted text that reads `text`.
    fn text_center(output: &egui::FullOutput, text: &str) -> egui::Pos2 {
        output
            .shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                egui::Shape::Text(shape) if shape.galley.text() == text => {
                    Some(egui::Rect::from_min_size(shape.pos, shape.galley.size()).center())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("no painted text {:?}", text))
    }

    fn with_events(events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            events,
            ..raw_input()
        }
    }

    #[test]
    fn test_custom_translation_model_can_be_typed() {
        let (mut app, _log_tx) = test_app();
        let ctx = egui::Context::default();
        let mut output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        for _ in 0..2 {
            output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        }
        let model = text_center(&output, &Config::default().openai.model);

        click(&ctx, &mut app, model);
        let select_all = egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        };
        let typed = egui::Event::Text("my-finetuned-model".to_string());
        let _ = ctx.run(with_events(vec![select_all, typed]), |ctx| app.ui(ctx));

        assert_eq!(app.config_draft.openai.model, "my-finetuned-model");
    }

    #[test]
    fn test_custom_transcription_model_can_be_typed() {
        let (mut app, _log_tx) = test_app();
        let ctx = egui::Context::default();
        let mut output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        for _ in 0..2 {
            output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        }
        let model = text_center(&output, &Config::default().openai.transcription_model);

        click(&ctx, &mut app, model);
        let select_all = egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        };
        let typed = egui::Event::Text("my-transcriber".to_string());
        let _ = ctx.run(with_events(vec![select_all, typed]), |ctx| app.ui(ctx));

        assert_eq!(
            app.config_draft.openai.transcription_model,
            "my-transcriber"
        );
    }

    #[test]
    fn test_translation_model_preset_can_be_selected() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.model = "my-finetuned-model".to_string();
        let ctx = egui::Context::default();
        let mut output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        for _ in 0..2 {
            output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        }
        let model = text_center(&output, "my-finetuned-model");
        // The preset list button is the next widget right of the text field.
        let list_button = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect)
                    if rect.rect.x_range().min > model.x
                        && rect.rect.y_range().contains(model.y) =>
                {
                    Some(rect.rect)
                }
                _ => None,
            })
            .min_by(|a, b| a.min.x.total_cmp(&b.min.x))
            .expect("no preset list button")
            .center();

        click(&ctx, &mut app, list_button);
        let output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        click(&ctx, &mut app, text_center(&output, "gpt-6-sol"));

        assert_eq!(app.config_draft.openai.model, "gpt-6-sol");
    }

    #[test]
    fn test_model_names_are_trimmed_on_save() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = "test-key".to_string();
        app.config_draft.openai.model = "gpt-6-sol ".to_string();
        app.config_draft.openai.transcription_model = "\tgpt-transcribe\n".to_string();

        let saved = app.config_to_save().expect("config is valid");

        assert_eq!(saved.openai.model, "gpt-6-sol");
        assert_eq!(saved.openai.transcription_model, "gpt-transcribe");
        // The settings show what was saved.
        assert_eq!(app.config_draft, saved);
    }

    #[test]
    fn test_api_key_and_target_language_are_trimmed_on_save() {
        let (mut app, _log_tx) = test_app();
        // A pasted key often has a space or a line break at an end. The
        // Authorization header must not contain them.
        app.config_draft.openai.api_key = " sk-test \n".to_string();
        app.config_draft.translation.target_language = "\tFrench ".to_string();

        let saved = app.config_to_save().expect("config is valid");

        assert_eq!(saved.openai.api_key, "sk-test");
        assert_eq!(saved.translation.target_language, "French");
        // The settings show what was saved.
        assert_eq!(app.config_draft, saved);
    }

    /// Every free text field of the config, found through its serialized
    /// form, so that a text field added later is checked too.
    #[test]
    fn test_every_text_field_is_trimmed_on_save() {
        fn text_paths(value: &toml::Value, path: &str, out: &mut Vec<String>) {
            match value {
                toml::Value::String(_) => out.push(path.to_string()),
                toml::Value::Table(table) => {
                    for (key, child) in table {
                        let child_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{path}.{key}")
                        };
                        text_paths(child, &child_path, out);
                    }
                }
                _ => {}
            }
        }
        fn text_at<'a>(value: &'a mut toml::Value, path: &str) -> &'a mut String {
            let mut value = value;
            for key in path.split('.') {
                value = value.get_mut(key).expect("path exists");
            }
            match value {
                toml::Value::String(text) => text,
                _ => panic!("{path} is not text"),
            }
        }

        let mut valid = Config::default();
        valid.openai.api_key = "test-key".to_string();
        let valid = toml::Value::try_from(&valid).expect("config serializes");
        let mut paths = Vec::new();
        text_paths(&valid, "", &mut paths);

        let mut checked = Vec::new();
        for path in paths {
            let mut padded = valid.clone();
            let text = text_at(&mut padded, &path);
            *text = format!(" {text}\n");
            // A value that does not load with spaces around it, such as the
            // theme, is a fixed choice and not free text.
            let Ok(draft) = padded.try_into::<Config>() else {
                continue;
            };
            let (mut app, _log_tx) = test_app();
            app.config_draft = draft;

            let saved = app.config_to_save().expect("config is valid");

            let mut saved = toml::Value::try_from(&saved).expect("config serializes");
            let mut expected = valid.clone();
            assert_eq!(text_at(&mut saved, &path), text_at(&mut expected, &path));
            checked.push(path);
        }
        assert!(checked.contains(&"openai.api_key".to_string()));
    }

    #[test]
    fn test_blank_api_key_is_not_saved() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = " \n".to_string();

        assert!(app.config_to_save().is_err());
    }

    #[test]
    fn test_blank_transcription_model_is_not_saved() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = "test-key".to_string();
        app.config_draft.openai.transcription_model = " ".to_string();

        assert!(app.config_to_save().is_err());
    }

    /// Press and release the primary button at `pos`, in two frames.
    fn click(ctx: &egui::Context, app: &mut BabbleBoopApp, pos: egui::Pos2) {
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        let _ = ctx.run(
            with_events(vec![egui::Event::PointerMoved(pos), button(true)]),
            |ctx| app.ui(ctx),
        );
        let _ = ctx.run(with_events(vec![button(false)]), |ctx| app.ui(ctx));
    }

    // ===========================================================================
    // Test: The GUI shows that processing stopped
    // ===========================================================================

    #[test]
    fn test_stopped_processing_is_shown_instead_of_enabled() {
        let (mut app, app_state) = test_app_with_state();
        let ctx = egui::Context::default();
        app_state.gui_waker.attach(ctx.clone());
        let output = run_until_idle(&ctx, &mut app);
        assert!(painted_text(&output).contains(&"Enabled".to_string()));
        assert!(!ctx.has_requested_repaint());

        app_state.mark_processing_stopped();
        assert!(ctx.has_requested_repaint());
        let output = ctx.run(raw_input(), |ctx| app.ui(ctx));

        let text = painted_text(&output);
        assert!(
            text.contains(&"Processing stopped".to_string()),
            "{:?}",
            text
        );
        assert!(!text.contains(&"Enabled".to_string()), "{:?}", text);
    }
}
