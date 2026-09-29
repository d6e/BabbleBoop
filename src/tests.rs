//! Tests for BabbleBoop
//!
//! These tests verify the fixes for issues identified during code review.

#[cfg(test)]
pub(crate) mod regression_tests {
    use crate::config::{
        AudioConfig, Config, OpenAiConfig, OscConfig, RateLimitConfig, ThemeMode, TranslationConfig,
    };
    use crate::data_dir::DataDir;
    use crate::price_estimator::{PriceEstimator, TokenCounts};
    use crate::rate_limiter::RateLimiter;

    /// Token counts for comparing translation prices.
    const TOKENS: TokenCounts = TokenCounts {
        input: 1000,
        output: 500,
    };

    /// A data folder that does not exist, for tests that save no cost and
    /// no recordings. A save there fails instead of writing to the working
    /// directory.
    fn missing_data_dir() -> DataDir {
        DataDir::new(
            std::env::temp_dir().join(format!("babble_boop_no_data_dir_{}", std::process::id())),
        )
    }

    /// An estimator for the prices of these models.
    fn price_estimator(model: &str, transcription_model: &str) -> PriceEstimator {
        PriceEstimator::new(missing_data_dir().cost_file(), model, transcription_model)
    }

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

    /// What a config that names `model` as the translation model gets: the
    /// translation cost, the price and shutdown warnings, and the
    /// instructions role and reasoning effort of the request. Warnings name
    /// the model as `shown_as`, so that a snapshot and its model compare
    /// equal.
    fn chat_model_behaviour(
        model: &str,
        shown_as: &str,
    ) -> (
        String,
        f64,
        Vec<String>,
        Vec<String>,
        String,
        Option<String>,
    ) {
        use crate::translation::ChatGptRequest;

        let rename = |warnings: Vec<String>| -> Vec<String> {
            warnings
                .into_iter()
                .map(|warning| warning.replace(&format!("'{}'", model), &format!("'{}'", shown_as)))
                .collect()
        };
        let body =
            serde_json::to_value(ChatGptRequest::translation(model, "Japanese", "Hello")).unwrap();
        (
            shown_as.to_string(),
            price_estimator(model, "gpt-transcribe").estimate_translation_cost(TOKENS),
            rename(PriceEstimator::unknown_pricing(model, "gpt-transcribe")),
            rename(crate::models::shutdown_warnings(model, "gpt-transcribe")),
            body["messages"][0]["role"].as_str().unwrap().to_string(),
            body.get("reasoning_effort")
                .map(|effort| effort.as_str().unwrap().to_string()),
        )
    }

    /// Like `chat_model_behaviour`, for the transcription model.
    fn transcription_model_behaviour(
        model: &str,
        shown_as: &str,
    ) -> (String, f64, Vec<String>, Vec<String>) {
        let rename = |warnings: Vec<String>| -> Vec<String> {
            warnings
                .into_iter()
                .map(|warning| warning.replace(&format!("'{}'", model), &format!("'{}'", shown_as)))
                .collect()
        };
        (
            shown_as.to_string(),
            price_estimator("gpt-6-luna", model)
                .estimate_transcription_cost(std::time::Duration::from_secs(60)),
            rename(PriceEstimator::unknown_pricing("gpt-6-luna", model)),
            rename(crate::models::shutdown_warnings("gpt-6-luna", model)),
        )
    }

    #[test]
    fn test_dated_snapshots_behave_as_their_model() {
        // Each model page lists these snapshots under "Snapshots". The
        // pricing and deprecations pages give no other price or shutdown
        // date for them than for their model. gpt-5.6 is an alias: the
        // gpt-5.6-sol page says it routes requests to GPT-5.6 Sol.
        let chat = [
            ("gpt-4.1-nano-2025-04-14", "gpt-4.1-nano"),
            ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini"),
            ("gpt-4.1-2025-04-14", "gpt-4.1"),
            ("gpt-5.4-mini-2026-03-17", "gpt-5.4-mini"),
            ("gpt-5.4-nano-2026-03-17", "gpt-5.4-nano"),
            ("gpt-5.6", "gpt-5.6-sol"),
            ("gpt-4o-2024-11-20", "gpt-4o"),
            ("gpt-4o-2024-08-06", "gpt-4o"),
            ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
            ("gpt-4-turbo-2024-04-09", "gpt-4-turbo"),
            ("gpt-4-0613", "gpt-4"),
            ("gpt-3.5-turbo-0125", "gpt-3.5-turbo"),
        ];
        let chat_snapshots: Vec<_> = chat
            .iter()
            .map(|(snapshot, _)| chat_model_behaviour(snapshot, snapshot))
            .collect();
        let chat_models: Vec<_> = chat
            .iter()
            .map(|(snapshot, model)| chat_model_behaviour(model, snapshot))
            .collect();

        // The deprecations page gives the shutdown date of
        // gpt-4o-mini-transcribe and does not list this snapshot on its own.
        let transcription = [(
            "gpt-4o-mini-transcribe-2025-12-15",
            "gpt-4o-mini-transcribe",
        )];
        let transcription_snapshots: Vec<_> = transcription
            .iter()
            .map(|(snapshot, _)| transcription_model_behaviour(snapshot, snapshot))
            .collect();
        let transcription_models: Vec<_> = transcription
            .iter()
            .map(|(snapshot, model)| transcription_model_behaviour(model, snapshot))
            .collect();

        assert_eq!(
            (chat_snapshots, transcription_snapshots),
            (chat_models, transcription_models)
        );
    }

    #[test]
    fn test_dated_snapshots_with_their_own_shutdown_date_are_warned() {
        // The deprecations page gives these snapshots a shutdown date of
        // their own. Their model pages give no other price than for their
        // model.
        let gpt_4_0314 = chat_model_behaviour("gpt-4-0314", "gpt-4-0314");
        let gpt_4 = chat_model_behaviour("gpt-4", "gpt-4-0314");
        assert_eq!(
            (gpt_4_0314.1, &gpt_4_0314.2),
            (gpt_4.1, &Vec::<String>::new())
        );
        assert_eq!(gpt_4_0314.3.len(), 1, "{:?}", gpt_4_0314.3);
        for text in ["'gpt-4-0314'", "2026-03-26", "Use gpt-4.1 instead"] {
            assert!(gpt_4_0314.3[0].contains(text), "{:?}", gpt_4_0314.3);
        }

        let march = transcription_model_behaviour(
            "gpt-4o-mini-transcribe-2025-03-20",
            "gpt-4o-mini-transcribe-2025-03-20",
        );
        let model = transcription_model_behaviour(
            "gpt-4o-mini-transcribe",
            "gpt-4o-mini-transcribe-2025-03-20",
        );
        assert_eq!((march.1, &march.2), (model.1, &Vec::<String>::new()));
        assert_eq!(march.3.len(), 1, "{:?}", march.3);
        for text in [
            "'gpt-4o-mini-transcribe-2025-03-20'",
            "2027-01-20",
            "Use gpt-4o-mini-transcribe-2025-12-15 instead",
        ] {
            assert!(march.3[0].contains(text), "{:?}", march.3);
        }

        // The gpt-3.5-turbo page lists gpt-3.5-turbo-instruct as a snapshot.
        // The pricing page gives it a price of its own.
        let instruct = chat_model_behaviour("gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct");
        assert_eq!(instruct.2, Vec::<String>::new());
        assert_eq!(instruct.3.len(), 1, "{:?}", instruct.3);
        for text in [
            "'gpt-3.5-turbo-instruct'",
            "2026-09-28",
            "Use gpt-5.6-terra instead",
        ] {
            assert!(instruct.3[0].contains(text), "{:?}", instruct.3);
        }
    }

    #[test]
    fn test_unknown_model_uses_default_model_pricing() {
        use std::time::Duration;

        let defaults = Config::default().openai;
        let unknown = price_estimator("unknown-model-xyz", "unknown-transcriber");
        let default = price_estimator(&defaults.model, &defaults.transcription_model);
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

    #[tokio::test(start_paused = true)]
    async fn test_rate_limiter_waits_only_when_the_budget_is_used_up() {
        use std::time::Duration;
        use tokio::time::Instant;

        let mut limiter = RateLimiter::new(2);
        let mut waits = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            limiter.wait().await;
            waits.push(start.elapsed());
        }

        // The limiter measures the minute on the real clock, which moves a
        // little while the paused test clock does not.
        assert!(
            waits[..2] == [Duration::ZERO; 2] && waits[2] >= Duration::from_secs(59),
            "{:?}",
            waits
        );
    }

    // ===========================================================================
    // Test: Config load/save round-trip
    // ===========================================================================

    #[test]
    fn test_config_round_trip() {
        use std::fs;

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
                silence_duration: 2.5,
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

        // A file in the temp directory, named per process, so parallel runs
        // do not share it and nothing is written into the repo
        let temp_path = std::env::temp_dir().join(format!(
            "babble_boop_config_roundtrip_{}.toml",
            std::process::id()
        ));

        // Save config
        config.save(&temp_path).expect("Failed to save config");

        // Load it back
        let loaded = Config::load(&temp_path).expect("Failed to load config");

        // Clean up before asserting, so a failure does not leave the file behind
        fs::remove_file(&temp_path).unwrap();

        // Compare the whole struct, so every field is checked
        assert_eq!(loaded.config, config);
        assert_eq!(loaded.warnings, []);
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
        // Removed fields (passthrough_enabled, passthrough_port,
        // silence_threshold) should be ignored. `Config::from_toml` gives a
        // warning for silence_threshold.
        assert_eq!(config.audio.silence_duration, 1.0);
    }

    // ===========================================================================
    // Test: API error messages are truncated by characters, not bytes
    // ===========================================================================

    /// The text shown in the activity log for an OpenAI style JSON error
    /// with `message`.
    fn displayed_api_error(message: &str) -> String {
        use crate::api_client::ApiError;

        let body = serde_json::json!({ "error": { "message": message, "code": "other" } });
        ApiError::Http {
            status: reqwest::StatusCode::BAD_REQUEST,
            body: body.to_string(),
        }
        .to_string()
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
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }

        RecordingManager::new(dir.clone(), 10)
            .save_recording(vec![0u8; 4], transcription, &test_logger())
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
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }

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
        use crate::api_client::{ApiError, OpenAi};
        use std::time::Duration;
        use tokio::net::TcpListener;

        // Accept the connection and keep it open without sending a response.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (_socket, _) = listener.accept().await.unwrap();
            std::future::pending::<()>().await;
        });

        let api = OpenAi::with_timeouts(
            &format!("http://{}/v1", addr),
            Duration::from_secs(5),
            Duration::from_millis(200),
        )
        .unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            api.post_json("chat/completions", "sk-test", &serde_json::json!({})),
        )
        .await;
        server.abort();

        let error = result
            .expect("request was not stopped by the client timeout")
            .expect_err("server never responds");
        assert!(
            matches!(&error, ApiError::Transport(e) if e.is_timeout()),
            "expected a timeout error, got {:?}",
            error
        );
    }

    // ===========================================================================
    // Test: Chatbox display pause
    // ===========================================================================

    /// A `Chatbox` that sends to a local UDP socket, the config to send
    /// with (`display_time` in ms, up to 10 chunks), and that socket.
    async fn local_chatbox(
        display_time: u64,
    ) -> (crate::chatbox::Chatbox, Config, std::net::UdpSocket) {
        use crate::chatbox::Chatbox;
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::net::UdpSocket;

        let receiver = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = receiver.local_addr().unwrap().port();
        config.osc.display_time = display_time;
        config.osc.max_message_chunks = 10;
        (Chatbox::new(socket), config, receiver)
    }

    /// The texts of the chatbox messages that reached `receiver` since the
    /// last call. `receiver` is a std socket read with a timeout in real
    /// time, so reading it does not move the paused tokio clock.
    fn received_texts(receiver: &std::net::UdpSocket) -> Vec<String> {
        let mut texts = Vec::new();
        let mut buf = [0u8; 1024];
        while let Ok(len) = receiver.recv(&mut buf) {
            match rosc::decoder::decode_udp(&buf[..len]).unwrap().1 {
                rosc::OscPacket::Message(message) => match message.args.first() {
                    Some(rosc::OscType::String(text)) => texts.push(text.clone()),
                    _ => panic!("unexpected chatbox message {:?}", message),
                },
                bundle => panic!("unexpected OSC bundle {:?}", bundle),
            }
        }
        texts
    }

    #[tokio::test(start_paused = true)]
    async fn test_chatbox_pauses_between_chunks_but_not_after_the_last() {
        use std::time::Duration;
        use tokio::time::{sleep_until, Instant};

        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        let message = format!("{}{}{}", "a".repeat(144), "b".repeat(144), "c");

        let start = Instant::now();
        let (sent_after, arrivals) = tokio::join!(
            async {
                chatbox.send(&message, &config).await.unwrap();
                start.elapsed()
            },
            async {
                // Between the expected send times: 0 s, 10 s and 20 s
                let mut arrivals = Vec::new();
                for seconds in [5, 15, 25] {
                    sleep_until(start + Duration::from_secs(seconds)).await;
                    arrivals.push(received_texts(&receiver));
                }
                arrivals
            }
        );

        assert_eq!(
            arrivals,
            [["a".repeat(144)], ["b".repeat(144)], ["c".to_string()]]
        );
        assert_eq!(sent_after, Duration::from_secs(20));
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_waits_the_rest_of_the_display_time() {
        use std::time::Duration;
        use tokio::time::{sleep, Instant};

        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        chatbox.send("first", &config).await.unwrap();
        sleep(Duration::from_secs(4)).await;

        let start = Instant::now();
        chatbox.send("second", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::from_secs(6));
        assert_eq!(received_texts(&receiver), ["first", "second"]);
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_waits_from_the_last_chunk() {
        use std::time::Duration;
        use tokio::time::Instant;

        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        // Two chunks: the second goes out 10 s after the first.
        chatbox.send(&"a".repeat(145), &config).await.unwrap();

        let start = Instant::now();
        chatbox.send("next", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::from_secs(10));
        assert_eq!(received_texts(&receiver).last().unwrap(), "next");
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_after_the_display_time_does_not_wait() {
        use std::time::Duration;
        use tokio::time::{sleep, Instant};

        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        chatbox.send("first", &config).await.unwrap();
        sleep(Duration::from_secs(15)).await;

        let start = Instant::now();
        chatbox.send("second", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::ZERO);
        assert_eq!(received_texts(&receiver), ["first", "second"]);
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_waits_the_current_display_time() {
        use std::time::Duration;
        use tokio::time::Instant;

        let (mut chatbox, mut config, _receiver) = local_chatbox(10_000).await;
        chatbox.send("first", &config).await.unwrap();
        config.osc.display_time = 2_000;

        let start = Instant::now();
        chatbox.send("second", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::from_secs(2));
    }

    #[tokio::test(start_paused = true)]
    async fn test_response_without_translation_does_not_wait_for_the_chatbox() {
        use crate::app_state::LogLevel;
        use tokio::time::Instant;

        let mut fixture = DeliveryFixture::new("refusal_after_message").await;
        fixture.config.osc.display_time = 10_000;
        fixture.sender.send("first", &fixture.config).await.unwrap();
        let body = chat_completion_body_with(
            "null",
            r#""I can't help with that.""#,
            "stop",
            Some(SHORT_USAGE),
        );
        let translation = crate::translation::ChatGptRequest::translation(
            &fixture.config.openai.model,
            "French",
            "Hello",
        )
        .parse_response(&body)
        .unwrap();

        let start = Instant::now();
        let result = crate::audio_processing::deliver_translation(
            translation,
            "Hello",
            &fixture.config,
            &mut fixture.sender,
            &fixture.typing_indicator,
            &mut fixture.price_estimator,
            &fixture.app_state,
        )
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
        fixture.sender.send("first", &fixture.config).await.unwrap();
        fixture
    }

    /// Set the translation toggle to `enabled` after `delay`, as the GUI
    /// does: it stores the value at once, and the processing loop handles
    /// the `SetEnabled` command only after `process_audio` returns.
    fn toggle_after(
        app_state: &std::sync::Arc<crate::app_state::AppState>,
        delay: std::time::Duration,
        enabled: bool,
    ) {
        let app_state = std::sync::Arc::clone(app_state);
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            app_state
                .enabled
                .store(enabled, std::sync::atomic::Ordering::Relaxed);
        });
    }

    /// Give the translation "Bonjour" of "Hello" to `deliver_translation`
    /// in `fixture`.
    async fn deliver_bonjour(
        fixture: &mut DeliveryFixture,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let body = chat_completion_body("Bonjour", Some(SHORT_USAGE));
        let translation = crate::translation::ChatGptRequest::translation(
            &fixture.config.openai.model,
            "French",
            "Hello",
        )
        .parse_response(&body)
        .unwrap();
        crate::audio_processing::deliver_translation(
            translation,
            "Hello",
            &fixture.config,
            &mut fixture.sender,
            &fixture.typing_indicator,
            &mut fixture.price_estimator,
            &fixture.app_state,
        )
        .await
    }

    /// The log lines of a translation that was not sent because
    /// translation was switched off.
    fn not_sent_log() -> Vec<(crate::app_state::LogLevel, String)> {
        use crate::app_state::LogLevel;
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
        use std::time::Duration;

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
        fixture
            .app_state
            .enabled
            .store(false, std::sync::atomic::Ordering::Relaxed);

        let result = deliver_bonjour(&mut fixture).await;

        let delivery = fixture.finish(result).await;
        assert_eq!(delivery.chatbox, [typing_off()], "{:?}", delivery);
        assert_eq!(delivery.log, not_sent_log());
        assert_eq!(delivery.total_cost, short_usage_cost());
    }

    #[tokio::test(start_paused = true)]
    async fn test_translation_switched_off_and_on_during_the_display_wait_is_sent() {
        use crate::app_state::LogLevel;
        use std::time::Duration;
        use tokio::time::Instant;

        let mut fixture = fixture_showing_first("off_and_on_during_display_wait").await;
        toggle_after(&fixture.app_state, Duration::from_secs(1), false);
        toggle_after(&fixture.app_state, Duration::from_secs(2), true);

        let start = Instant::now();
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
    // Test: Shutdown interrupts in flight work
    // ===========================================================================

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_interrupts_chatbox_display_pause() {
        use crate::shutdown::Shutdown;
        use std::time::Duration;
        use tokio::time::{sleep, Instant};

        let (mut chatbox, config, _receiver) = local_chatbox(30_000).await;
        // Three chunks, so the chatbox pauses 60 s in total.
        let message = "a".repeat(300);

        let shutdown = Shutdown::new();
        let requester = shutdown.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(1)).await;
            requester.request();
        });

        let start = Instant::now();
        let result = shutdown.run_until(chatbox.send(&message, &config)).await;

        assert!(result.is_none(), "shutdown did not stop the chatbox send");
        assert_eq!(start.elapsed(), Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_interrupts_wait_before_next_message() {
        use crate::shutdown::Shutdown;
        use std::time::Duration;
        use tokio::time::{sleep, Instant};

        let (mut chatbox, config, receiver) = local_chatbox(30_000).await;
        chatbox.send("first", &config).await.unwrap();
        assert_eq!(received_texts(&receiver), ["first"]);

        let shutdown = Shutdown::new();
        let requester = shutdown.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(1)).await;
            requester.request();
        });

        let start = Instant::now();
        let result = shutdown.run_until(chatbox.send("second", &config)).await;

        assert!(result.is_none(), "shutdown did not stop the wait");
        assert_eq!(start.elapsed(), Duration::from_secs(1));
        assert_eq!(received_texts(&receiver), Vec::<String>::new());
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
        chat_completion_body_with(
            &serde_json::to_string(content).unwrap(),
            "null",
            "stop",
            usage,
        )
    }

    /// Chat Completions response body with one choice. `content` and
    /// `refusal` are the JSON of the message fields, such as `null`.
    fn chat_completion_body_with(
        content: &str,
        refusal: &str,
        finish_reason: &str,
        usage: Option<&str>,
    ) -> String {
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
                        "refusal": {},
                        "annotations": []
                    }},
                    "logprobs": null,
                    "finish_reason": "{}"
                }}],
                "service_tier": "default",
                "system_fingerprint": null{}
            }}"#,
            content, refusal, finish_reason, usage
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

        assert_eq!(translation.text, Ok("Quelle heure est-il ?".to_string()));
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

            assert_eq!(translation.text, Ok("Quelle heure est-il ?".to_string()));
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

        let estimator = price_estimator("gpt-5.6-sol", "gpt-transcribe");
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

    // ===========================================================================
    // Test: A response without a translation is logged, not sent
    // ===========================================================================

    /// Usage of a short response, such as a refusal.
    const SHORT_USAGE: &str = r#"{
        "prompt_tokens": 58,
        "completion_tokens": 12,
        "total_tokens": 70
    }"#;

    /// What the processing loop did with a response from the API.
    #[derive(Debug)]
    struct Delivery {
        /// Level and text of the activity log entries.
        log: Vec<(crate::app_state::LogLevel, String)>,
        /// The OSC messages that reached the chatbox address.
        chatbox: Vec<rosc::OscMessage>,
        /// The total cost after the response, from zero.
        total_cost: f64,
    }

    /// A chatbox on a local UDP socket, the activity log, and a total cost
    /// that starts at zero, in a file of its own named after the test.
    struct DeliveryFixture {
        config: Config,
        chatbox: tokio::net::UdpSocket,
        sender: crate::chatbox::Chatbox,
        app_state: std::sync::Arc<crate::app_state::AppState>,
        log_rx: tokio::sync::mpsc::Receiver<crate::app_state::LogEntry>,
        typing_indicator: crate::typing_indicator::TypingIndicator,
        price_estimator: PriceEstimator,
        cost_file: std::path::PathBuf,
    }

    impl DeliveryFixture {
        async fn new(test_name: &str) -> Self {
            use crate::app_state::AppState;
            use crate::chatbox::Chatbox;
            use crate::typing_indicator::TypingIndicator;
            use std::sync::Arc;
            use tokio::net::UdpSocket;

            let chatbox = UdpSocket::bind("127.0.0.1:0").await.unwrap();
            let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
            let mut config = Config::default();
            config.osc.address = "127.0.0.1".to_string();
            config.osc.output_port = chatbox.local_addr().unwrap().port();
            config.osc.display_time = 0;
            let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
            let (log_tx, log_rx) = tokio::sync::mpsc::channel(10);
            let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
            let typing_indicator = TypingIndicator::new(socket.clone(), app_state.logger.clone());
            let cost_file = std::env::temp_dir().join(format!(
                "babble_boop_{}_{}_total_cost.txt",
                test_name,
                std::process::id()
            ));
            if cost_file.exists() {
                std::fs::remove_file(&cost_file).unwrap();
            }
            let price_estimator = PriceEstimator::new(
                cost_file.clone(),
                &config.openai.model,
                &config.openai.transcription_model,
            );
            DeliveryFixture {
                config,
                chatbox,
                sender: Chatbox::new(socket),
                app_state,
                log_rx,
                typing_indicator,
                price_estimator,
                cost_file,
            }
        }

        /// Log an error as the processing loop logs an error of
        /// `process_audio`, then collect what reached the activity log and
        /// the chatbox.
        async fn finish(mut self, result: Result<(), Box<dyn std::error::Error>>) -> Delivery {
            use std::time::Duration;

            if let Err(e) = result {
                crate::processing_loop::log_processing_error(&*e, &self.app_state.logger);
            }
            if self.cost_file.exists() {
                std::fs::remove_file(&self.cost_file).unwrap();
            }
            assert_eq!(
                self.app_state.get_total_cost(),
                self.price_estimator.total_cost
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
                total_cost: self.price_estimator.total_cost,
            }
        }
    }

    /// Give the Chat Completions response `body` to `parse_response` and
    /// `deliver_translation` in a `DeliveryFixture`.
    async fn deliver_response(test_name: &str, body: &str) -> Delivery {
        use crate::audio_processing::deliver_translation;
        use crate::translation::ChatGptRequest;

        let mut fixture = DeliveryFixture::new(test_name).await;
        let request = ChatGptRequest::translation(&fixture.config.openai.model, "French", "Hello");

        let result = match request.parse_response(body) {
            Ok(translation) => {
                deliver_translation(
                    translation,
                    "Hello",
                    &fixture.config,
                    &mut fixture.sender,
                    &fixture.typing_indicator,
                    &mut fixture.price_estimator,
                    &fixture.app_state,
                )
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
        use crate::app_state::LogLevel;

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

    #[test]
    fn test_refusal_tokens_are_estimated_from_the_refusal_without_usage() {
        use crate::translation::{ChatGptRequest, NoTranslation};

        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "Hello");
        let refusal = "I'm sorry, but I can't help with that.";
        let body = chat_completion_body_with(
            "null",
            &serde_json::to_string(refusal).unwrap(),
            "stop",
            None,
        );

        let translation = request.parse_response(&body).unwrap();

        assert_eq!(
            translation.text,
            Err(NoTranslation::Refused(refusal.to_string()))
        );
        // Four bytes a token: 38 bytes of refusal give 9 tokens.
        assert_eq!(translation.tokens.output, 9);
    }

    #[test]
    fn test_blank_refusal_is_not_a_refusal() {
        use crate::translation::ChatGptRequest;

        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "Hello");
        let body = chat_completion_body_with(r#""Bonjour""#, r#""""#, "stop", None);

        let translation = request.parse_response(&body).unwrap();

        assert_eq!(translation.text, Ok("Bonjour".to_string()));
    }

    #[tokio::test]
    async fn test_translation_is_sent_to_the_chatbox_and_costed() {
        use crate::app_state::LogLevel;

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
        use crate::audio_processing::accept_transcription;
        use crate::transcription::parse_transcription;
        use std::time::Duration;

        let mut fixture = DeliveryFixture::new(test_name).await;
        let (text, result) = match parse_transcription(body) {
            Ok(text) => (
                accept_transcription(
                    text,
                    Duration::from_secs(1),
                    &fixture.config,
                    &fixture.typing_indicator,
                    &mut fixture.price_estimator,
                    &fixture.app_state,
                )
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
            .estimate_transcription_cost(std::time::Duration::from_secs(1))
    }

    #[tokio::test]
    async fn test_transcription_is_costed_and_translated() {
        use crate::app_state::LogLevel;

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
        use crate::app_state::LogLevel;

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

    /// A body with a null text is an error of `process_audio`, and there
    /// is no transcription to cost.
    #[tokio::test]
    async fn test_transcription_body_with_null_text_is_an_error() {
        use crate::app_state::LogLevel;

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
        let mut services = ProcessingServices::new(&config, &missing_data_dir(), &test_logger());
        services.price_estimator.total_cost = 1.25;

        config.openai.model = "gpt-4o".to_string();
        config.openai.transcription_model = "gpt-4o-mini-transcribe".to_string();
        services.apply_config(&config, &test_logger());

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
        use crate::processing_loop::ProcessingServices;
        use std::fs;

        let dir =
            std::env::temp_dir().join(format!("babble_boop_data_folder_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("total_cost.txt"), "1.5").unwrap();
        let data_dir = DataDir::new(dir.clone());
        let config = Config {
            keep_audio_files: true,
            ..Config::default()
        };
        let logger = test_logger();

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
        let saved_cost = fs::read_to_string(dir.join("total_cost.txt")).ok();
        let mut recordings: Vec<String> = fs::read_dir(dir.join("recordings"))
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        recordings.sort();
        // Clean up before asserting, so a failure does not leave files behind
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(loaded_cost, 1.5);
        assert_eq!(saved_cost.as_deref(), Some("2"));
        assert_eq!(recordings.len(), 2, "{:?}", recordings);
        assert!(recordings[0].ends_with("_first.wav"), "{:?}", recordings);
        assert!(recordings[1].ends_with("_second.wav"), "{:?}", recordings);
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
            ProcessingServices::new(&config, &missing_data_dir(), logger);
        });
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert!(entries[0].message.contains("'my-finetuned-model'"));

        let mut services =
            ProcessingServices::new(&Config::default(), &missing_data_dir(), &test_logger());
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
            ProcessingServices::new(&config, &missing_data_dir(), logger);
        });
        assert_eq!(entries.len(), 1, "{:?}", entries);
        for text in ["'gpt-3.5-turbo'", "2026-10-23", "gpt-5.6-terra"] {
            assert!(entries[0].message.contains(text), "{:?}", entries[0]);
        }

        let mut services =
            ProcessingServices::new(&Config::default(), &missing_data_dir(), &test_logger());
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
                    ProcessingServices::new(&config, &missing_data_dir(), logger);
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
            ProcessingServices::new(&config, &missing_data_dir(), logger)
                .apply_config(&config, logger);
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
        let mut services = ProcessingServices::new(&config, &missing_data_dir(), &test_logger());
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
        use crate::types::{AudioEvent, CapturedAudio, Extent};

        let audio = || CapturedAudio {
            samples: vec![0.5; 4],
            channels: 1,
            sample_rate: 16_000,
        };
        let entries = entries_logged_by(|logger| {
            for event in [
                AudioEvent::StartRecording,
                AudioEvent::AudioPart(audio()),
                AudioEvent::AudioData(audio(), Extent::Whole),
                AudioEvent::StopRecording,
                AudioEvent::RecordingDiscarded,
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
                    "Test Microphone started, the recording in progress is discarded",
                    LogLevel::Info
                ),
                (
                    "Lost 3 audio events because the processing queue was full",
                    LogLevel::Error
                ),
                ("Audio input error: device unplugged", LogLevel::Error),
            ]
        );
    }

    /// The events that `events` returns before it waits 60 s for the next
    /// one.
    async fn events_until_quiet(
        events: &mut crate::processing_loop::AudioEvents,
    ) -> Vec<crate::types::AudioEvent> {
        use std::time::Duration;

        let mut received = Vec::new();
        while let Ok(event) = tokio::time::timeout(Duration::from_secs(60), events.recv()).await {
            received.push(event);
        }
        received
    }

    /// cpal reports playback stream errors on an audio thread, which does
    /// not log. They reach the activity log through the processing loop,
    /// and an error that cpal repeats is logged once.
    #[tokio::test(start_paused = true)]
    async fn test_playback_errors_reach_the_activity_log_once_per_distinct_error() {
        use crate::app_state::{LogLevel, Logger};
        use crate::audio_playback::playback_error_reporter;
        use crate::processing_loop::PlaybackErrors;
        use std::time::Duration;

        /// Log the errors that came, and return the new activity log entries.
        async fn logged(
            errors: &mut PlaybackErrors,
            logger: &Logger,
            log_rx: &mut tokio::sync::mpsc::Receiver<crate::app_state::LogEntry>,
        ) -> Vec<(String, LogLevel)> {
            while tokio::time::timeout(Duration::from_secs(60), errors.log_next(logger))
                .await
                .is_ok()
            {}
            std::iter::from_fn(|| log_rx.try_recv().ok())
                .map(|entry| (entry.message, entry.level))
                .collect()
        }
        let (log_tx, mut log_rx) = tokio::sync::mpsc::channel(10);
        let logger = Logger::new(log_tx, Default::default());
        let mut errors = PlaybackErrors::default();

        // cpal calls the error callback in a loop while the device is gone
        let mut first_stream = playback_error_reporter(errors.sender());
        for error in [
            "device unplugged",
            "device unplugged",
            "underrun",
            "device unplugged",
        ] {
            first_stream.report(error);
        }
        assert_eq!(
            logged(&mut errors, &logger, &mut log_rx).await,
            ["device unplugged", "underrun", "device unplugged"]
                .map(|e| (format!("Playback error: {}", e), LogLevel::Error))
        );

        // The next playback has a new stream, which reports its first error
        let mut second_stream = playback_error_reporter(errors.sender());
        second_stream.report("device unplugged");
        second_stream.report("device unplugged");
        assert_eq!(
            logged(&mut errors, &logger, &mut log_rx).await,
            [(
                "Playback error: device unplugged".to_string(),
                LogLevel::Error
            )]
        );
    }

    /// When the audio input ends in a recording, no StopRecording comes from
    /// the callback. The processing loop turns the typing indicator off only
    /// on StopRecording, so it would stay on in VRChat.
    #[tokio::test(start_paused = true)]
    async fn test_the_end_of_audio_input_ends_the_recording_and_is_logged_once() {
        use crate::app_state::{LogLevel, Logger};
        use crate::processing_loop::AudioEvents;
        use crate::types::AudioEvent;

        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, mut log_rx) = tokio::sync::mpsc::channel(10);
        let mut events = AudioEvents::new(rx, Logger::new(log_tx, Default::default()));
        tx.try_send(AudioEvent::StartRecording).unwrap();
        // The audio thread ends and drops every sender
        drop(tx);

        assert_eq!(
            events_until_quiet(&mut events).await,
            vec![AudioEvent::StartRecording, AudioEvent::StopRecording]
        );
        // After that, no event comes; the loop waits for its other branches
        assert_eq!(events_until_quiet(&mut events).await, vec![]);
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

    /// A crash of the callback, or a stream error such as an unplugged
    /// device, can end the input while the channel stays open. The recording
    /// then ends with no StopRecording from the callback.
    #[tokio::test(start_paused = true)]
    async fn test_an_audio_input_error_ends_the_recording() {
        use crate::processing_loop::AudioEvents;
        use crate::types::AudioEvent;

        for message in [
            "Audio input crashed: index out of bounds. Restart BabbleBoop to record again.",
            "Audio input error: device unplugged",
        ] {
            let (tx, rx) = tokio::sync::mpsc::channel(10);
            let mut events = AudioEvents::new(rx, test_logger());
            let error = || AudioEvent::InputError(message.to_string());
            tx.try_send(AudioEvent::StartRecording).unwrap();
            tx.try_send(error()).unwrap();

            assert_eq!(
                events_until_quiet(&mut events).await,
                vec![
                    AudioEvent::StartRecording,
                    error(),
                    AudioEvent::StopRecording
                ],
                "{}",
                message
            );
            // The recording ends once, and later events come as sent
            tx.try_send(AudioEvent::StartRecording).unwrap();
            assert_eq!(
                events_until_quiet(&mut events).await,
                vec![AudioEvent::StartRecording],
                "{}",
                message
            );
        }
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

    // ===========================================================================
    // Test: Any minimum transcription duration in config.toml is safe
    // ===========================================================================

    /// What `process_audio` did with a recording.
    #[derive(Debug, PartialEq)]
    pub(crate) enum MinimumCheck {
        /// The recording was skipped as too short.
        Skipped,
        /// The recording went on to transcription.
        Transcribed,
    }

    /// Run `process_audio` on one second of audio with `min_seconds` as the
    /// minimum transcription duration.
    async fn check_one_second_against_minimum(min_seconds: f32) -> MinimumCheck {
        let second = crate::types::CapturedAudio {
            samples: vec![0.25; 16_000],
            channels: 1,
            sample_rate: 16_000,
        };
        check_against_minimum(second, crate::types::Extent::Whole, min_seconds).await
    }

    /// Run `process_audio` on `audio`, which holds `extent` of a recording,
    /// with `min_seconds` as the minimum transcription duration. The API is
    /// a local server that answers every request with 404.
    pub(crate) async fn check_against_minimum(
        audio: crate::types::CapturedAudio,
        extent: crate::types::Extent,
        min_seconds: f32,
    ) -> MinimumCheck {
        use crate::api_client::test_server::TestServer;
        use crate::api_client::OpenAi;
        use crate::app_state::AppState;
        use crate::audio_processing::process_audio;
        use crate::chatbox::Chatbox;
        use crate::processing_loop::encode_for_upload;
        use crate::typing_indicator::TypingIndicator;
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::net::UdpSocket;

        let mut server = TestServer::start(Vec::new()).await;
        let api = OpenAi::new(&server.base_url).unwrap();

        let chatbox = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = chatbox.local_addr().unwrap().port();
        config.audio.min_transcription_duration = min_seconds;
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let typing_indicator = TypingIndicator::new(socket.clone(), app_state.logger.clone());
        let wav = encode_for_upload(audio).await.unwrap();

        let result = tokio::time::timeout(
            Duration::from_secs(10),
            process_audio(
                &api,
                wav,
                extent,
                &config,
                &mut Chatbox::new(socket),
                &mut RateLimiter::new(50),
                &typing_indicator,
                &mut price_estimator(&config.openai.model, &config.openai.transcription_model),
                None,
                &app_state,
            ),
        )
        .await
        .expect("process_audio did not finish");

        if !server.received().is_empty() {
            assert!(result.is_err(), "the server answered 404");
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

    #[tokio::test]
    async fn test_minimum_duration_does_not_apply_to_a_part_of_a_long_recording() {
        // A part holds 30 s unless it is the last part; a config.toml
        // minimum can be longer than that.
        let second = crate::types::CapturedAudio {
            samples: vec![0.25; 16_000],
            channels: 1,
            sample_rate: 16_000,
        };
        assert_eq!(
            check_against_minimum(second, crate::types::Extent::Part, 2.0).await,
            MinimumCheck::Transcribed
        );
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
        .error_with_details("API error", "HTTP 500")));
    }

    #[test]
    fn test_cost_update_wakes_gui() {
        use crate::app_state::AppState;
        use eframe::egui;

        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        let app_state = AppState::new(cmd_tx, log_tx);
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
        let app_state = crate::app_state::AppState::new(cmd_tx, log_tx);
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
    // Test: Waiting for the audio input to start cannot hang shutdown
    // ===========================================================================

    #[tokio::test(start_paused = true)]
    async fn test_audio_start_returns_the_stream_format() {
        use crate::processing_loop::{wait_for_audio_start, AUDIO_START_TIMEOUT};
        use crate::shutdown::Shutdown;
        use tokio::sync::oneshot;

        let (started_tx, started_rx) = oneshot::channel::<Result<u32, String>>();
        started_tx.send(Ok(48_000)).unwrap();
        let result = wait_for_audio_start(started_rx, &Shutdown::new(), AUDIO_START_TIMEOUT).await;
        assert_eq!(result, Ok(Some(48_000)));
    }

    #[tokio::test(start_paused = true)]
    async fn test_audio_start_error_is_a_startup_error() {
        use crate::processing_loop::{wait_for_audio_start, AUDIO_START_TIMEOUT};
        use crate::shutdown::Shutdown;
        use tokio::sync::oneshot;

        let (started_tx, started_rx) = oneshot::channel::<Result<u32, String>>();
        started_tx.send(Err("no input device".to_string())).unwrap();
        let result = wait_for_audio_start(started_rx, &Shutdown::new(), AUDIO_START_TIMEOUT).await;
        assert_eq!(
            result,
            Err("cannot start audio input: no input device".to_string())
        );
    }

    #[tokio::test(start_paused = true)]
    async fn test_audio_start_times_out_when_the_stream_does_not_start() {
        use crate::processing_loop::{wait_for_audio_start, AUDIO_START_TIMEOUT};
        use crate::shutdown::Shutdown;
        use tokio::sync::oneshot;
        use tokio::time::{timeout, Instant};

        // The audio thread is stuck in the driver: the sender stays alive
        let (_started_tx, started_rx) = oneshot::channel::<Result<u32, String>>();
        let started = Instant::now();
        let result = timeout(
            AUDIO_START_TIMEOUT * 2,
            wait_for_audio_start(started_rx, &Shutdown::new(), AUDIO_START_TIMEOUT),
        )
        .await
        .expect("waiting for the audio input has no time limit");
        assert_eq!(
            result,
            Err("cannot start audio input: timed out after 15 s".to_string())
        );
        assert_eq!(started.elapsed(), AUDIO_START_TIMEOUT);
    }

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_stops_waiting_for_the_audio_start() {
        use crate::processing_loop::{wait_for_audio_start, AUDIO_START_TIMEOUT};
        use crate::shutdown::Shutdown;
        use std::time::Duration;
        use tokio::sync::oneshot;
        use tokio::time::{timeout, Instant};

        let (_started_tx, started_rx) = oneshot::channel::<Result<u32, String>>();
        let shutdown = Shutdown::new();
        let requester = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            requester.request();
        });
        let started = Instant::now();
        let result = timeout(
            AUDIO_START_TIMEOUT * 2,
            wait_for_audio_start(started_rx, &shutdown, AUDIO_START_TIMEOUT),
        )
        .await
        .expect("shutdown does not stop the wait");
        assert_eq!(result, Ok(None));
        assert_eq!(started.elapsed(), Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn test_audio_start_fails_when_the_audio_thread_ends_without_a_result() {
        use crate::processing_loop::{wait_for_audio_start, AUDIO_START_TIMEOUT};
        use crate::shutdown::Shutdown;
        use tokio::sync::oneshot;
        use tokio::time::Instant;

        // As when start_audio_recording panics
        let (started_tx, started_rx) = oneshot::channel::<Result<u32, String>>();
        drop(started_tx);
        let started = Instant::now();
        let result = wait_for_audio_start(started_rx, &Shutdown::new(), AUDIO_START_TIMEOUT).await;
        assert_eq!(
            result,
            Err("cannot start audio input: the audio thread stopped".to_string())
        );
        assert_eq!(started.elapsed(), std::time::Duration::ZERO);
    }

    /// Run `hold_audio_stream` on a thread. The receiver gets a message
    /// when it returns, and the returned `Arc` is the stream: its strong
    /// count falls to 1 when the stream is dropped.
    fn hold_on_thread(
        app_state: std::sync::Arc<crate::app_state::AppState>,
    ) -> (std::sync::mpsc::Receiver<()>, std::sync::Arc<()>) {
        use crate::processing_loop::hold_audio_stream;

        let stream = std::sync::Arc::new(());
        let held = std::sync::Arc::clone(&stream);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            hold_audio_stream(held, &app_state);
            done_tx.send(()).unwrap();
        });
        (done_rx, stream)
    }

    #[test]
    fn test_audio_stream_is_held_while_the_processing_loop_runs() {
        use std::sync::Arc;
        use std::time::Duration;

        let (app_state, _cmd_rx, _log_rx) = app_state_for_test();
        let (done, stream) = hold_on_thread(Arc::new(app_state));
        assert!(done.recv_timeout(Duration::from_millis(300)).is_err());
        assert_eq!(Arc::strong_count(&stream), 2);
    }

    #[test]
    fn test_audio_stream_is_dropped_on_shutdown() {
        use std::sync::Arc;
        use std::time::Duration;

        let (app_state, _cmd_rx, _log_rx) = app_state_for_test();
        let app_state = Arc::new(app_state);
        let (done, stream) = hold_on_thread(Arc::clone(&app_state));
        app_state.request_shutdown();
        done.recv_timeout(Duration::from_secs(5))
            .expect("the stream is still held after shutdown");
        assert_eq!(Arc::strong_count(&stream), 1);
    }

    #[test]
    fn test_audio_stream_is_dropped_when_the_processing_thread_ends() {
        use std::sync::Arc;
        use std::time::Duration;

        // As after the loop timed out waiting for the stream, or stopped
        let (app_state, _cmd_rx, _log_rx) = app_state_for_test();
        let app_state = Arc::new(app_state);
        let (done, stream) = hold_on_thread(Arc::clone(&app_state));
        app_state.mark_processing_stopped();
        done.recv_timeout(Duration::from_secs(5))
            .expect("the stream is still held with nothing to receive its events");
        assert_eq!(Arc::strong_count(&stream), 1);
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
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::net::UdpSocket;

        let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = receiver.local_addr().unwrap().port();
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        let logger = Logger::new(log_tx, Default::default());
        let indicator = TypingIndicator::new(socket, logger.clone());

        apply_enabled(false, &config, &indicator, &logger).await;

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
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// A config file in a folder that does not exist, so a click on Save
    /// fails instead of writing a file.
    fn unused_config_file() -> PathBuf {
        std::env::temp_dir()
            .join(format!("babble_boop_no_config_dir_{}", std::process::id()))
            .join("config.toml")
    }

    fn test_app_with_state() -> (BabbleBoopApp, Arc<AppState>) {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        (
            BabbleBoopApp::new(
                Arc::clone(&app_state),
                Config::default(),
                log_rx,
                unused_config_file(),
            ),
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
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        (
            BabbleBoopApp::new(Arc::clone(&app_state), config, log_rx, unused_config_file()),
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
            assert_eq!(level_track.len(), 1, "{:?}", theme);
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

    // ===========================================================================
    // Test: The duration line shows when a recording can be sent
    // ===========================================================================

    /// One frame of the audio settings while a recording that is 0.5 s long
    /// is in progress, with a minimum transcription duration of 1 s.
    fn recording_frame(recording_split: bool) -> egui::FullOutput {
        use std::sync::atomic::Ordering;

        let (mut app, app_state) = test_app_with_state();
        app.config_draft.audio.min_transcription_duration = 1.0;
        app_state.is_recording.store(true, Ordering::Relaxed);
        app_state
            .recording_duration
            .store(0.5_f32.to_bits(), Ordering::Relaxed);
        app_state
            .recording_split
            .store(recording_split, Ordering::Relaxed);
        egui::Context::default().run(raw_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        })
    }

    /// The painted "Duration:" status in the frame from `recording_frame`.
    fn duration_status(recording_split: bool) -> Vec<String> {
        painted_text(&recording_frame(recording_split))
            .into_iter()
            .filter(|text| text.starts_with("0.5s"))
            .collect()
    }

    /// Widths of the filled parts of the 10 point high meters in the frame
    /// from `recording_frame`. The noise gate is closed and there is no
    /// quiet time, so only the duration meter has a filled part.
    fn duration_bar_fill_widths(recording_split: bool) -> Vec<f32> {
        small_meter_fill_widths(&recording_frame(recording_split))
    }

    /// Widths of the filled parts of the 10 point high meters in `output`.
    fn small_meter_fill_widths(output: &egui::FullOutput) -> Vec<f32> {
        let track = crate::theme::get_colors(Config::default().theme).meter_background;
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect) if rect.rect.height() == 10.0 && rect.fill != track => {
                    Some(rect.rect.width())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_last_part_of_a_split_recording_shows_ready_below_the_minimum() {
        // The minimum applies only to a whole recording, not to the last
        // part of a recording that reached the length limit.
        assert_eq!(duration_status(true), ["0.5s (ready)"]);
    }

    #[test]
    fn test_whole_recording_below_the_minimum_shows_not_ready() {
        assert_eq!(duration_status(false), ["0.5s / 1.0s"]);
    }

    #[test]
    fn test_last_part_of_a_split_recording_fills_the_duration_bar() {
        // The 200 point bar is full, because the minimum does not apply.
        assert_eq!(duration_bar_fill_widths(true), [200.0]);
    }

    #[test]
    fn test_whole_recording_below_the_minimum_half_fills_the_duration_bar() {
        // 0.5 s of the 1 s minimum fills half of the 200 point bar.
        assert_eq!(duration_bar_fill_widths(false), [100.0]);
    }

    // ===========================================================================
    // Test: The silence line shows seconds
    // ===========================================================================

    /// One frame of the audio settings while a recording is in progress,
    /// 0.5 s after the noise gate closed, with a silence duration of 2 s.
    fn silence_frame() -> egui::FullOutput {
        use std::sync::atomic::Ordering;

        let (mut app, app_state) = test_app_with_state();
        app.config_draft.audio.silence_duration = 2.0;
        app_state.is_recording.store(true, Ordering::Relaxed);
        app_state
            .quiet_time
            .store(0.5_f32.to_bits(), Ordering::Relaxed);
        egui::Context::default().run(raw_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        })
    }

    #[test]
    fn test_silence_line_shows_the_seconds_of_silence_and_the_duration() {
        let text = painted_text(&silence_frame());
        assert!(text.contains(&"0.5s / 2.0s".to_string()), "{:?}", text);
    }

    #[test]
    fn test_silence_bar_fills_by_seconds() {
        // 0.5 s of the 2 s duration fills a quarter of the 200 point bar.
        // The gate is closed and the duration is 0, so only the silence
        // meter has a filled part.
        assert_eq!(small_meter_fill_widths(&silence_frame()), [50.0]);
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

    /// Run one frame with `events` for what it does to `app`.
    fn run_with_events(ctx: &egui::Context, app: &mut BabbleBoopApp, events: Vec<egui::Event>) {
        #[expect(
            clippy::let_underscore_must_use,
            reason = "the tests that call this do not check the painted output"
        )]
        let _ = ctx.run(with_events(events), |ctx| app.ui(ctx));
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
        run_with_events(&ctx, &mut app, vec![select_all, typed]);

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
        run_with_events(&ctx, &mut app, vec![select_all, typed]);

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

    #[test]
    fn test_blank_osc_address_is_not_saved() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = "test-key".to_string();
        app.config_draft.osc.address = " \n".to_string();

        let error = app.config_to_save().unwrap_err();
        assert!(error.contains("address"), "{:?}", error);
    }

    /// The settings window clamps a value to the range of its field when it
    /// draws the field, even if the user does not touch it. A config that
    /// loads unchanged must also stay unchanged in the window.
    #[test]
    fn test_high_limits_are_not_clamped_in_the_settings() {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let config = Config {
            keep_audio_files: true,
            max_audio_files: 500,
            rate_limit: crate::config::RateLimitConfig {
                requests_per_minute: 500,
            },
            ..Config::default()
        };
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, config.clone(), log_rx, unused_config_file());
        // Tall enough that the settings below the log are not scrolled away
        let tall = |events| egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 3000.0),
            )),
            events,
            ..Default::default()
        };
        let ctx = egui::Context::default();
        let mut output = ctx.run(tall(Vec::new()), |ctx| app.ui(ctx));

        // Open the two sections, which are closed at the start
        for header in ["Rate Limit", "Debug"] {
            let pos = text_center(&output, header);
            let button = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            let press = vec![egui::Event::PointerMoved(pos), button(true)];
            #[expect(
                clippy::let_underscore_must_use,
                reason = "only the output after the release is used"
            )]
            let _ = ctx.run(tall(press), |ctx| app.ui(ctx));
            output = ctx.run(tall(vec![button(false)]), |ctx| app.ui(ctx));
        }
        for _ in 0..3 {
            output = ctx.run(tall(Vec::new()), |ctx| app.ui(ctx));
        }

        // Both fields were drawn, so the clamp had its chance to run
        let text = painted_text(&output);
        assert!(text.contains(&"Max Audio Files:".to_string()), "{:?}", text);
        assert!(
            text.contains(&"Requests per Minute:".to_string()),
            "{:?}",
            text
        );
        assert_eq!(app.config_draft, config);
    }

    /// Press and release the primary button at `pos`, in two frames.
    fn click(ctx: &egui::Context, app: &mut BabbleBoopApp, pos: egui::Pos2) {
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        run_with_events(ctx, app, vec![egui::Event::PointerMoved(pos), button(true)]);
        run_with_events(ctx, app, vec![button(false)]);
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

    // ===========================================================================
    // Test: A value that the loader replaced can be saved to the file
    // ===========================================================================

    const UNSAVED: &str = "● Unsaved changes";

    /// The app after a start with the default config file, but with
    /// `osc.display_time` set to `display_time` in the file. The API key is
    /// blank, so a click on Save fails the validation and writes no file.
    fn app_from_file_with_display_time(display_time: i64) -> BabbleBoopApp {
        let mut file = toml::Value::try_from(Config::default()).unwrap();
        file["osc"]["display_time"] = toml::Value::Integer(display_time);
        let loaded = Config::from_toml(&toml::to_string(&file).unwrap()).unwrap();
        assert_eq!(
            loaded.config.openai.api_key, "",
            "a click on Save must fail the validation"
        );
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, loaded.config, log_rx, unused_config_file());
        app.note_replaced_values(&loaded.warnings);
        app
    }

    /// 999999 is above the maximum, so the loader uses 30000.
    fn app_with_replaced_display_time() -> BabbleBoopApp {
        app_from_file_with_display_time(999_999)
    }

    #[test]
    fn test_replaced_config_value_shows_unsaved_changes() {
        let mut app = app_with_replaced_display_time();

        let text = painted_text(&run_until_idle(&egui::Context::default(), &mut app));

        assert!(text.contains(&UNSAVED.to_string()), "{:?}", text);
        // Reset has nothing to discard, as the draft has the values in use
        assert!(!text.contains(&"Reset".to_string()), "{:?}", text);
    }

    #[test]
    fn test_replaced_config_value_can_be_saved() {
        let mut app = app_with_replaced_display_time();
        let ctx = egui::Context::default();
        let output = run_until_idle(&ctx, &mut app);

        click(&ctx, &mut app, text_center(&output, "Save Settings"));

        // The click reached the save handler, which refused the blank key
        let text = painted_text(&run_until_idle(&ctx, &mut app));
        assert!(
            text.contains(&"OpenAI API key is required".to_string()),
            "{:?}",
            text
        );
    }

    #[test]
    fn test_reset_keeps_the_replaced_value_and_the_unsaved_changes() {
        let mut app = app_with_replaced_display_time();
        let ctx = egui::Context::default();
        app.config_draft.osc.display_time = 5000;
        let output = run_until_idle(&ctx, &mut app);

        click(&ctx, &mut app, text_center(&output, "Reset"));

        let text = painted_text(&run_until_idle(&ctx, &mut app));
        assert!(
            text.contains(&"Changes discarded".to_string()),
            "{:?}",
            text
        );
        assert_eq!(app.config_draft.osc.display_time, 30000);
        assert!(text.contains(&UNSAVED.to_string()), "{:?}", text);
    }

    #[test]
    fn test_config_file_without_replaced_values_has_nothing_to_save() {
        let mut app = app_from_file_with_display_time(5000);
        let ctx = egui::Context::default();
        let output = run_until_idle(&ctx, &mut app);
        assert!(
            !painted_text(&output).contains(&UNSAVED.to_string()),
            "{:?}",
            painted_text(&output)
        );

        click(&ctx, &mut app, text_center(&output, "Save Settings"));

        let text = painted_text(&run_until_idle(&ctx, &mut app));
        assert!(
            !text.contains(&"OpenAI API key is required".to_string()),
            "{:?}",
            text
        );
    }

    // ===========================================================================
    // Test: Save writes the config file that the settings came from
    // ===========================================================================

    #[test]
    fn test_save_writes_the_config_file_that_was_loaded() {
        use std::fs;

        let dir = std::env::temp_dir().join(format!("babble_boop_gui_save_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        let config_file = dir.join("config.toml");
        let mut file_config = Config::default();
        file_config.openai.api_key = "sk-test".to_string();
        file_config.save(&config_file).unwrap();
        let loaded = Config::load(&config_file).unwrap();
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, loaded.config, log_rx, config_file.clone());
        app.config_draft.translation.target_language = "German".to_string();
        let ctx = egui::Context::default();
        let output = run_until_idle(&ctx, &mut app);

        click(&ctx, &mut app, text_center(&output, "Save Settings"));

        let text = painted_text(&run_until_idle(&ctx, &mut app));
        let saved = Config::load(&config_file).map_err(|e| e.to_string());
        let files: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name())
            .collect();
        // Clean up before asserting, so a failure does not leave files behind
        fs::remove_dir_all(&dir).unwrap();

        assert!(
            text.contains(&"Settings saved successfully".to_string()),
            "{:?}",
            text
        );
        let saved = saved.unwrap().config;
        assert_eq!(saved.translation.target_language, "German");
        assert_eq!(saved.openai.api_key, "sk-test");
        assert_eq!(files, ["config.toml"]);
    }
}
