//! Helpers shared by more than one module's `#[cfg(test)] mod tests`.
//!
//! Each helper here replaces two or more near identical copies that used to
//! live in `src/tests.rs` or scattered across individual test modules.

use crate::app_state::{AppCommand, AppState, LogEntry, LogLevel, Logger};
use crate::price_estimator::{PriceEstimator, TokenCounts};
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

// ===========================================================================
// Activity log capture
// ===========================================================================

/// A `Logger` and the channel that receives what it sends, for a test that
/// checks what was logged.
pub(crate) struct LogCapture {
    logger: Logger,
    rx: mpsc::Receiver<LogEntry>,
}

impl LogCapture {
    /// A log channel with room for `capacity` entries.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        LogCapture {
            logger: Logger::new(tx, Default::default()),
            rx,
        }
    }

    /// A log channel with room for 10 entries, enough for any one test.
    pub(crate) fn new() -> Self {
        Self::with_capacity(10)
    }

    pub(crate) fn logger(&self) -> Logger {
        self.logger.clone()
    }

    /// Level and text of the entries logged since the last call.
    pub(crate) fn entries(&mut self) -> Vec<(LogLevel, String)> {
        std::iter::from_fn(|| self.rx.try_recv().ok())
            .map(|entry| (entry.level, entry.message))
            .collect()
    }

    /// The full entries logged since the last call.
    pub(crate) fn full_entries(&mut self) -> Vec<LogEntry> {
        std::iter::from_fn(|| self.rx.try_recv().ok()).collect()
    }
}

/// A `Logger` whose entries nobody reads, for a test that needs one but
/// does not check what it logs.
pub(crate) fn silent_logger() -> Logger {
    LogCapture::new().logger()
}

// ===========================================================================
// AppState builder
// ===========================================================================

/// `AppState` with fresh command and log channels, for a test that does not
/// need to reach either channel.
pub(crate) fn app_state() -> Arc<AppState> {
    let (state, _cmd_rx, _log_rx) = app_state_with_channels();
    state
}

/// `AppState` plus its command and log receivers, for a test that reaches
/// one of the channels directly.
pub(crate) fn app_state_with_channels() -> (
    Arc<AppState>,
    mpsc::Receiver<AppCommand>,
    mpsc::Receiver<LogEntry>,
) {
    let (cmd_tx, cmd_rx) = mpsc::channel(10);
    let (log_tx, log_rx) = mpsc::channel(10);
    (Arc::new(AppState::new(cmd_tx, log_tx)), cmd_rx, log_rx)
}

// ===========================================================================
// A self cleaning temp directory
// ===========================================================================

/// A fresh, empty directory under the system temp directory, removed on
/// drop, including after a panic. Named after `name` and the process id, so
/// parallel test runs and processes do not share it.
pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// A fresh, empty directory named `babble_boop_<name>_<pid>`. Any
    /// leftover directory of the same name, such as from a previous run
    /// that panicked, is removed first.
    pub(crate) fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("babble_boop_{}_{}", name, std::process::id()));
        if path.exists() {
            make_writable(&path);
            fs::remove_dir_all(&path).unwrap();
        }
        fs::create_dir_all(&path).unwrap();
        TempDir { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if self.path.exists() {
            make_writable(&self.path);
            _ = fs::remove_dir_all(&self.path);
        }
    }
}

/// Give read, write and search permission to `path` and everything under
/// it, so a directory or file a test made read only can still be removed.
fn make_writable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(metadata) = fs::symlink_metadata(path) {
            if !metadata.file_type().is_symlink() {
                let mut perms = metadata.permissions();
                perms.set_mode(0o700);
                _ = fs::set_permissions(path, perms);
            }
            if metadata.is_dir() {
                if let Ok(entries) = fs::read_dir(path) {
                    for entry in entries.flatten() {
                        make_writable(&entry.path());
                    }
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        if let Ok(metadata) = fs::metadata(path) {
            let mut perms = metadata.permissions();
            perms.set_readonly(false);
            _ = fs::set_permissions(path, perms);
        }
    }
}

// ===========================================================================
// OSC receiver with decode
// ===========================================================================

/// A decoded VRChat chatbox OSC message: whether the typing indicator is
/// shown, or the chat text to display.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Osc {
    Typing(bool),
    Input(String),
}

/// The next chatbox OSC message at `socket`, decoded, waited for up to
/// `timeout`. Panics if nothing arrives in time, if a bundle arrives
/// instead of a message, or if the message is not a chatbox typing or
/// input message.
pub(crate) async fn recv_osc(
    socket: &tokio::net::UdpSocket,
    timeout: Duration,
) -> (Osc, SocketAddr) {
    let mut buf = [0u8; 1024];
    let (len, sender) = tokio::time::timeout(timeout, socket.recv_from(&mut buf))
        .await
        .expect("no OSC message was received")
        .unwrap();
    let rosc::OscPacket::Message(message) = rosc::decoder::decode_udp(&buf[..len]).unwrap().1
    else {
        panic!("received an OSC bundle");
    };
    let osc = match (message.addr.as_str(), message.args.first()) {
        ("/chatbox/typing", Some(rosc::OscType::Bool(typing))) => Osc::Typing(*typing),
        ("/chatbox/input", Some(rosc::OscType::String(text))) => Osc::Input(text.clone()),
        _ => panic!("unexpected OSC message {:?}", message),
    };
    (osc, sender)
}

// ===========================================================================
// Audio test signals
// ===========================================================================

/// A sine wave of `seconds` at `sample_rate`, with the given `amplitude`.
pub(crate) fn sine(frequency: f32, sample_rate: u32, seconds: f32, amplitude: f32) -> Vec<f32> {
    let len = (sample_rate as f32 * seconds) as usize;
    (0..len)
        .map(|n| {
            let t = n as f32 / sample_rate as f32;
            amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect()
}

/// The number of times consecutive `samples` cross zero, a rough measure of
/// frequency independent of amplitude.
pub(crate) fn zero_crossings<T: PartialOrd + Default>(samples: impl Iterator<Item = T>) -> usize {
    let zero = T::default();
    let signs: Vec<bool> = samples.map(|s| s < zero).collect();
    signs.windows(2).filter(|pair| pair[0] != pair[1]).count()
}

// ===========================================================================
// Chat Completions response bodies
// ===========================================================================

/// Chat Completions response body in the shape the API returns. `usage` is
/// the JSON of the usage field, or `None` to leave the field out.
pub(crate) fn chat_completion_body(content: &str, usage: Option<&str>) -> String {
    chat_completion_body_with(
        &serde_json::to_string(content).unwrap(),
        "null",
        "stop",
        usage,
    )
}

/// Chat Completions response body with one choice. `content` and `refusal`
/// are the JSON of the message fields, such as `null`.
pub(crate) fn chat_completion_body_with(
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
pub(crate) const REASONING_USAGE: &str = r#"{
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

/// Usage of a short response, such as a refusal.
pub(crate) const SHORT_USAGE: &str = r#"{
    "prompt_tokens": 58,
    "completion_tokens": 12,
    "total_tokens": 70
}"#;

// ===========================================================================
// Pricing helpers
// ===========================================================================

/// Token counts for comparing translation prices.
pub(crate) const TOKENS: TokenCounts = TokenCounts {
    input: 1000,
    output: 500,
};

/// A data folder that does not exist, for a test that saves no cost and no
/// recordings. A save there fails instead of writing to the working
/// directory.
pub(crate) fn missing_data_dir() -> crate::data_dir::DataDir {
    crate::data_dir::DataDir::new(
        std::env::temp_dir().join(format!("babble_boop_no_data_dir_{}", std::process::id())),
    )
}

/// An estimator for the prices of these models, backed by `missing_data_dir`
/// so nothing it saves reaches disk.
pub(crate) fn price_estimator(model: &str, transcription_model: &str) -> PriceEstimator {
    PriceEstimator::new(missing_data_dir().cost_file(), model, transcription_model)
}

// ===========================================================================
// The minimum transcription duration check
// ===========================================================================

/// What `Pipeline::process` did with a recording.
#[derive(Debug, PartialEq)]
pub(crate) enum MinimumCheck {
    /// The recording was skipped as too short.
    Skipped,
    /// The recording went on to transcription.
    Transcribed,
}

/// Run `Pipeline::process` on `audio`, which holds `extent` of a recording,
/// with `min_seconds` as the minimum transcription duration. The API is a
/// local server that answers every request with 404.
pub(crate) async fn check_against_minimum(
    audio: crate::types::CapturedAudio,
    extent: crate::types::Extent,
    min_seconds: f32,
) -> MinimumCheck {
    use crate::api_client::test_server::TestServer;
    use crate::api_client::OpenAi;
    use crate::chatbox::Chatbox;
    use crate::config::Config;
    use crate::pipeline::{Pipeline, ProcessingServices};
    use crate::processing_loop::encode_for_upload;
    use crate::rate_limiter::RateLimiter;
    use crate::typing_indicator::TypingIndicator;
    use tokio::net::UdpSocket;

    let mut server = TestServer::start(Vec::new()).await;
    let api = OpenAi::new(&server.base_url).unwrap();

    let chatbox = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let mut config = Config::default();
    config.osc.address = "127.0.0.1".to_string();
    config.osc.output_port = chatbox.local_addr().unwrap().port();
    config.audio.min_transcription_duration = min_seconds;
    let (app_state, _cmd_rx, _log_rx) = app_state_with_channels();
    let typing_indicator = TypingIndicator::new(socket.clone(), app_state.logger.clone());
    let (wav, audio_duration) = encode_for_upload(audio).await.unwrap();

    let mut services = ProcessingServices::new(&config, &missing_data_dir(), &app_state.logger);
    services.rate_limiter = RateLimiter::new(50);
    services.price_estimator =
        price_estimator(&config.openai.model, &config.openai.transcription_model);
    let mut pipeline = Pipeline::new(
        Arc::clone(&app_state),
        api,
        Chatbox::new(socket),
        typing_indicator,
        services,
    );

    let result = tokio::time::timeout(
        Duration::from_secs(10),
        pipeline.process(wav, audio_duration, extent, &config),
    )
    .await
    .expect("Pipeline::process did not finish");

    if !server.received().is_empty() {
        assert!(result.is_err(), "the server answered 404");
        MinimumCheck::Transcribed
    } else {
        assert!(result.is_ok(), "{:?}", result.err());
        MinimumCheck::Skipped
    }
}
