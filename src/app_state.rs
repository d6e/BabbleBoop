use crate::config::{AudioConfig, Config};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::mpsc;

/// Log level for activity log entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogLevel {
    Info,
    Success,
    Error,
}

/// A log entry for the activity log displayed in the GUI.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: Instant,
    pub message: String,
    pub level: LogLevel,
}

/// Logger that outputs to stdout/stderr and the GUI activity log.
#[derive(Clone)]
pub struct Logger {
    log_tx: mpsc::Sender<LogEntry>,
}

impl Logger {
    pub fn new(log_tx: mpsc::Sender<LogEntry>) -> Self {
        Self { log_tx }
    }

    /// Log an info message to stdout and the activity log.
    pub fn info(&self, message: impl Into<String>) {
        let msg = message.into();
        println!("{}", msg);
        let _ = self.log_tx.try_send(LogEntry {
            timestamp: Instant::now(),
            message: msg,
            level: LogLevel::Info,
        });
    }

    /// Log a success message to stdout and the activity log.
    pub fn success(&self, message: impl Into<String>) {
        let msg = message.into();
        println!("{}", msg);
        let _ = self.log_tx.try_send(LogEntry {
            timestamp: Instant::now(),
            message: msg,
            level: LogLevel::Success,
        });
    }

    /// Log an error message to stderr and the activity log.
    pub fn error(&self, message: impl Into<String>) {
        let msg = message.into();
        eprintln!("{}", msg);
        let _ = self.log_tx.try_send(LogEntry {
            timestamp: Instant::now(),
            message: msg,
            level: LogLevel::Error,
        });
    }

    /// Log an API error. Shows raw details to stderr but a cleaner message to the activity log.
    pub fn error_api(&self, message: impl Into<String>) {
        let raw_msg = message.into();
        eprintln!("{}", raw_msg);

        let clean_msg = parse_api_error_for_display(&raw_msg);
        let _ = self.log_tx.try_send(LogEntry {
            timestamp: Instant::now(),
            message: clean_msg,
            level: LogLevel::Error,
        });
    }
}

/// Parse an API error message and extract a user-friendly version for display.
fn parse_api_error_for_display(error: &str) -> String {
    // Try to find JSON in the error message
    if let Some(json_start) = error.find('{') {
        if let Ok(parsed) = serde_json::from_str::<Value>(&error[json_start..]) {
            if let Some(err_obj) = parsed.get("error") {
                // Extract the error code if available
                let code = err_obj
                    .get("code")
                    .and_then(|c| c.as_str())
                    .unwrap_or("");

                // Map common error codes to user-friendly messages
                match code {
                    "invalid_api_key" => {
                        return "Invalid API key. Check your OpenAI API key in settings.".into();
                    }
                    "insufficient_quota" => {
                        return "OpenAI API quota exceeded. Check your billing.".into();
                    }
                    "rate_limit_exceeded" => {
                        return "Rate limit exceeded. Please wait and try again.".into();
                    }
                    "model_not_found" => {
                        return "Model not found. Check your model settings.".into();
                    }
                    _ => {}
                }

                // Fall back to the message field if no specific code matched
                if let Some(message) = err_obj.get("message").and_then(|m| m.as_str()) {
                    // Truncate if too long
                    if message.len() > 120 {
                        return format!("{}...", &message[..117]);
                    }
                    return message.to_string();
                }
            }
        }
    }

    // Fall back to original message if parsing fails
    error.to_string()
}

/// Shared audio parameters that can be hot-reloaded without restarting the audio stream.
/// Uses atomics for lock-free access from the audio callback thread.
pub struct AudioParams {
    pub noise_gate_threshold: AtomicU32, // f32 stored as bits
    pub noise_gate_hold_time: AtomicU32, // f32 stored as bits
    pub silence_threshold: AtomicU32,
}

impl AudioParams {
    pub fn new(config: &AudioConfig) -> Self {
        Self {
            noise_gate_threshold: AtomicU32::new(config.noise_gate_threshold.to_bits()),
            noise_gate_hold_time: AtomicU32::new(config.noise_gate_hold_time.to_bits()),
            silence_threshold: AtomicU32::new(config.silence_threshold),
        }
    }

    pub fn update(&self, config: &AudioConfig) {
        self.noise_gate_threshold
            .store(config.noise_gate_threshold.to_bits(), Ordering::Relaxed);
        self.noise_gate_hold_time
            .store(config.noise_gate_hold_time.to_bits(), Ordering::Relaxed);
        self.silence_threshold
            .store(config.silence_threshold, Ordering::Relaxed);
    }

    pub fn get_noise_gate_threshold(&self) -> f32 {
        f32::from_bits(self.noise_gate_threshold.load(Ordering::Relaxed))
    }

    pub fn get_noise_gate_hold_time(&self) -> f32 {
        f32::from_bits(self.noise_gate_hold_time.load(Ordering::Relaxed))
    }

    pub fn get_silence_threshold(&self) -> u32 {
        self.silence_threshold.load(Ordering::Relaxed)
    }
}

pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub enabled: Arc<AtomicBool>,
    pub shutdown: Arc<AtomicBool>,
    pub command_tx: mpsc::Sender<AppCommand>,
    pub log_tx: mpsc::Sender<LogEntry>,
    pub logger: Logger,
    /// Current audio input level (f32 stored as bits) for the level meter
    pub current_audio_level: Arc<AtomicU32>,
    /// Hot-reloadable audio parameters shared with the audio thread
    pub audio_params: Arc<AudioParams>,
    /// Flag indicating test recording mode is active
    pub test_mode_active: Arc<AtomicBool>,
    /// Buffer for test recording samples (written by audio thread, read by main thread)
    pub test_recording_buffer: Arc<std::sync::Mutex<Vec<f32>>>,
    /// Total API cost (f64 stored as bits) for display in GUI
    pub total_cost: Arc<std::sync::atomic::AtomicU64>,
}

#[derive(Debug)]
pub enum AppCommand {
    SetEnabled(bool),
    UpdateConfig(Config),
    StartTestRecording,
    StopTestRecording,
    TestRecordingComplete(Vec<u8>), // WAV data for playback
    Quit,
}

impl AppState {
    pub fn new(
        config: Config,
        command_tx: mpsc::Sender<AppCommand>,
        log_tx: mpsc::Sender<LogEntry>,
    ) -> Self {
        let audio_params = Arc::new(AudioParams::new(&config.audio));
        let logger = Logger::new(log_tx.clone());
        Self {
            config: Arc::new(RwLock::new(config)),
            enabled: Arc::new(AtomicBool::new(true)),
            shutdown: Arc::new(AtomicBool::new(false)),
            command_tx,
            log_tx,
            logger,
            current_audio_level: Arc::new(AtomicU32::new(0)),
            audio_params,
            test_mode_active: Arc::new(AtomicBool::new(false)),
            test_recording_buffer: Arc::new(std::sync::Mutex::new(Vec::new())),
            total_cost: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn set_total_cost(&self, cost: f64) {
        self.total_cost
            .store(cost.to_bits(), Ordering::Relaxed);
    }

    pub fn get_total_cost(&self) -> f64 {
        f64::from_bits(self.total_cost.load(Ordering::Relaxed))
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}
