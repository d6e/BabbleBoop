use crate::config::{AudioConfig, Config};
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

impl LogEntry {
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            timestamp: Instant::now(),
            message: message.into(),
            level: LogLevel::Info,
        }
    }

    pub fn success(message: impl Into<String>) -> Self {
        Self {
            timestamp: Instant::now(),
            message: message.into(),
            level: LogLevel::Success,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            timestamp: Instant::now(),
            message: message.into(),
            level: LogLevel::Error,
        }
    }
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
        Self {
            config: Arc::new(RwLock::new(config)),
            enabled: Arc::new(AtomicBool::new(true)),
            shutdown: Arc::new(AtomicBool::new(false)),
            command_tx,
            log_tx,
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
