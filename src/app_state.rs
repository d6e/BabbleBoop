use crate::config::{AudioConfig, Config};
use crate::shutdown::Shutdown;
use eframe::egui;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
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

/// Wakes the GUI when another thread changes what it shows. The GUI
/// repaints only on input or on request, so without this a new log entry
/// or cost stays hidden until the mouse moves over the window.
#[derive(Clone, Default)]
pub struct GuiWaker {
    ctx: Arc<OnceLock<egui::Context>>,
}

impl GuiWaker {
    /// Set the context to wake. The GUI calls this once when it starts.
    pub fn attach(&self, ctx: egui::Context) {
        #[expect(
            clippy::let_underscore_must_use,
            reason = "set fails only if a context is attached already, and that context stays"
        )]
        let _ = self.ctx.set(ctx);
    }

    /// Ask the GUI to repaint. Does nothing before the GUI starts; its first
    /// frame shows the current state.
    pub fn wake(&self) {
        if let Some(ctx) = self.ctx.get() {
            ctx.request_repaint();
        }
    }
}

/// Logger that outputs to stdout/stderr and the GUI activity log.
#[derive(Clone)]
pub struct Logger {
    log_tx: mpsc::Sender<LogEntry>,
    gui_waker: GuiWaker,
}

impl Logger {
    pub fn new(log_tx: mpsc::Sender<LogEntry>, gui_waker: GuiWaker) -> Self {
        Self { log_tx, gui_waker }
    }

    /// Log an info message to stdout and the activity log.
    pub fn info(&self, message: impl Into<String>) {
        let msg = message.into();
        println!("{}", msg);
        self.send(msg, LogLevel::Info);
    }

    /// Log a success message to stdout and the activity log.
    pub fn success(&self, message: impl Into<String>) {
        let msg = message.into();
        println!("{}", msg);
        self.send(msg, LogLevel::Success);
    }

    /// Log an error message to stderr and the activity log.
    pub fn error(&self, message: impl Into<String>) {
        let msg = message.into();
        eprintln!("{}", msg);
        self.send(msg, LogLevel::Error);
    }

    /// Log an error with `message` in the activity log, and with `message`
    /// and `details` on stderr. For an error whose whole text does not
    /// help the user, such as the raw response of an API.
    pub fn error_with_details(&self, message: impl Into<String>, details: &str) {
        let msg = message.into();
        eprintln!("{}\n  {}", msg, details);
        self.send(msg, LogLevel::Error);
    }

    /// Add an entry to the activity log and wake the GUI to show it.
    fn send(&self, message: String, level: LogLevel) {
        #[expect(
            clippy::let_underscore_must_use,
            reason = "logging does not wait: an entry is dropped if the channel is full or closed"
        )]
        let _ = self.log_tx.try_send(LogEntry {
            timestamp: Instant::now(),
            message,
            level,
        });
        self.gui_waker.wake();
    }
}

/// The last failure of an action that runs again and again, such as a send
/// on every utterance. While the cause stays, each attempt fails the same
/// way, so the activity log gets a failure only when it differs from the
/// failure before it.
#[derive(Default)]
pub struct FailureLog {
    last: Option<String>,
}

impl FailureLog {
    /// Log `message` as an error, unless the last attempt failed with the
    /// same message.
    pub fn failed(&mut self, logger: &Logger, message: String) {
        if self.last.as_ref() != Some(&message) {
            logger.error(message.as_str());
            self.last = Some(message);
        }
    }

    /// Record a success. Returns true if the last attempt failed, so the
    /// caller can log that the action works again.
    #[must_use]
    pub fn succeeded(&mut self) -> bool {
        self.last.take().is_some()
    }
}

/// Run the body of a background thread and log in the activity log if it
/// returns an error or panics, so that the thread does not stop with a
/// message on stderr only. The panic does not propagate.
pub fn run_logging_failure<E: std::fmt::Display>(
    logger: &Logger,
    name: &str,
    body: impl FnOnce() -> Result<(), E>,
) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => {}
        Ok(Err(e)) => logger.error(format!("{} stopped: {}", name, e)),
        Err(payload) => {
            logger.error(format!("{} crashed: {}", name, panic_reason(&*payload)));
        }
    }
}

/// The message of a panic caught with `catch_unwind`.
pub fn panic_reason(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
}

/// Shared audio parameters that can be hot-reloaded without restarting the audio stream.
/// Uses atomics for lock-free access from the audio callback thread.
pub struct AudioParams {
    pub noise_gate_threshold: AtomicU32, // f32 stored as bits
    pub noise_gate_hold_time: AtomicU32, // f32 stored as bits
    pub silence_duration: AtomicU32,     // f32 stored as bits
}

impl AudioParams {
    pub fn new(config: &AudioConfig) -> Self {
        Self {
            noise_gate_threshold: AtomicU32::new(config.noise_gate_threshold.to_bits()),
            noise_gate_hold_time: AtomicU32::new(config.noise_gate_hold_time.to_bits()),
            silence_duration: AtomicU32::new(config.silence_duration.to_bits()),
        }
    }

    pub fn update(&self, config: &AudioConfig) {
        self.noise_gate_threshold
            .store(config.noise_gate_threshold.to_bits(), Ordering::Relaxed);
        self.noise_gate_hold_time
            .store(config.noise_gate_hold_time.to_bits(), Ordering::Relaxed);
        self.silence_duration
            .store(config.silence_duration.to_bits(), Ordering::Relaxed);
    }

    pub fn get_noise_gate_threshold(&self) -> f32 {
        f32::from_bits(self.noise_gate_threshold.load(Ordering::Relaxed))
    }

    pub fn get_noise_gate_hold_time(&self) -> f32 {
        f32::from_bits(self.noise_gate_hold_time.load(Ordering::Relaxed))
    }

    pub fn get_silence_duration(&self) -> f32 {
        f32::from_bits(self.silence_duration.load(Ordering::Relaxed))
    }
}

pub struct AppState {
    pub enabled: Arc<AtomicBool>,
    pub shutdown: Shutdown,
    pub command_tx: mpsc::Sender<AppCommand>,
    pub log_tx: mpsc::Sender<LogEntry>,
    pub logger: Logger,
    /// Wakes the GUI when state it shows changes on another thread
    pub gui_waker: GuiWaker,
    /// Current audio input level (f32 stored as bits) for the level meter
    pub current_audio_level: Arc<AtomicU32>,
    /// Hot-reloadable audio parameters shared with the audio thread. The
    /// processing loop sets them from its settings before the audio input
    /// starts and when the settings change.
    pub audio_params: Arc<AudioParams>,
    /// Flag indicating test recording mode is active
    pub test_mode_active: Arc<AtomicBool>,
    /// Buffer for test recording samples. The audio callback writes it; the
    /// processing thread swaps it at start and stop (`TestRecording`).
    pub test_recording_buffer: Arc<std::sync::Mutex<Vec<f32>>>,
    /// Total API cost (f64 stored as bits) for display in GUI
    pub total_cost: Arc<std::sync::atomic::AtomicU64>,
    /// Whether audio is currently being recorded
    pub is_recording: Arc<AtomicBool>,
    /// Seconds of input since the noise gate closed during the current
    /// recording (f32 stored as bits)
    pub quiet_time: Arc<AtomicU32>,
    /// Whether the noise gate is currently active/open
    pub noise_gate_active: Arc<AtomicBool>,
    /// Remaining hold time in seconds (f32 stored as bits)
    pub noise_gate_hold_remaining: Arc<AtomicU32>,
    /// Current recording duration in seconds (f32 stored as bits)
    pub recording_duration: Arc<AtomicU32>,
    /// Whether the current recording reached the length limit
    pub recording_split: Arc<AtomicBool>,
    /// Set when the processing thread ends
    processing_stopped: AtomicBool,
}

#[derive(Debug)]
pub enum AppCommand {
    SetEnabled(bool),
    UpdateConfig(Config),
    StartTestRecording,
    StopTestRecording,
    Quit,
}

impl AppState {
    pub fn new(command_tx: mpsc::Sender<AppCommand>, log_tx: mpsc::Sender<LogEntry>) -> Self {
        let audio_params = Arc::new(AudioParams::new(&AudioConfig::default()));
        let gui_waker = GuiWaker::default();
        let logger = Logger::new(log_tx.clone(), gui_waker.clone());
        Self {
            enabled: Arc::new(AtomicBool::new(true)),
            shutdown: Shutdown::new(),
            command_tx,
            log_tx,
            logger,
            gui_waker,
            current_audio_level: Arc::new(AtomicU32::new(0)),
            audio_params,
            test_mode_active: Arc::new(AtomicBool::new(false)),
            test_recording_buffer: Arc::new(std::sync::Mutex::new(Vec::new())),
            total_cost: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            is_recording: Arc::new(AtomicBool::new(false)),
            quiet_time: Arc::new(AtomicU32::new(0)),
            noise_gate_active: Arc::new(AtomicBool::new(false)),
            noise_gate_hold_remaining: Arc::new(AtomicU32::new(0)),
            recording_duration: Arc::new(AtomicU32::new(0)),
            recording_split: Arc::new(AtomicBool::new(false)),
            processing_stopped: AtomicBool::new(false),
        }
    }

    pub fn set_total_cost(&self, cost: f64) {
        self.total_cost.store(cost.to_bits(), Ordering::Relaxed);
        self.gui_waker.wake();
    }

    pub fn get_total_cost(&self) -> f64 {
        f64::from_bits(self.total_cost.load(Ordering::Relaxed))
    }

    /// Record that the processing thread ended, so the GUI stops showing
    /// translation as enabled.
    pub fn mark_processing_stopped(&self) {
        self.processing_stopped.store(true, Ordering::Relaxed);
        self.gui_waker.wake();
    }

    pub fn is_processing_stopped(&self) -> bool {
        self.processing_stopped.load(Ordering::Relaxed)
    }

    pub fn request_shutdown(&self) {
        self.shutdown.request();
    }

    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown.is_requested()
    }
}
