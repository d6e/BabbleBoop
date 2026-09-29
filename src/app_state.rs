use crate::config::{AudioConfig, Config};
use crate::recorder::{RecorderSettings, RecorderStatus};
use crate::shutdown::Shutdown;
use eframe::egui;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, TryLockError};
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

/// An `f32` that threads share without a lock, stored as its bits in an
/// `AtomicU32`. Loads and stores are `Relaxed`: each value stands alone and
/// orders no other memory.
#[derive(Debug, Default)]
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    pub fn store(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }
}

/// Audio settings that change without a restart of the audio stream. The
/// audio callback reads them for each buffer.
pub struct AudioParams {
    noise_gate_threshold: AtomicF32,
    noise_gate_hold_time: AtomicF32,
    silence_duration: AtomicF32,
}

impl AudioParams {
    pub fn new(config: &AudioConfig) -> Self {
        Self {
            noise_gate_threshold: AtomicF32::new(config.noise_gate_threshold),
            noise_gate_hold_time: AtomicF32::new(config.noise_gate_hold_time),
            silence_duration: AtomicF32::new(config.silence_duration),
        }
    }

    pub fn update(&self, config: &AudioConfig) {
        self.noise_gate_threshold.store(config.noise_gate_threshold);
        self.noise_gate_hold_time.store(config.noise_gate_hold_time);
        self.silence_duration.store(config.silence_duration);
    }

    /// The settings for the recorder. A change between two loads can mix
    /// old and new values for one buffer; the next buffer has all new ones.
    pub fn recorder_settings(&self) -> RecorderSettings {
        RecorderSettings {
            noise_gate_threshold: self.noise_gate_threshold.load(),
            noise_gate_hold_time: self.noise_gate_hold_time.load(),
            silence_duration: self.silence_duration.load(),
        }
    }
}

/// The state that the audio callback shares with the GUI and the processing
/// loop. The callback must never wait, so it reads and writes this state
/// with atomics and `try_lock` only.
pub struct AudioShared {
    /// Set by the processing loop from its settings
    pub params: AudioParams,
    /// Peak level of the last buffer, for the level meter
    pub level: AtomicF32,
    /// The recorder status after the last buffer. One snapshot, so the GUI
    /// never shows fields of two different buffers.
    status: Mutex<RecorderStatus>,
    /// Whether a test recording runs. While it does, the callback copies
    /// the input into `test_buffer` instead of the recorder.
    pub test_mode: AtomicBool,
    /// The samples of the test recording. The callback writes it; the
    /// processing loop swaps it at start and stop (`TestRecording`).
    pub test_buffer: Mutex<Vec<f32>>,
}

impl AudioShared {
    /// Shared state with default settings and an idle recorder.
    pub fn new() -> Self {
        Self {
            params: AudioParams::new(&AudioConfig::default()),
            level: AtomicF32::default(),
            status: Mutex::new(RecorderStatus::default()),
            test_mode: AtomicBool::new(false),
            test_buffer: Mutex::new(Vec::new()),
        }
    }

    /// Replace the recorder status, unless another thread holds the lock.
    /// Called from the audio callback, which must not wait: the GUI then
    /// shows the old status until the next buffer publishes again.
    pub fn publish(&self, status: RecorderStatus) {
        match self.status.try_lock() {
            Ok(mut current) => *current = status,
            // The lock only guards a copy of a `Copy` value, so the value
            // is whole even if a thread panicked while it held the lock.
            Err(TryLockError::Poisoned(poisoned)) => *poisoned.into_inner() = status,
            Err(TryLockError::WouldBlock) => {}
        }
    }

    /// The last published recorder status.
    pub fn status(&self) -> RecorderStatus {
        *self.lock_status()
    }

    /// Hold the status lock. `status` holds it only to copy the value.
    fn lock_status(&self) -> MutexGuard<'_, RecorderStatus> {
        self.status.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Hold the status lock, as the GUI does while it copies the status.
    #[cfg(test)]
    pub(crate) fn hold_status(&self) -> MutexGuard<'_, RecorderStatus> {
        self.lock_status()
    }
}

impl Default for AudioShared {
    fn default() -> Self {
        Self::new()
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
    /// State shared with the audio callback. The processing loop sets its
    /// parameters from its settings before the audio input starts and when
    /// the settings change.
    pub audio: Arc<AudioShared>,
    /// Total API cost (f64 stored as bits) for display in GUI
    pub total_cost: Arc<std::sync::atomic::AtomicU64>,
    /// Set when the processing thread ends
    processing_stopped: AtomicBool,
}

#[derive(Debug)]
pub enum AppCommand {
    SetEnabled(bool),
    UpdateConfig(Config),
    StartTestRecording,
    StopTestRecording,
}

impl AppState {
    pub fn new(command_tx: mpsc::Sender<AppCommand>, log_tx: mpsc::Sender<LogEntry>) -> Self {
        let gui_waker = GuiWaker::default();
        let logger = Logger::new(log_tx.clone(), gui_waker.clone());
        Self {
            enabled: Arc::new(AtomicBool::new(true)),
            shutdown: Shutdown::new(),
            command_tx,
            log_tx,
            logger,
            gui_waker,
            audio: Arc::new(AudioShared::new()),
            total_cost: Arc::new(std::sync::atomic::AtomicU64::new(0)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_the_status_is_read_as_one_snapshot() {
        // Every field differs between the two, so a status with fields of
        // both is neither
        let first = RecorderStatus {
            is_recording: true,
            quiet_time: 1.0,
            gate_open: true,
            hold_remaining: 1.0,
            recording_duration: 1.0,
            split: true,
        };
        let second = RecorderStatus {
            is_recording: false,
            quiet_time: 2.0,
            gate_open: false,
            hold_remaining: 2.0,
            recording_duration: 2.0,
            split: false,
        };
        let audio = AudioShared::new();
        audio.publish(first);
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(Ordering::Relaxed) {
                    audio.publish(second);
                    audio.publish(first);
                }
            });
            let mixed = (0..200_000)
                .map(|_| audio.status())
                .find(|status| *status != first && *status != second);
            done.store(true, Ordering::Relaxed);
            assert_eq!(mixed, None);
        });
    }
}
