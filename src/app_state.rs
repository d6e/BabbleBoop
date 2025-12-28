use crate::config::Config;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::mpsc;

/// A log entry for the activity log displayed in the GUI.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: Instant,
    pub original: String,
    pub translated: String,
}

impl LogEntry {
    pub fn new(original: String, translated: String) -> Self {
        Self {
            timestamp: Instant::now(),
            original,
            translated,
        }
    }
}

pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub enabled: Arc<AtomicBool>,
    pub shutdown: Arc<AtomicBool>,
    pub command_tx: mpsc::Sender<AppCommand>,
    pub log_tx: mpsc::Sender<LogEntry>,
}

#[derive(Debug)]
pub enum AppCommand {
    SetEnabled(bool),
    UpdateConfig(Config),
    Quit,
}

impl AppState {
    pub fn new(
        config: Config,
        command_tx: mpsc::Sender<AppCommand>,
        log_tx: mpsc::Sender<LogEntry>,
    ) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            enabled: Arc::new(AtomicBool::new(true)),
            shutdown: Arc::new(AtomicBool::new(false)),
            command_tx,
            log_tx,
        }
    }

    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}
