use crate::config::Config;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

pub struct AppState {
    pub config: Arc<RwLock<Config>>,
    pub enabled: Arc<AtomicBool>,
    pub command_tx: mpsc::Sender<AppCommand>,
}

#[derive(Debug)]
pub enum AppCommand {
    SetEnabled(bool),
    UpdateConfig(Config),
    Quit,
}

impl AppState {
    pub fn new(config: Config, command_tx: mpsc::Sender<AppCommand>) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            enabled: Arc::new(AtomicBool::new(true)),
            command_tx,
        }
    }
}
