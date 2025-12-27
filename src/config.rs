use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

/// Default path to the configuration file
pub const CONFIG_PATH: &str = "config.toml";

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Config {
    pub osc: OscConfig,
    pub openai: OpenAiConfig,
    pub translation: TranslationConfig,
    pub audio: AudioConfig,
    pub rate_limit: RateLimitConfig,
    pub keep_audio_files: bool,
    pub max_audio_files: usize,
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let data = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&data)?;
        Ok(config)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let data = toml::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OscConfig {
    pub address: String,
    pub input_port: u16,
    pub output_port: u16,
    pub max_message_chunks: usize,
    pub display_time: u64,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct TranslationConfig {
    pub target_language: String,
    pub include_original_message: bool,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct AudioConfig {
    pub silence_threshold: u32,
    pub noise_gate_threshold: f32,
    pub noise_gate_hold_time: f32,
    pub min_transcription_duration: f32,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
}
