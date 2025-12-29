use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

/// Default path to the configuration file
pub const CONFIG_PATH: &str = "config.toml";

#[derive(Deserialize, Serialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Config {
    pub osc: OscConfig,
    pub openai: OpenAiConfig,
    pub translation: TranslationConfig,
    pub audio: AudioConfig,
    pub rate_limit: RateLimitConfig,
    #[serde(alias = "debug", default)]
    pub keep_audio_files: bool,
    #[serde(default = "default_max_audio_files")]
    pub max_audio_files: usize,
    #[serde(default)]
    pub theme: ThemeMode,
}

fn default_max_audio_files() -> usize {
    10
}

impl Default for Config {
    fn default() -> Self {
        Self {
            osc: OscConfig::default(),
            openai: OpenAiConfig::default(),
            translation: TranslationConfig::default(),
            audio: AudioConfig::default(),
            rate_limit: RateLimitConfig::default(),
            keep_audio_files: false,
            max_audio_files: 10,
            theme: ThemeMode::default(),
        }
    }
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let data = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&data)?;
        Ok(config)
    }

    /// Load config from path, or create a default config file if it doesn't exist.
    /// Returns the config and a boolean indicating if a new file was created.
    pub fn load_or_create<P: AsRef<Path>>(path: P) -> Result<(Self, bool), Box<dyn Error>> {
        let path = path.as_ref();
        if path.exists() {
            Ok((Self::load(path)?, false))
        } else {
            let config = Self::default();
            config.save(path)?;
            Ok((config, true))
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let data = toml::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct OscConfig {
    pub address: String,
    pub input_port: u16,
    pub output_port: u16,
    pub max_message_chunks: usize,
    pub display_time: u64,
}

impl Default for OscConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            input_port: 9001,
            output_port: 9000,
            max_message_chunks: 9,
            display_time: 3000,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_transcription_model")]
    pub transcription_model: String,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: "gpt-4o-mini".to_string(),
            transcription_model: "whisper-1".to_string(),
        }
    }
}

fn default_transcription_model() -> String {
    "whisper-1".to_string()
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct TranslationConfig {
    pub target_language: String,
    pub include_original_message: bool,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            target_language: "Japanese".to_string(),
            include_original_message: false,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct AudioConfig {
    pub silence_threshold: u32,
    pub noise_gate_threshold: f32,
    pub noise_gate_hold_time: f32,
    pub min_transcription_duration: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            silence_threshold: 100,
            noise_gate_threshold: 0.3,
            noise_gate_hold_time: 0.20,
            min_transcription_duration: 1.0,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 50,
        }
    }
}
