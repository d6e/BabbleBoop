use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Config {
    pub osc: OscConfig,
    pub openai: OpenAiConfig,
    pub translation: TranslationConfig,
    pub audio: AudioConfig,
    pub rate_limit: RateLimitConfig,
    pub debug: bool,
}

#[derive(Deserialize, Clone)]
pub struct OscConfig {
    pub address: String,
    pub input_port: u16,
    pub output_port: u16,
    pub max_message_chunks: usize,
    pub display_time: u64,
}

#[derive(Deserialize, Clone)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
}

#[derive(Deserialize, Clone)]
pub struct TranslationConfig {
    pub target_language: String,
    pub include_original_message: bool,
}

#[derive(Deserialize, Clone)]
pub struct AudioConfig {
    pub silence_threshold: u32,
    pub noise_gate_threshold: f32,
    pub noise_gate_hold_time: f32,
    pub min_transcription_duration: f32,
}

#[derive(Deserialize, Clone)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
}
