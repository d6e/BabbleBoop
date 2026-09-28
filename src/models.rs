//! Known OpenAI models and their prices. The config takes any model name; a
//! name that is not in these tables gets the default model prices.
//!
//! Source: developers.openai.com, checked 2026-09-28. Prices from
//! /api/docs/pricing (Standard tier, short context).

/// A model for the Chat Completions API.
pub struct ChatModel {
    pub name: &'static str,
    /// US dollars for each million input tokens.
    pub input_price: f64,
    /// US dollars for each million output tokens. Reasoning tokens are
    /// billed as output tokens.
    pub output_price: f64,
}

/// A model for the audio transcriptions API.
pub struct TranscriptionModel {
    pub name: &'static str,
    /// US dollars for each minute of audio.
    pub price_per_minute: f64,
}

const fn chat(name: &'static str, input_price: f64, output_price: f64) -> ChatModel {
    ChatModel {
        name,
        input_price,
        output_price,
    }
}

/// Translation model for a new config. Its prices also apply to a model that
/// is not in `CHAT_MODELS`.
pub const DEFAULT_CHAT_MODEL: ChatModel = chat("gpt-4o-mini", 0.15, 0.60);

pub const CHAT_MODELS: &[ChatModel] = &[
    chat("gpt-6-luna", 0.10, 0.50),
    chat("gpt-6-sol", 2.00, 10.00),
    chat("gpt-5.4-nano", 0.20, 1.25),
    chat("gpt-5.4-mini", 0.75, 4.50),
    chat("gpt-5.6-luna", 0.20, 1.20),
    chat("gpt-5.6-terra", 2.00, 12.00),
    chat("gpt-5.6-sol", 4.00, 20.00),
    chat("gpt-4.1", 2.00, 8.00),
    chat("gpt-4.1-mini", 0.40, 1.60),
    chat("gpt-4.1-nano", 0.10, 0.40),
    chat("gpt-4o", 2.50, 10.00),
    chat("gpt-4o-2024-05-13", 5.00, 15.00),
    DEFAULT_CHAT_MODEL,
    chat("gpt-4-turbo", 10.00, 30.00),
    chat("gpt-4-turbo-2024-04-09", 10.00, 30.00),
    chat("gpt-4", 30.00, 60.00),
    chat("gpt-4-0613", 30.00, 60.00),
    chat("gpt-3.5-turbo", 0.50, 1.50),
    chat("gpt-3.5-turbo-0125", 0.50, 1.50),
    chat("gpt-3.5-turbo-1106", 1.00, 2.00),
];

const fn transcription(name: &'static str, price_per_minute: f64) -> TranscriptionModel {
    TranscriptionModel {
        name,
        price_per_minute,
    }
}

/// Transcription model for a new config. Its price also applies to a model
/// that is not in `TRANSCRIPTION_MODELS`.
pub const DEFAULT_TRANSCRIPTION_MODEL: TranscriptionModel = transcription("whisper-1", 0.006);

pub const TRANSCRIPTION_MODELS: &[TranscriptionModel] = &[
    transcription("gpt-transcribe", 0.0045),
    transcription("gpt-4o-mini-transcribe", 0.003),
    transcription("gpt-4o-transcribe", 0.006),
    DEFAULT_TRANSCRIPTION_MODEL,
];

pub fn chat_model(name: &str) -> Option<&'static ChatModel> {
    CHAT_MODELS.iter().find(|model| model.name == name)
}

pub fn transcription_model(name: &str) -> Option<&'static TranscriptionModel> {
    TRANSCRIPTION_MODELS.iter().find(|model| model.name == name)
}
