//! Known OpenAI models: prices, request options and the models the settings
//! suggest. The config takes any
//! model name; a name that is not in these tables gets the default model
//! prices and no extra request options.
//!
//! Source: developers.openai.com, checked 2026-09-28. Prices from
//! /api/docs/pricing (Standard tier, short context), reasoning effort
//! support from each model page under /api/docs/models.

/// Message role for the translation instructions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstructionsRole {
    /// GPT-4o and GPT-4.1 models, and model names that are not known.
    System,
    /// Reasoning models. The Chat Completions reference says: "With o1
    /// models and newer, `developer` messages replace the previous `system`
    /// messages."
    Developer,
}

/// A model for the Chat Completions API.
pub struct ChatModel {
    pub name: &'static str,
    /// US dollars for each million input tokens.
    pub input_price: f64,
    /// US dollars for each million output tokens. Reasoning tokens are
    /// billed as output tokens.
    pub output_price: f64,
    pub instructions_role: InstructionsRole,
    /// `reasoning_effort` to send. Translation does not need reasoning, and
    /// reasoning tokens add cost and delay. `None` sends nothing, so the
    /// model uses its own default.
    pub reasoning_effort: Option<&'static str>,
    /// Shown in the model list of the settings.
    pub suggested: bool,
}

/// A model for the audio transcriptions API.
pub struct TranscriptionModel {
    pub name: &'static str,
    /// US dollars for each minute of audio.
    pub price_per_minute: f64,
    /// Shown in the model list of the settings.
    pub suggested: bool,
}

const fn chat(
    name: &'static str,
    input_price: f64,
    output_price: f64,
    instructions_role: InstructionsRole,
    reasoning_effort: Option<&'static str>,
    suggested: bool,
) -> ChatModel {
    ChatModel {
        name,
        input_price,
        output_price,
        instructions_role,
        reasoning_effort,
        suggested,
    }
}

use InstructionsRole::{Developer, System};

/// Translation model for a new config. Its prices also apply to a model that
/// is not in `CHAT_MODELS`.
pub const DEFAULT_CHAT_MODEL: ChatModel =
    chat("gpt-6-luna", 0.10, 0.50, Developer, Some("none"), true);

pub const CHAT_MODELS: &[ChatModel] = &[
    // GPT-6 Luna (the default) and Sol support reasoning effort `none`.
    DEFAULT_CHAT_MODEL,
    chat("gpt-6-sol", 2.00, 10.00, Developer, Some("none"), true),
    // GPT-5.4 nano and mini use reasoning effort `none` by default.
    chat("gpt-5.4-nano", 0.20, 1.25, Developer, None, true),
    chat("gpt-5.4-mini", 0.75, 4.50, Developer, None, true),
    // GPT-5.6 models use reasoning effort `medium` by default.
    chat("gpt-5.6-luna", 0.20, 1.20, Developer, None, false),
    chat("gpt-5.6-terra", 2.00, 12.00, Developer, None, false),
    chat("gpt-5.6-sol", 4.00, 20.00, Developer, None, false),
    chat("gpt-4.1", 2.00, 8.00, System, None, false),
    chat("gpt-4.1-mini", 0.40, 1.60, System, None, true),
    chat("gpt-4.1-nano", 0.10, 0.40, System, None, false),
    chat("gpt-4o", 2.50, 10.00, System, None, false),
    chat("gpt-4o-2024-05-13", 5.00, 15.00, System, None, false),
    chat("gpt-4o-mini", 0.15, 0.60, System, None, true),
    chat("gpt-4-turbo", 10.00, 30.00, System, None, false),
    chat("gpt-4-turbo-2024-04-09", 10.00, 30.00, System, None, false),
    chat("gpt-4", 30.00, 60.00, System, None, false),
    chat("gpt-4-0613", 30.00, 60.00, System, None, false),
    chat("gpt-3.5-turbo", 0.50, 1.50, System, None, false),
    chat("gpt-3.5-turbo-0125", 0.50, 1.50, System, None, false),
    chat("gpt-3.5-turbo-1106", 1.00, 2.00, System, None, false),
];

const fn transcription(
    name: &'static str,
    price_per_minute: f64,
    suggested: bool,
) -> TranscriptionModel {
    TranscriptionModel {
        name,
        price_per_minute,
        suggested,
    }
}

/// Transcription model for a new config. Its price also applies to a model
/// that is not in `TRANSCRIPTION_MODELS`.
pub const DEFAULT_TRANSCRIPTION_MODEL: TranscriptionModel =
    transcription("gpt-transcribe", 0.0045, true);

pub const TRANSCRIPTION_MODELS: &[TranscriptionModel] = &[
    DEFAULT_TRANSCRIPTION_MODEL,
    transcription("gpt-4o-mini-transcribe", 0.003, false),
    transcription("gpt-4o-transcribe", 0.006, false),
    transcription("whisper-1", 0.006, false),
];

pub fn chat_model(name: &str) -> Option<&'static ChatModel> {
    CHAT_MODELS.iter().find(|model| model.name == name)
}

pub fn transcription_model(name: &str) -> Option<&'static TranscriptionModel> {
    TRANSCRIPTION_MODELS.iter().find(|model| model.name == name)
}

/// Translation models for the model list of the settings.
pub fn suggested_chat_models() -> Vec<&'static str> {
    CHAT_MODELS
        .iter()
        .filter(|model| model.suggested)
        .map(|model| model.name)
        .collect()
}

/// Transcription models for the model list of the settings.
pub fn suggested_transcription_models() -> Vec<&'static str> {
    TRANSCRIPTION_MODELS
        .iter()
        .filter(|model| model.suggested)
        .map(|model| model.name)
        .collect()
}
