//! Known OpenAI models: prices, request options and the models the settings
//! suggest. The config takes any
//! model name; a name that is not in these tables gets the default model
//! prices and no extra request options.
//!
//! Source: developers.openai.com, checked 2026-09-28. Prices from
//! /api/docs/pricing (Standard tier, short context), shutdown dates and
//! replacements from /api/docs/deprecations, reasoning effort support from
//! each model page under /api/docs/models.

/// Scheduled removal of a model from the API.
pub struct Shutdown {
    /// Shutdown date as YYYY-MM-DD.
    pub date: &'static str,
    /// Model that OpenAI recommends instead.
    pub replacement: &'static str,
}

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
    pub shutdown: Option<Shutdown>,
}

/// A model for the audio transcriptions API.
pub struct TranscriptionModel {
    pub name: &'static str,
    /// US dollars for each minute of audio.
    pub price_per_minute: f64,
    /// Shown in the model list of the settings.
    pub suggested: bool,
    pub shutdown: Option<Shutdown>,
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
        shutdown: None,
    }
}

/// Row of `CHAT_MODELS` for a GPT-4.1, GPT-4o or older model that is
/// scheduled to shut down.
const fn retiring_chat(
    name: &'static str,
    input_price: f64,
    output_price: f64,
    date: &'static str,
    replacement: &'static str,
) -> ChatModel {
    ChatModel {
        shutdown: Some(Shutdown { date, replacement }),
        ..chat(name, input_price, output_price, System, None, false)
    }
}

use InstructionsRole::{Developer, System};

/// Translation model for a new config. Its prices also apply to a model that
/// is not in `CHAT_MODELS`.
pub const DEFAULT_CHAT_MODEL: ChatModel =
    chat("gpt-6-luna", 0.10, 0.50, Developer, Some("none"), true);

#[rustfmt::skip]
pub const CHAT_MODELS: &[ChatModel] = &[
    // GPT-6 Luna (the default) and Sol support reasoning effort `none`.
    DEFAULT_CHAT_MODEL,
    chat("gpt-6-sol", 2.00, 10.00, Developer, Some("none"), true),
    // GPT-5.4 nano and mini use reasoning effort `none` by default.
    chat("gpt-5.4-nano", 0.20, 1.25, Developer, None, true),
    chat("gpt-5.4-mini", 0.75, 4.50, Developer, None, true),
    // GPT-5.6 models use reasoning effort `medium` by default and support
    // `none`. The shutdown warnings recommend them.
    chat("gpt-5.6-luna", 0.20, 1.20, Developer, Some("none"), false),
    chat("gpt-5.6-terra", 2.00, 12.00, Developer, Some("none"), false),
    // Promotional price, available at least through 2026-11-21
    // (/api/docs/pricing, note below the Standard table).
    chat("gpt-5.6-sol", 4.00, 20.00, Developer, Some("none"), false),
    chat("gpt-4.1", 2.00, 8.00, System, None, false),
    chat("gpt-4.1-mini", 0.40, 1.60, System, None, true),
    retiring_chat("gpt-4.1-nano", 0.10, 0.40, "2026-10-23", "gpt-5.6-luna"),
    chat("gpt-4o", 2.50, 10.00, System, None, false),
    // Snapshots of gpt-4o. The page /api/docs/models/gpt-4o lists them as
    // available and gives only the gpt-4o prices. The pricing page does
    // not have them in its text token tables.
    chat("gpt-4o-2024-11-20", 2.50, 10.00, System, None, false),
    chat("gpt-4o-2024-08-06", 2.50, 10.00, System, None, false),
    retiring_chat("gpt-4o-2024-05-13", 5.00, 15.00, "2026-10-23", "gpt-5.6-sol"),
    chat("gpt-4o-mini", 0.15, 0.60, System, None, true),
    // Default snapshot of gpt-4o-mini, as the page
    // /api/docs/models/gpt-4o-mini says. That page gives only the
    // gpt-4o-mini prices.
    chat("gpt-4o-mini-2024-07-18", 0.15, 0.60, System, None, false),
    retiring_chat("gpt-4-turbo", 10.00, 30.00, "2026-10-23", "gpt-5.6-sol"),
    retiring_chat("gpt-4-turbo-2024-04-09", 10.00, 30.00, "2026-10-23", "gpt-5.6-sol"),
    retiring_chat("gpt-4", 30.00, 60.00, "2026-10-23", "gpt-5.6-sol"),
    retiring_chat("gpt-4-0613", 30.00, 60.00, "2026-10-23", "gpt-5.6-sol"),
    retiring_chat("gpt-3.5-turbo", 0.50, 1.50, "2026-10-23", "gpt-5.6-terra"),
    retiring_chat("gpt-3.5-turbo-0125", 0.50, 1.50, "2026-10-23", "gpt-5.6-terra"),
    retiring_chat("gpt-3.5-turbo-1106", 1.00, 2.00, "2026-09-28", "gpt-5.6-terra"),
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
        shutdown: None,
    }
}

/// Transcription model for a new config. Its price also applies to a model
/// that is not in `TRANSCRIPTION_MODELS`.
pub const DEFAULT_TRANSCRIPTION_MODEL: TranscriptionModel =
    transcription("gpt-transcribe", 0.0045, true);

/// Row of `TRANSCRIPTION_MODELS` for a model that shuts down on 2027-02-26.
const fn retiring_transcription(name: &'static str, price_per_minute: f64) -> TranscriptionModel {
    TranscriptionModel {
        shutdown: Some(Shutdown {
            date: "2027-02-26",
            replacement: DEFAULT_TRANSCRIPTION_MODEL.name,
        }),
        ..transcription(name, price_per_minute, false)
    }
}

pub const TRANSCRIPTION_MODELS: &[TranscriptionModel] = &[
    DEFAULT_TRANSCRIPTION_MODEL,
    retiring_transcription("gpt-4o-mini-transcribe", 0.003),
    retiring_transcription("gpt-4o-transcribe", 0.006),
    retiring_transcription("whisper-1", 0.006),
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

/// A warning for each model that is scheduled to shut down, for the
/// activity log. The warning gives the date and does not compare it with
/// today, so it also shows after the date, when requests fail.
pub fn shutdown_warnings(model: &str, transcription_model: &str) -> Vec<String> {
    let chat = chat_model(model).and_then(|known| known.shutdown.as_ref());
    let transcription =
        self::transcription_model(transcription_model).and_then(|known| known.shutdown.as_ref());
    [
        ("Model", model, chat),
        ("Transcription model", transcription_model, transcription),
    ]
    .into_iter()
    .filter_map(|(kind, name, shutdown)| {
        shutdown.map(|shutdown| {
            format!(
                "{} '{}' has an OpenAI shutdown date of {}. Use {} instead.",
                kind, name, shutdown.date, shutdown.replacement
            )
        })
    })
    .collect()
}
