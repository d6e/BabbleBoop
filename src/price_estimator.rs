use crate::models::{self, DEFAULT_CHAT_MODEL, DEFAULT_TRANSCRIPTION_MODEL};
use std::error::Error;
use std::fs;
use std::time::Duration;

/// File to persist total API cost across sessions
const TOTAL_COST_FILE: &str = "total_cost.txt";

/// Tokens of one translation request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenCounts {
    pub input: usize,
    /// Includes the reasoning tokens, which are billed as output tokens.
    pub output: usize,
}

pub struct PriceEstimator {
    transcription_price_per_minute: f64,
    gpt_input_price_per_million_tokens: f64,
    gpt_output_price_per_million_tokens: f64,
    pub total_cost: f64,
}

impl PriceEstimator {
    pub fn new(model: &str, transcription_model: &str) -> Self {
        let mut estimator = PriceEstimator {
            transcription_price_per_minute: 0.0,
            gpt_input_price_per_million_tokens: 0.0,
            gpt_output_price_per_million_tokens: 0.0,
            total_cost: Self::load_total_cost().unwrap_or(0.0),
        };
        estimator.set_models(model, transcription_model);
        estimator
    }

    /// Use the prices of these models for new estimates. The total cost is
    /// kept. See `unknown_pricing` for models without a known price.
    pub fn set_models(&mut self, model: &str, transcription_model: &str) {
        let chat = models::chat_model(model).unwrap_or(&DEFAULT_CHAT_MODEL);
        let transcription = models::transcription_model(transcription_model)
            .unwrap_or(&DEFAULT_TRANSCRIPTION_MODEL);

        self.transcription_price_per_minute = transcription.price_per_minute;
        self.gpt_input_price_per_million_tokens = chat.input_price;
        self.gpt_output_price_per_million_tokens = chat.output_price;
    }

    /// A warning for each model that has no known price, for the activity
    /// log. Cost estimates use the default model prices for these models.
    pub fn unknown_pricing(model: &str, transcription_model: &str) -> Vec<String> {
        let mut warnings = Vec::new();
        if models::chat_model(model).is_none() {
            warnings.push(format!(
                "No price known for model '{}'. The cost uses {} prices.",
                model, DEFAULT_CHAT_MODEL.name
            ));
        }
        if models::transcription_model(transcription_model).is_none() {
            warnings.push(format!(
                "No price known for transcription model '{}'. The cost uses {} prices.",
                transcription_model, DEFAULT_TRANSCRIPTION_MODEL.name
            ));
        }
        warnings
    }

    pub fn estimate_transcription_cost(&self, duration: Duration) -> f64 {
        let minutes = duration.as_secs_f64() / 60.0;
        minutes * self.transcription_price_per_minute
    }

    pub fn estimate_translation_cost(&self, tokens: TokenCounts) -> f64 {
        let input_cost =
            (tokens.input as f64 / 1_000_000.0) * self.gpt_input_price_per_million_tokens;
        let output_cost =
            (tokens.output as f64 / 1_000_000.0) * self.gpt_output_price_per_million_tokens;
        input_cost + output_cost
    }

    pub fn add_cost(&mut self, cost: f64) {
        self.total_cost += cost;
        self.save_total_cost();
    }

    fn load_total_cost() -> Result<f64, Box<dyn Error>> {
        let content = fs::read_to_string(TOTAL_COST_FILE)?;
        Ok(content.trim().parse()?)
    }

    fn save_total_cost(&self) {
        if let Err(e) = fs::write(TOTAL_COST_FILE, self.total_cost.to_string()) {
            eprintln!("Failed to save total cost: {}", e);
        }
    }
}
