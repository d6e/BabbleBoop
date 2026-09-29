use crate::app_state::{FailureLog, Logger};
use crate::models::{self, DEFAULT_CHAT_MODEL, DEFAULT_TRANSCRIPTION_MODEL};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

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
    cost_file: PathBuf,
    /// The cost is saved after each transcription and each translation,
    /// so up to twice per utterance; a save that fails keeps failing until
    /// the user fixes the cause.
    save_failure: FailureLog,
}

impl PriceEstimator {
    /// An estimator that loads and saves the total cost of all sessions
    /// in `cost_file`.
    pub fn new(cost_file: PathBuf, model: &str, transcription_model: &str) -> Self {
        let mut estimator = PriceEstimator {
            transcription_price_per_minute: 0.0,
            gpt_input_price_per_million_tokens: 0.0,
            gpt_output_price_per_million_tokens: 0.0,
            total_cost: Self::load_total_cost(&cost_file).unwrap_or(0.0),
            cost_file,
            save_failure: FailureLog::default(),
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

    /// Add to the total cost and save it. A failed save goes to the
    /// activity log, once until the save works again or fails differently.
    pub fn add_cost(&mut self, cost: f64, logger: &Logger) {
        self.total_cost += cost;
        match fs::write(&self.cost_file, self.total_cost.to_string()) {
            Ok(()) => {
                if self.save_failure.succeeded() {
                    logger.info(format!(
                        "Saved the total cost to {} again",
                        self.cost_file.display()
                    ));
                }
            }
            Err(e) => self.save_failure.failed(
                logger,
                format!(
                    "Cannot save the total cost to {}: {}",
                    self.cost_file.display(),
                    e
                ),
            ),
        }
    }

    fn load_total_cost(cost_file: &Path) -> Result<f64, Box<dyn Error>> {
        let content = fs::read_to_string(cost_file)?;
        Ok(content.trim().parse()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{LogEntry, LogLevel};
    use tokio::sync::mpsc;

    /// Level and text of the entries in the activity log since the last call.
    fn new_entries(log_rx: &mut mpsc::Receiver<LogEntry>) -> Vec<(LogLevel, String)> {
        std::iter::from_fn(|| log_rx.try_recv().ok())
            .map(|entry| (entry.level, entry.message))
            .collect()
    }

    #[test]
    fn test_a_failed_cost_save_is_logged_once_per_distinct_error() {
        let dir = std::env::temp_dir().join(format!("babble_boop_cost_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        let file = dir.join("total_cost.txt");
        let (log_tx, mut log_rx) = mpsc::channel(10);
        let logger = Logger::new(log_tx, Default::default());
        let mut estimator = PriceEstimator::new(file.clone(), "gpt-6-luna", "gpt-transcribe");
        let saves_fail_with = |entries: Vec<(LogLevel, String)>| {
            assert_eq!(entries.len(), 1, "{:?}", entries);
            assert_eq!(entries[0].0, LogLevel::Error, "{:?}", entries);
            assert!(
                entries[0].1.contains(&file.display().to_string()),
                "{:?}",
                entries
            );
            entries[0].1.clone()
        };

        // The directory does not exist: every save fails the same way
        estimator.add_cost(0.5, &logger);
        let not_found = saves_fail_with(new_entries(&mut log_rx));
        estimator.add_cost(0.5, &logger);
        estimator.add_cost(0.5, &logger);
        assert_eq!(new_entries(&mut log_rx), []);

        // A different failure: the file is a directory
        fs::create_dir_all(&file).unwrap();
        estimator.add_cost(0.5, &logger);
        let is_a_directory = saves_fail_with(new_entries(&mut log_rx));
        assert_ne!(is_a_directory, not_found);
        estimator.add_cost(0.5, &logger);
        assert_eq!(new_entries(&mut log_rx), []);

        // The save works again
        fs::remove_dir(&file).unwrap();
        estimator.add_cost(0.5, &logger);
        let entries = new_entries(&mut log_rx);
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert_eq!(entries[0].0, LogLevel::Info, "{:?}", entries);
        assert_eq!(fs::read_to_string(&file).unwrap(), "3");
        estimator.add_cost(0.5, &logger);
        assert_eq!(new_entries(&mut log_rx), []);

        // The first failure again, after a save that worked
        fs::remove_dir_all(&dir).unwrap();
        estimator.add_cost(0.5, &logger);
        assert_eq!(saves_fail_with(new_entries(&mut log_rx)), not_found);
    }
}
