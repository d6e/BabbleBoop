use crate::app_state::{blocking_task_failure, FailureLog, Logger};
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
    /// The save runs on the blocking pool, so a slow disk does not stop
    /// the async task of the processing loop.
    pub async fn add_cost(&mut self, cost: f64, logger: &Logger) {
        self.total_cost += cost;
        let (cost_file, total_cost) = (self.cost_file.clone(), self.total_cost);
        let saved =
            tokio::task::spawn_blocking(move || save_total_cost(&cost_file, total_cost)).await;
        let saved = match saved {
            Ok(saved) => saved.map_err(|e| e.to_string()),
            Err(e) => Err(blocking_task_failure(&e).to_string()),
        };
        match saved {
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

    /// Blocks on the file system.
    fn load_total_cost(cost_file: &Path) -> Result<f64, Box<dyn Error>> {
        let content = fs::read_to_string(cost_file)?;
        Ok(content.trim().parse()?)
    }
}

/// Write `total_cost` to `cost_file`. Blocks on the file system.
fn save_total_cost(cost_file: &Path, total_cost: f64) -> std::io::Result<()> {
    fs::write(cost_file, total_cost.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::LogLevel;
    use crate::test_support::LogCapture;

    #[tokio::test]
    async fn test_a_failed_cost_save_is_logged_once_per_distinct_error() {
        let dir = std::env::temp_dir().join(format!("babble_boop_cost_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        let file = dir.join("total_cost.txt");
        let mut log = LogCapture::new();
        let logger = log.logger();
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
        estimator.add_cost(0.5, &logger).await;
        let not_found = saves_fail_with(log.entries());
        estimator.add_cost(0.5, &logger).await;
        estimator.add_cost(0.5, &logger).await;
        assert_eq!(log.entries(), []);

        // A different failure: the file is a directory
        fs::create_dir_all(&file).unwrap();
        estimator.add_cost(0.5, &logger).await;
        let is_a_directory = saves_fail_with(log.entries());
        assert_ne!(is_a_directory, not_found);
        estimator.add_cost(0.5, &logger).await;
        assert_eq!(log.entries(), []);

        // The save works again
        fs::remove_dir(&file).unwrap();
        estimator.add_cost(0.5, &logger).await;
        let entries = log.entries();
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert_eq!(entries[0].0, LogLevel::Info, "{:?}", entries);
        assert_eq!(fs::read_to_string(&file).unwrap(), "3");
        estimator.add_cost(0.5, &logger).await;
        assert_eq!(log.entries(), []);

        // The first failure again, after a save that worked
        fs::remove_dir_all(&dir).unwrap();
        estimator.add_cost(0.5, &logger).await;
        assert_eq!(saves_fail_with(log.entries()), not_found);
    }

    // ===========================================================================
    // Regression test: Model pricing now covers common models
    // ===========================================================================

    #[test]
    fn test_older_models_keep_pricing() {
        // Models that existing config files can name, until they shut down.
        for (model, transcription_model) in [
            ("gpt-4o", "whisper-1"),
            ("gpt-4o-mini", "gpt-4o-transcribe"),
            ("gpt-4-turbo", "gpt-4o-mini-transcribe"),
            ("gpt-4", "whisper-1"),
            ("gpt-3.5-turbo", "whisper-1"),
        ] {
            assert_eq!(
                PriceEstimator::unknown_pricing(model, transcription_model),
                Vec::<String>::new()
            );
        }
    }

    #[test]
    fn test_current_models_have_pricing() {
        for (model, transcription_model) in [
            ("gpt-6-luna", "gpt-transcribe"),
            ("gpt-6-sol", "gpt-transcribe"),
            ("gpt-5.4-nano", "gpt-transcribe"),
            ("gpt-5.4-mini", "gpt-transcribe"),
            ("gpt-4.1-mini", "gpt-transcribe"),
        ] {
            assert_eq!(
                PriceEstimator::unknown_pricing(model, transcription_model),
                Vec::<String>::new()
            );
        }
    }

    /// What a config that names `model` as the translation model gets: the
    /// translation cost, the price and shutdown warnings, and the
    /// instructions role and reasoning effort of the request. Warnings name
    /// the model as `shown_as`, so that a snapshot and its model compare
    /// equal.
    fn chat_model_behaviour(
        model: &str,
        shown_as: &str,
    ) -> (
        String,
        f64,
        Vec<String>,
        Vec<String>,
        String,
        Option<String>,
    ) {
        use crate::test_support::{price_estimator, TOKENS};
        use crate::translation::ChatGptRequest;

        let rename = |warnings: Vec<String>| -> Vec<String> {
            warnings
                .into_iter()
                .map(|warning| warning.replace(&format!("'{}'", model), &format!("'{}'", shown_as)))
                .collect()
        };
        let body =
            serde_json::to_value(ChatGptRequest::translation(model, "Japanese", "Hello")).unwrap();
        (
            shown_as.to_string(),
            price_estimator(model, "gpt-transcribe").estimate_translation_cost(TOKENS),
            rename(PriceEstimator::unknown_pricing(model, "gpt-transcribe")),
            rename(crate::models::shutdown_warnings(model, "gpt-transcribe")),
            body["messages"][0]["role"].as_str().unwrap().to_string(),
            body.get("reasoning_effort")
                .map(|effort| effort.as_str().unwrap().to_string()),
        )
    }

    /// Like `chat_model_behaviour`, for the transcription model.
    fn transcription_model_behaviour(
        model: &str,
        shown_as: &str,
    ) -> (String, f64, Vec<String>, Vec<String>) {
        use crate::test_support::price_estimator;

        let rename = |warnings: Vec<String>| -> Vec<String> {
            warnings
                .into_iter()
                .map(|warning| warning.replace(&format!("'{}'", model), &format!("'{}'", shown_as)))
                .collect()
        };
        (
            shown_as.to_string(),
            price_estimator("gpt-6-luna", model)
                .estimate_transcription_cost(std::time::Duration::from_secs(60)),
            rename(PriceEstimator::unknown_pricing("gpt-6-luna", model)),
            rename(crate::models::shutdown_warnings("gpt-6-luna", model)),
        )
    }

    #[test]
    fn test_dated_snapshots_behave_as_their_model() {
        // Each model page lists these snapshots under "Snapshots". The
        // pricing and deprecations pages give no other price or shutdown
        // date for them than for their model. gpt-5.6 is an alias: the
        // gpt-5.6-sol page says it routes requests to GPT-5.6 Sol.
        let chat = [
            ("gpt-4.1-nano-2025-04-14", "gpt-4.1-nano"),
            ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini"),
            ("gpt-4.1-2025-04-14", "gpt-4.1"),
            ("gpt-5.4-mini-2026-03-17", "gpt-5.4-mini"),
            ("gpt-5.4-nano-2026-03-17", "gpt-5.4-nano"),
            ("gpt-5.6", "gpt-5.6-sol"),
            ("gpt-4o-2024-11-20", "gpt-4o"),
            ("gpt-4o-2024-08-06", "gpt-4o"),
            ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),
            ("gpt-4-turbo-2024-04-09", "gpt-4-turbo"),
            ("gpt-4-0613", "gpt-4"),
            ("gpt-3.5-turbo-0125", "gpt-3.5-turbo"),
        ];
        let chat_snapshots: Vec<_> = chat
            .iter()
            .map(|(snapshot, _)| chat_model_behaviour(snapshot, snapshot))
            .collect();
        let chat_models: Vec<_> = chat
            .iter()
            .map(|(snapshot, model)| chat_model_behaviour(model, snapshot))
            .collect();

        // The deprecations page gives the shutdown date of
        // gpt-4o-mini-transcribe and does not list this snapshot on its own.
        let transcription = [(
            "gpt-4o-mini-transcribe-2025-12-15",
            "gpt-4o-mini-transcribe",
        )];
        let transcription_snapshots: Vec<_> = transcription
            .iter()
            .map(|(snapshot, _)| transcription_model_behaviour(snapshot, snapshot))
            .collect();
        let transcription_models: Vec<_> = transcription
            .iter()
            .map(|(snapshot, model)| transcription_model_behaviour(model, snapshot))
            .collect();

        assert_eq!(
            (chat_snapshots, transcription_snapshots),
            (chat_models, transcription_models)
        );
    }

    #[test]
    fn test_dated_snapshots_with_their_own_shutdown_date_are_warned() {
        // The deprecations page gives these snapshots a shutdown date of
        // their own. Their model pages give no other price than for their
        // model.
        let gpt_4_0314 = chat_model_behaviour("gpt-4-0314", "gpt-4-0314");
        let gpt_4 = chat_model_behaviour("gpt-4", "gpt-4-0314");
        assert_eq!(
            (gpt_4_0314.1, &gpt_4_0314.2),
            (gpt_4.1, &Vec::<String>::new())
        );
        assert_eq!(gpt_4_0314.3.len(), 1, "{:?}", gpt_4_0314.3);
        for text in ["'gpt-4-0314'", "2026-03-26", "Use gpt-4.1 instead"] {
            assert!(gpt_4_0314.3[0].contains(text), "{:?}", gpt_4_0314.3);
        }

        let march = transcription_model_behaviour(
            "gpt-4o-mini-transcribe-2025-03-20",
            "gpt-4o-mini-transcribe-2025-03-20",
        );
        let model = transcription_model_behaviour(
            "gpt-4o-mini-transcribe",
            "gpt-4o-mini-transcribe-2025-03-20",
        );
        assert_eq!((march.1, &march.2), (model.1, &Vec::<String>::new()));
        assert_eq!(march.3.len(), 1, "{:?}", march.3);
        for text in [
            "'gpt-4o-mini-transcribe-2025-03-20'",
            "2027-01-20",
            "Use gpt-4o-mini-transcribe-2025-12-15 instead",
        ] {
            assert!(march.3[0].contains(text), "{:?}", march.3);
        }

        // The gpt-3.5-turbo page lists gpt-3.5-turbo-instruct as a snapshot.
        // The pricing page gives it a price of its own.
        let instruct = chat_model_behaviour("gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct");
        assert_eq!(instruct.2, Vec::<String>::new());
        assert_eq!(instruct.3.len(), 1, "{:?}", instruct.3);
        for text in [
            "'gpt-3.5-turbo-instruct'",
            "2026-09-28",
            "Use gpt-5.6-terra instead",
        ] {
            assert!(instruct.3[0].contains(text), "{:?}", instruct.3);
        }
    }

    #[test]
    fn test_unknown_model_uses_default_model_pricing() {
        use crate::config::Config;
        use crate::test_support::{price_estimator, TOKENS};
        use std::time::Duration;

        let defaults = Config::default().openai;
        let unknown = price_estimator("unknown-model-xyz", "unknown-transcriber");
        let default = price_estimator(&defaults.model, &defaults.transcription_model);
        assert_eq!(
            unknown.estimate_translation_cost(TOKENS),
            default.estimate_translation_cost(TOKENS)
        );
        let minute = Duration::from_secs(60);
        assert_eq!(
            unknown.estimate_transcription_cost(minute),
            default.estimate_transcription_cost(minute)
        );

        // The warnings name the models whose prices the estimate uses.
        let warnings = PriceEstimator::unknown_pricing("unknown-model-xyz", "unknown-transcriber");
        assert_eq!(warnings.len(), 2, "{:?}", warnings);
        assert!(warnings[0].contains(&format!("{} prices", defaults.model)));
        assert!(warnings[1].contains(&format!("{} prices", defaults.transcription_model)));
    }
}
