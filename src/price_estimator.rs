use std::error::Error;
use std::fs;
use std::time::Duration;

pub struct PriceEstimator {
    whisper_price_per_minute: f64,
    gpt_input_price_per_million_tokens: f64,
    gpt_output_price_per_million_tokens: f64,
    pub total_cost: f64,
}

impl PriceEstimator {
    pub fn new(model: &str) -> Self {
        let (input_price, output_price, known) = Self::get_model_pricing(model);

        if !known {
            eprintln!(
                "Warning: Unknown model '{}' for pricing. Cost estimates will be inaccurate.",
                model
            );
        }

        let total_cost = Self::load_total_cost().unwrap_or(0.0);

        PriceEstimator {
            whisper_price_per_minute: 0.006,
            gpt_input_price_per_million_tokens: input_price,
            gpt_output_price_per_million_tokens: output_price,
            total_cost,
        }
    }

    fn get_model_pricing(model: &str) -> (f64, f64, bool) {
        // Prices per million tokens (input, output) as of late 2024
        // See: https://openai.com/api/pricing/
        match model {
            // GPT-4o models
            "gpt-4o" | "gpt-4o-2024-11-20" | "gpt-4o-2024-08-06" => (2.50, 10.00, true),
            "gpt-4o-2024-05-13" => (5.00, 15.00, true),

            // GPT-4o mini models
            "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => (0.15, 0.60, true),

            // GPT-4 Turbo models
            "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" | "gpt-4-turbo-preview"
            | "gpt-4-0125-preview" | "gpt-4-1106-preview" => (10.00, 30.00, true),

            // GPT-4 models
            "gpt-4" | "gpt-4-0613" => (30.00, 60.00, true),
            "gpt-4-32k" | "gpt-4-32k-0613" => (60.00, 120.00, true),

            // GPT-3.5 Turbo models
            "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-1106" => (0.50, 1.50, true),
            "gpt-3.5-turbo-instruct" => (1.50, 2.00, true),

            // Unknown model - use gpt-4o-mini pricing as conservative default
            _ => (0.15, 0.60, false),
        }
    }

    pub fn estimate_transcription_cost(&self, duration: Duration) -> f64 {
        let minutes = duration.as_secs_f64() / 60.0;
        minutes * self.whisper_price_per_minute
    }

    pub fn estimate_translation_cost(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        let input_cost =
            (input_tokens as f64 / 1_000_000.0) * self.gpt_input_price_per_million_tokens;
        let output_cost =
            (output_tokens as f64 / 1_000_000.0) * self.gpt_output_price_per_million_tokens;
        input_cost + output_cost
    }

    pub fn add_cost(&mut self, cost: f64) {
        self.total_cost += cost;
        self.save_total_cost();
    }

    fn load_total_cost() -> Result<f64, Box<dyn Error>> {
        let content = fs::read_to_string("total_cost.txt")?;
        Ok(content.trim().parse()?)
    }

    fn save_total_cost(&self) {
        if let Err(e) = fs::write("total_cost.txt", self.total_cost.to_string()) {
            eprintln!("Failed to save total cost: {}", e);
        }
    }
}
