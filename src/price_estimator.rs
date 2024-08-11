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
        let (input_price, output_price) = match model {
            "gpt-4o" => (5.00, 15.00),
            "gpt-4o-2024-08-06" => (2.50, 10.00),
            "gpt-4o-2024-05-13" => (5.00, 15.00),
            "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => (0.150, 0.600),
            _ => (0.0, 0.0),
        };

        let total_cost = Self::load_total_cost().unwrap_or(0.0);

        PriceEstimator {
            whisper_price_per_minute: 0.006,
            gpt_input_price_per_million_tokens: input_price,
            gpt_output_price_per_million_tokens: output_price,
            total_cost,
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
