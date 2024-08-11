use vrchat_osc_llm::config::Config;
use vrchat_osc_llm::audio_recording::start_audio_recording;
use vrchat_osc_llm::audio_processing::process_audio;
use vrchat_osc_llm::rate_limiter::RateLimiter;
use vrchat_osc_llm::price_estimator::PriceEstimator;
use vrchat_osc_llm::typing_indicator::TypingIndicator;
use vrchat_osc_llm::types::AudioEvent;

use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {}", panic_info);
        println!("\nPress Enter to exit...");
        io::stdout().flush().unwrap();
        let _ = io::stdin().read_line(&mut String::new());
    }));

    let result = run_main().await;

    println!("\nPress Enter to exit...");
    io::stdout().flush().unwrap();
    let _ = io::stdin().read_line(&mut String::new());

    result
}

async fn run_main() -> Result<(), Box<dyn Error>> {
    let config: Arc<Config> = Arc::new(toml::from_str(&fs::read_to_string("config.toml")?)?);

    let socket_address = format!("{}:{}", config.osc.address, config.osc.input_port);
    let socket = Arc::new(UdpSocket::bind(&socket_address).await?);

    println!("Starting continuous audio recording...");
    println!("Translating to: {}", config.translation.target_language);
    println!(
        "Rate limit: {} requests per minute",
        config.rate_limit.requests_per_minute
    );

    let (tx, mut rx) = mpsc::channel::<AudioEvent>(100);

    let typing_indicator = TypingIndicator::new(Arc::clone(&socket), Arc::clone(&config));

    let config_clone = Arc::clone(&config);
    std::thread::spawn(move || {
        if let Err(e) = start_audio_recording(&config_clone, tx) {
            eprintln!("Error starting audio recording: {}", e);
        }
    });

    let mut rate_limiter = RateLimiter::new(config.rate_limit.requests_per_minute);
    let mut price_estimator = PriceEstimator::new(&config.openai.model);
    println!("Loaded total cost: ${:.4}", price_estimator.total_cost);

    while let Some(event) = rx.recv().await {
        match event {
            AudioEvent::StartRecording => {
                typing_indicator.start_typing().await;
            }
            AudioEvent::StopRecording => {
                typing_indicator.stop_typing().await;
            }
            AudioEvent::AudioData(audio_data) => {
                if let Err(e) = process_audio(
                    audio_data,
                    &config,
                    &socket,
                    &mut rate_limiter,
                    &typing_indicator,
                    &mut price_estimator,
                )
                .await
                {
                    eprintln!("Error processing audio: {}", e);
                }
            }
        }
    }

    Ok(())
}