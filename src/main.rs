use babble_boop::audio_processing::process_audio;
use babble_boop::audio_recording::start_audio_recording;
use babble_boop::config::Config;
use babble_boop::price_estimator::PriceEstimator;
use babble_boop::rate_limiter::RateLimiter;
use babble_boop::recording_manager::RecordingManager;
use babble_boop::types::AudioEvent;
use babble_boop::typing_indicator::TypingIndicator;

use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Set a panic hook to handle panics and prevent the program from closing immediately
    std::panic::set_hook(Box::new(|panic_info| {
        eprintln!("Panic occurred: {}", panic_info);

        println!("");
        println!("Press Enter to exit...");
        io::stdout().flush().unwrap();
        let _ = io::stdin().read_line(&mut String::new());
    }));

    let result = run_main().await;

    println!("");
    println!("Press Enter to exit...");
    io::stdout().flush().unwrap();
    let _ = io::stdin().read_line(&mut String::new());

    result
}

async fn run_main() -> Result<(), Box<dyn Error>> {

    let config_path = "config.toml";
    let config_data = match fs::read_to_string(config_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading config file: {}", e);
            eprintln!("Please ensure that the 'config.toml' file exists in the same directory as the executable.");
            eprintln!("You can refer to 'config.toml.example' for an example configuration.");
            return Err(Box::new(e));
        }
    };

    let config: Config = toml::from_str(&config_data)?;
    let config = Arc::new(config);

    let socket_address = format!("{}:{}", config.osc.address, config.osc.input_port);
    let socket = Arc::new(UdpSocket::bind(&socket_address).await?);

    println!("Starting continuous audio recording...");
    println!("Translating to: {}", config.translation.target_language);
    println!(
        "Rate limit: {} requests per minute",
        config.rate_limit.requests_per_minute
    );
    println!("Debug mode: {}", config.debug);

    let (tx, mut rx) = mpsc::channel::<AudioEvent>(100);

    let typing_indicator = TypingIndicator::new(Arc::clone(&socket), Arc::clone(&config));

    // Start the audio recording in a separate thread
    let config_clone = Arc::clone(&config);
    std::thread::spawn(move || {
        if let Err(e) = start_audio_recording(&config_clone, tx) {
            eprintln!("Error starting audio recording: {}", e);
        }
    });

    let mut rate_limiter = RateLimiter::new(config.rate_limit.requests_per_minute);
    let mut price_estimator = PriceEstimator::new(&config.openai.model);
    println!("Loaded total cost: ${:.4}", price_estimator.total_cost);

    let recording_manager = if config.debug {
        Some(RecordingManager::new(PathBuf::from("recordings"), 10))
    } else {
        None
    };

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
                    recording_manager.as_ref(),
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
