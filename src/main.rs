use babble_boop::app_state::{AppCommand, AppState};
use babble_boop::audio_processing::process_audio;
use babble_boop::audio_recording::start_audio_recording;
use babble_boop::config::Config;
use babble_boop::gui::run_gui;
use babble_boop::price_estimator::PriceEstimator;
use babble_boop::rate_limiter::RateLimiter;
use babble_boop::recording_manager::RecordingManager;
use babble_boop::types::AudioEvent;
use babble_boop::typing_indicator::TypingIndicator;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

const CONFIG_PATH: &str = "config.toml";

fn main() {
    // Load config
    let config = match Config::load(CONFIG_PATH) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading config file: {}", e);
            eprintln!("Please ensure that the 'config.toml' file exists in the same directory as the executable.");
            eprintln!("You can refer to 'config.toml.example' for an example configuration.");
            std::process::exit(1);
        }
    };

    // Create command channel for GUI -> processing communication
    let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(32);

    // Create shared app state
    let app_state = Arc::new(AppState::new(config, cmd_tx));
    let app_state_clone = Arc::clone(&app_state);

    // Spawn background thread with tokio runtime for audio processing
    let processing_handle = std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        rt.block_on(async {
            if let Err(e) = run_processing_loop(app_state_clone, cmd_rx).await {
                eprintln!("Processing error: {}", e);
            }
        });
    });

    // Run GUI on main thread
    if let Err(e) = run_gui(app_state) {
        eprintln!("GUI error: {}", e);
    }

    // Wait for processing thread to finish
    let _ = processing_handle.join();
}

async fn run_processing_loop(
    app_state: Arc<AppState>,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Read initial config
    let config = app_state.config.read().unwrap().clone();

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

    // Start the audio recording in a separate thread
    let config_for_audio = app_state.config.read().unwrap().clone();
    let shutdown_signal = Arc::clone(&app_state.shutdown);
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::spawn(move || {
        match start_audio_recording(&config_for_audio, tx) {
            Ok(stream) => {
                let _ = init_tx.send(Ok(()));
                let _stream = stream;
                // Check shutdown signal periodically instead of parking forever
                while !shutdown_signal.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                println!("Audio recording thread shutting down...");
                // Stream is dropped here, stopping audio capture
            }
            Err(e) => {
                let _ = init_tx.send(Err(e.to_string()));
            }
        }
    });

    // Wait for stream initialization
    init_rx
        .recv()
        .map_err(|_| "Audio recording thread failed to start")?
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;

    let mut rate_limiter = RateLimiter::new(config.rate_limit.requests_per_minute);
    let mut price_estimator = PriceEstimator::new(&config.openai.model);
    println!("Loaded total cost: ${:.4}", price_estimator.total_cost);

    let mut recording_manager = if config.debug {
        Some(RecordingManager::new(PathBuf::from("recordings"), 10))
    } else {
        None
    };

    let typing_indicator = TypingIndicator::new(Arc::clone(&socket), Arc::clone(&app_state.config));

    loop {
        tokio::select! {
            // Prioritize command channel to handle Quit promptly
            biased;

            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(AppCommand::SetEnabled(enabled)) => {
                        println!("Translation {}", if enabled { "enabled" } else { "disabled" });
                    }
                    Some(AppCommand::UpdateConfig(new_config)) => {
                        println!("Config updated");
                        // Update rate limiter if needed
                        rate_limiter = RateLimiter::new(new_config.rate_limit.requests_per_minute);
                        // Update recording manager if debug changed
                        recording_manager = if new_config.debug {
                            Some(RecordingManager::new(PathBuf::from("recordings"), 10))
                        } else {
                            None
                        };
                    }
                    Some(AppCommand::Quit) | None => {
                        // Quit command received or channel closed
                        println!("Shutting down...");
                        break;
                    }
                }
            }
            Some(event) = rx.recv() => {
                // Check if enabled
                if !app_state.enabled.load(Ordering::Relaxed) {
                    // Still handle typing indicator but skip processing
                    match &event {
                        AudioEvent::StartRecording | AudioEvent::StopRecording => {
                            // Skip typing indicator when disabled
                        }
                        AudioEvent::AudioData(_) => {
                            // Skip processing when disabled
                            continue;
                        }
                    }
                    continue;
                }

                match event {
                    AudioEvent::StartRecording => {
                        typing_indicator.start_typing().await;
                    }
                    AudioEvent::StopRecording => {
                        typing_indicator.stop_typing().await;
                    }
                    AudioEvent::AudioData(audio_data) => {
                        // Read current config for processing
                        let current_config = app_state.config.read().unwrap().clone();
                        if let Err(e) = process_audio(
                            audio_data,
                            &current_config,
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
        }
    }

    Ok(())
}
