use babble_boop::app_state::{AppCommand, AppState, LogEntry};
use babble_boop::audio_playback::play_wav_buffer;
use babble_boop::audio_processing::process_audio;
use babble_boop::audio_recording::start_audio_recording;
use babble_boop::config::{Config, CONFIG_PATH};
use babble_boop::gui::{run_error_dialog, run_gui};
use babble_boop::price_estimator::PriceEstimator;
use babble_boop::rate_limiter::RateLimiter;
use babble_boop::recording_manager::RecordingManager;
use babble_boop::types::AudioEvent;
use babble_boop::typing_indicator::TypingIndicator;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

fn main() {
    // Load or create config
    let (config, first_run) = match Config::load_or_create(CONFIG_PATH) {
        Ok(result) => result,
        Err(e) => {
            let message = format!(
                "Failed to load or create config file: {}\n\n\
                Please check file permissions and try again.",
                e
            );
            eprintln!("{}", message);
            let _ = run_error_dialog("Configuration Error", &message);
            std::process::exit(1);
        }
    };

    // Create command channel for GUI -> processing communication
    let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(32);

    // Create log channel for processing -> GUI communication
    let (log_tx, log_rx) = mpsc::channel::<LogEntry>(100);

    // Create shared app state
    let app_state = Arc::new(AppState::new(config, cmd_tx, log_tx));
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
    if let Err(e) = run_gui(app_state, log_rx, first_run) {
        let message = format!("Failed to start the application window: {}", e);
        eprintln!("{}", message);
        let _ = run_error_dialog("GUI Error", &message);
    }

    // Wait for processing thread to finish
    let _ = processing_handle.join();
}

async fn run_processing_loop(
    app_state: Arc<AppState>,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Read initial config
    let config = app_state
        .config
        .read()
        .expect("Config lock poisoned")
        .clone();

    let socket_address = format!("{}:{}", config.osc.address, config.osc.input_port);
    let socket = Arc::new(UdpSocket::bind(&socket_address).await?);

    println!("Starting continuous audio recording...");
    println!("Translating to: {}", config.translation.target_language);
    println!(
        "Rate limit: {} requests per minute",
        config.rate_limit.requests_per_minute
    );
    println!("Keep audio files: {}", config.keep_audio_files);

    let (tx, mut rx) = mpsc::channel::<AudioEvent>(100);

    // Start the audio recording in a separate thread
    let audio_params = Arc::clone(&app_state.audio_params);
    let audio_level = Arc::clone(&app_state.current_audio_level);
    let shutdown_signal = Arc::clone(&app_state.shutdown);
    let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    std::thread::spawn(move || {
        match start_audio_recording(audio_params, audio_level, tx) {
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
    let mut price_estimator =
        PriceEstimator::new(&config.openai.model, &config.openai.transcription_model);
    println!("Loaded total cost: ${:.4}", price_estimator.total_cost);

    let mut recording_manager = if config.keep_audio_files {
        Some(RecordingManager::new(
            PathBuf::from("recordings"),
            config.max_audio_files,
        ))
    } else {
        None
    };

    let typing_indicator = TypingIndicator::new(Arc::clone(&socket), Arc::clone(&app_state.config));

    // Test recording state
    let mut test_recording_buffer: Vec<u8> = Vec::new();
    let mut test_recording_start: Option<Instant> = None;
    let test_recording_duration = Duration::from_secs(3);
    // Keep playback stream alive until playback completes
    let mut _playback_stream: Option<cpal::Stream> = None;
    let playback_active = Arc::new(AtomicBool::new(false));

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
                        // Update hot-reloadable audio params
                        app_state.audio_params.update(&new_config.audio);
                        // Update rate limiter if needed
                        rate_limiter = RateLimiter::new(new_config.rate_limit.requests_per_minute);
                        // Update recording manager if keep_audio_files changed
                        recording_manager = if new_config.keep_audio_files {
                            Some(RecordingManager::new(PathBuf::from("recordings"), new_config.max_audio_files))
                        } else {
                            None
                        };
                    }
                    Some(AppCommand::StartTestRecording) => {
                        println!("Starting test recording...");
                        test_recording_buffer.clear();
                        test_recording_start = Some(Instant::now());
                        app_state.test_mode_active.store(true, Ordering::SeqCst);
                    }
                    Some(AppCommand::TestRecordingComplete(wav_data)) => {
                        println!("Playing back test recording ({} bytes)...", wav_data.len());
                        match play_wav_buffer(wav_data, Arc::clone(&playback_active)) {
                            Ok(stream) => {
                                _playback_stream = Some(stream);
                            }
                            Err(e) => {
                                eprintln!("Failed to play test recording: {}", e);
                                playback_active.store(false, Ordering::SeqCst);
                            }
                        }
                    }
                    Some(AppCommand::Quit) | None => {
                        // Quit command received or channel closed
                        println!("Shutting down...");
                        break;
                    }
                }
            }
            Some(event) = rx.recv() => {
                // Check if test recording is active and has timed out
                if let Some(start_time) = test_recording_start {
                    if start_time.elapsed() >= test_recording_duration {
                        println!("Test recording complete, {} bytes captured", test_recording_buffer.len());
                        app_state.test_mode_active.store(false, Ordering::SeqCst);
                        test_recording_start = None;

                        // Send the captured audio for playback
                        if !test_recording_buffer.is_empty() {
                            let wav_data = std::mem::take(&mut test_recording_buffer);
                            if let Err(e) = app_state.command_tx.try_send(AppCommand::TestRecordingComplete(wav_data)) {
                                eprintln!("Failed to send test recording complete: {}", e);
                            }
                        }
                    }
                }

                // Handle test recording mode: capture audio data instead of processing
                if app_state.test_mode_active.load(Ordering::Relaxed) {
                    if let AudioEvent::AudioData(audio_data) = event {
                        test_recording_buffer.extend_from_slice(&audio_data);
                    }
                    continue;
                }

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
                        let current_config = app_state.config.read().expect("Config lock poisoned").clone();
                        if let Err(e) = process_audio(
                            audio_data,
                            &current_config,
                            &socket,
                            &mut rate_limiter,
                            &typing_indicator,
                            &mut price_estimator,
                            recording_manager.as_ref(),
                            &app_state.log_tx,
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
