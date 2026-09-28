use babble_boop::api_client::build_api_client;
use babble_boop::app_state::{run_logging_failure, AppCommand, AppState, LogEntry};
use babble_boop::audio_playback::AudioOutput;
use babble_boop::audio_processing::process_audio;
use babble_boop::audio_recording::{start_audio_recording, SharedAudioState};
use babble_boop::config::{Config, CONFIG_PATH};
use babble_boop::gui::{run_error_dialog, run_gui};
use babble_boop::processing_loop::{
    apply_enabled, convert_for_playback, encode_for_upload, log_audio_event, ProcessingServices,
    TestRecording, TEST_RECORDING_LIMIT,
};
use babble_boop::types::AudioEvent;
use babble_boop::typing_indicator::TypingIndicator;

use std::ops::ControlFlow;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

/// Stop the test recording and play it back. The stream is kept in
/// `playback` until the next playback or the end of the loop. Breaks if
/// shutdown stopped the work.
async fn finish_test_recording(
    test_recording: &mut TestRecording,
    playback: &mut Option<cpal::Stream>,
    app_state: &AppState,
) -> ControlFlow<()> {
    let logger = &app_state.logger;
    let Some(audio) = test_recording.stop() else {
        return ControlFlow::Continue(());
    };
    if audio.samples.is_empty() {
        logger.info("Test recording stopped, nothing recorded");
        return ControlFlow::Continue(());
    }
    logger.info(format!(
        "Test recording stopped, {} samples",
        audio.samples.len()
    ));
    let output = match AudioOutput::open_default() {
        Ok(output) => output,
        Err(e) => {
            logger.error(format!("Failed to play test recording: {}", e));
            return ControlFlow::Continue(());
        }
    };
    let converted = app_state.shutdown.run_until(convert_for_playback(
        audio,
        output.channels(),
        output.sample_rate(),
    ));
    let samples = match converted.await {
        Some(Ok(samples)) => samples,
        Some(Err(e)) => {
            logger.error(format!("Failed to play test recording: {}", e));
            return ControlFlow::Continue(());
        }
        None => return ControlFlow::Break(()),
    };
    logger.info("Playing back test recording...");
    match output.play(samples) {
        Ok(stream) => *playback = Some(stream),
        Err(e) => logger.error(format!("Failed to play test recording: {}", e)),
    }
    ControlFlow::Continue(())
}

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
    let shutdown = app_state.shutdown.clone();

    // Spawn background thread with tokio runtime for audio processing.
    // Errors and panics go to the activity log, as the GUI shows no other
    // sign that processing stopped.
    let processing_handle = std::thread::spawn(move || {
        let logger = app_state_clone.logger.clone();
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                logger.error(format!("Processing stopped: cannot start tokio: {}", e));
                app_state_clone.mark_processing_stopped();
                return;
            }
        };
        run_logging_failure(&logger, "Processing", || {
            rt.block_on(run_processing_loop(Arc::clone(&app_state_clone), cmd_rx))
        });
        app_state_clone.mark_processing_stopped();
        // Dropping the runtime waits for blocking tasks without a limit, and
        // a cancelled request can leave a DNS lookup running on one.
        rt.shutdown_timeout(Duration::from_secs(1));
    });

    // Run GUI on main thread
    if let Err(e) = run_gui(app_state, log_rx, first_run) {
        let message = format!("Failed to start the application window: {}", e);
        eprintln!("{}", message);
        let _ = run_error_dialog("GUI Error", &message);
    }

    // The GUI requests shutdown on exit; request it again in case the GUI
    // failed to start, so the processing thread does not run forever.
    shutdown.request();
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
    let socket = UdpSocket::bind(&socket_address)
        .await
        .map_err(|e| format!("cannot open OSC port {}: {}", socket_address, e))?;
    let socket = Arc::new(socket);

    // Before the audio thread starts, so a failure here does not leave
    // audio capture running with nothing to receive it.
    let api_client =
        build_api_client().map_err(|e| format!("cannot create the HTTP client: {}", e))?;

    app_state.logger.info("Starting audio recording...");
    app_state.logger.info(format!(
        "Translating to: {}",
        config.translation.target_language
    ));

    let (tx, mut rx) = mpsc::channel::<AudioEvent>(100);

    // Start the audio recording in a separate thread
    let shared_audio = SharedAudioState::new(&app_state);
    let shutdown_signal = app_state.shutdown.clone();
    let audio_logger = app_state.logger.clone();
    let (init_tx, init_rx) =
        std::sync::mpsc::channel::<Result<babble_boop::audio_recording::AudioStreamInfo, String>>();
    // This thread only owns the stream; the callback runs on a thread of
    // the audio backend. A panic here would stop capture with no sign in
    // the GUI, so it goes to the activity log.
    std::thread::spawn(move || {
        run_logging_failure(&audio_logger, "Audio input", || {
            match start_audio_recording(shared_audio, tx) {
                Ok((stream, stream_info)) => {
                    let _ = init_tx.send(Ok(stream_info));
                    let _stream = stream;
                    // Check shutdown signal periodically instead of parking forever
                    while !shutdown_signal.is_requested() {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    // Stream is dropped here, stopping audio capture
                }
                Err(e) => {
                    // The processing loop reports this as a startup error
                    let _ = init_tx.send(Err(e.to_string()));
                }
            }
            Ok::<(), std::convert::Infallible>(())
        });
    });

    // Wait for stream initialization and get stream info
    let audio_stream_info = init_rx
        .recv()
        .map_err(|_| "audio recording thread failed to start")?
        .map_err(|e| format!("cannot start audio input: {}", e))?;

    app_state.logger.info(format!(
        "Audio: {} ch, {} Hz",
        audio_stream_info.channels, audio_stream_info.sample_rate
    ));

    let mut services = ProcessingServices::new(&config, &app_state.logger);
    // Initialize the shared cost from the loaded value
    app_state.set_total_cost(services.price_estimator.total_cost);

    let typing_indicator = TypingIndicator::new(Arc::clone(&socket), Arc::clone(&app_state.config));

    let mut test_recording = TestRecording::new(
        &app_state,
        audio_stream_info.channels,
        audio_stream_info.sample_rate,
    );
    // Keep playback stream alive until playback completes
    let mut playback_stream: Option<cpal::Stream> = None;

    loop {
        tokio::select! {
            // Prioritize shutdown and the command channel to quit promptly
            biased;

            _ = app_state.shutdown.requested() => break,
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(AppCommand::SetEnabled(enabled)) => {
                        apply_enabled(enabled, &typing_indicator, &app_state.logger).await;
                    }
                    Some(AppCommand::UpdateConfig(new_config)) => {
                        app_state.logger.info("Config updated");
                        // Update hot-reloadable audio params
                        app_state.audio_params.update(&new_config.audio);
                        services.apply_config(&new_config, &app_state.logger);
                    }
                    Some(AppCommand::StartTestRecording) => {
                        app_state.logger.info("Test recording started...");
                        test_recording.start();
                    }
                    Some(AppCommand::StopTestRecording) => {
                        if finish_test_recording(&mut test_recording, &mut playback_stream, &app_state).await.is_break() {
                            break;
                        }
                    }
                    Some(AppCommand::Quit) | None => {
                        // Quit command received or channel closed
                        break;
                    }
                }
            }
            _ = test_recording.limit_reached() => {
                app_state.logger.info(format!(
                    "Test recording reached {} s",
                    TEST_RECORDING_LIMIT.as_secs()
                ));
                if finish_test_recording(&mut test_recording, &mut playback_stream, &app_state).await.is_break() {
                    break;
                }
            }
            Some(event) = rx.recv() => {
                log_audio_event(&event, &app_state.logger);

                // Ignore speech while translation is off. SetEnabled(false)
                // turns off a typing indicator that is still on.
                if !app_state.enabled.load(Ordering::Relaxed) {
                    continue;
                }

                // A part of a long recording: the typing indicator is to stay
                // on after process_audio turns it off.
                let recording_goes_on = matches!(event, AudioEvent::AudioPart(_));
                match event {
                    AudioEvent::StartRecording => {
                        typing_indicator.start_typing().await;
                    }
                    AudioEvent::StopRecording => {
                        typing_indicator.stop_typing().await;
                    }
                    // Logged above
                    AudioEvent::EventsDropped(_) | AudioEvent::InputError(_) => {}
                    AudioEvent::AudioData(audio) | AudioEvent::AudioPart(audio) => {
                        let audio_data = match app_state.shutdown.run_until(encode_for_upload(audio)).await {
                            Some(Ok(wav)) => wav,
                            Some(Err(e)) => {
                                app_state.logger.error(format!("Error: {}", e));
                                continue;
                            }
                            None => break,
                        };
                        // Read current config for processing
                        let current_config = app_state.config.read().expect("Config lock poisoned").clone();
                        // Shutdown drops the work, including the chatbox
                        // display pause and rate limiter wait.
                        let result = app_state.shutdown.run_until(process_audio(
                            &api_client,
                            audio_data,
                            &current_config,
                            &socket,
                            &mut services.rate_limiter,
                            &typing_indicator,
                            &mut services.price_estimator,
                            services.recording_manager.as_ref(),
                            &app_state,
                        ))
                        .await;
                        match result {
                            Some(Ok(())) => {}
                            Some(Err(e)) => app_state.logger.error_api(format!("Error: {}", e)),
                            None => break,
                        }
                        if recording_goes_on {
                            typing_indicator.start_typing().await;
                        }
                    }
                }
            }
        }
    }

    // Shutdown can stop processing between StartRecording and the end of
    // process_audio. Do not leave VRChat showing the typing indicator.
    typing_indicator.stop_typing().await;

    Ok(())
}
