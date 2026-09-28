use babble_boop::api_client::build_api_client;
use babble_boop::app_state::{run_logging_failure, AppCommand, AppState, LogEntry};
use babble_boop::audio_playback::AudioOutput;
use babble_boop::audio_processing::process_audio;
use babble_boop::audio_recording::{start_audio_recording, SharedAudioState};
use babble_boop::chatbox::Chatbox;
use babble_boop::config::Config;
use babble_boop::data_dir::{self, DataDir};
use babble_boop::gui::{run_error_dialog, run_gui};
use babble_boop::processing_loop::{
    apply_enabled, convert_for_playback, encode_for_upload, hold_audio_stream,
    wait_for_audio_start, AudioEvents, PlaybackErrors, ProcessingServices, TestRecording,
    AUDIO_START_TIMEOUT, TEST_RECORDING_LIMIT,
};
use babble_boop::types::{AudioEvent, Extent};
use babble_boop::typing_indicator::TypingIndicator;

use std::ops::ControlFlow;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

/// Stop the test recording and play it back. The stream is kept in
/// `playback` until the next playback or the end of the loop, and reports
/// its errors to `playback_errors`. Breaks if shutdown stopped the work.
async fn finish_test_recording(
    test_recording: &mut TestRecording,
    playback: &mut Option<cpal::Stream>,
    playback_errors: &PlaybackErrors,
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
    match output.play(samples, playback_errors.sender()) {
        Ok(stream) => *playback = Some(stream),
        Err(e) => logger.error(format!("Failed to play test recording: {}", e)),
    }
    ControlFlow::Continue(())
}

fn main() {
    let data = data_dir::locate();
    let config_file = data.dir.config_file();
    // Load or create config
    let (loaded, first_run) = match Config::load_or_create(&config_file) {
        Ok(result) => result,
        Err(e) => {
            let message = format!(
                "Failed to load or create config file {}: {}\n\n\
                Please check file permissions and try again.",
                config_file.display(),
                e
            );
            eprintln!("{}", message);
            #[expect(
                clippy::let_underscore_must_use,
                reason = "the message is on stderr already if the dialog fails"
            )]
            let _ = run_error_dialog("Configuration Error", &message);
            std::process::exit(1);
        }
    };

    // Create command channel for GUI -> processing communication
    let (cmd_tx, cmd_rx) = mpsc::channel::<AppCommand>(32);

    // Create log channel for processing -> GUI communication
    let (log_tx, log_rx) = mpsc::channel::<LogEntry>(100);

    // Create shared app state
    let app_state = Arc::new(AppState::new(loaded.config, cmd_tx, log_tx));
    // The logger did not exist when the data folder was selected and the
    // config was loaded
    if data.is_error() {
        app_state.logger.error(data.to_string());
    } else {
        app_state.logger.info(data.to_string());
    }
    for warning in &loaded.warnings {
        app_state.logger.info(warning.to_string());
    }
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
            rt.block_on(run_processing_loop(
                Arc::clone(&app_state_clone),
                cmd_rx,
                data.dir,
            ))
        });
        app_state_clone.mark_processing_stopped();
        // Dropping the runtime waits for blocking tasks without a limit, and
        // a cancelled request can leave a DNS lookup running on one.
        rt.shutdown_timeout(Duration::from_secs(1));
    });

    // Run GUI on main thread
    if let Err(e) = run_gui(app_state, log_rx, first_run, loaded.warnings, config_file) {
        let message = format!("Failed to start the application window: {}", e);
        eprintln!("{}", message);
        #[expect(
            clippy::let_underscore_must_use,
            reason = "the message is on stderr already if the dialog fails"
        )]
        let _ = run_error_dialog("GUI Error", &message);
    }

    // The GUI requests shutdown on exit; request it again in case the GUI
    // failed to start, so the processing thread does not run forever.
    shutdown.request();
    #[expect(
        clippy::let_underscore_must_use,
        reason = "join fails only if the thread panicked, and the program exits in both cases"
    )]
    let _ = processing_handle.join();
}

async fn run_processing_loop(
    app_state: Arc<AppState>,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
    data_dir: DataDir,
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

    let (tx, rx) = mpsc::channel::<AudioEvent>(100);
    let mut audio_events = AudioEvents::new(rx, app_state.logger.clone());

    // Start the audio recording in a separate thread
    let shared_audio = SharedAudioState::new(&app_state);
    let audio_app_state = Arc::clone(&app_state);
    let audio_logger = app_state.logger.clone();
    let (init_tx, init_rx) = tokio::sync::oneshot::channel();
    // This thread only owns the stream; the callback runs on a thread of
    // the audio backend. A panic here would stop capture with no sign in
    // the GUI, so it goes to the activity log.
    std::thread::spawn(move || {
        run_logging_failure(&audio_logger, "Audio input", || {
            match start_audio_recording(shared_audio, tx) {
                Ok((stream, stream_info)) => {
                    #[expect(
                        clippy::let_underscore_must_use,
                        reason = "send fails only if the processing loop stopped waiting; the processing thread then ends, and hold_audio_stream drops the stream"
                    )]
                    let _ = init_tx.send(Ok(stream_info));
                    hold_audio_stream(stream, &audio_app_state);
                }
                Err(e) => {
                    // The processing loop reports this as a startup error
                    #[expect(
                        clippy::let_underscore_must_use,
                        reason = "send fails only if the processing loop stopped waiting for the stream"
                    )]
                    let _ = init_tx.send(Err(e.to_string()));
                }
            }
            Ok::<(), std::convert::Infallible>(())
        });
    });

    // The audio thread is not joined: if the driver never returns, the
    // thread stays blocked until the process exits.
    let Some(audio_stream_info) =
        wait_for_audio_start(init_rx, &app_state.shutdown, AUDIO_START_TIMEOUT).await?
    else {
        // Shutdown was requested before the stream started
        return Ok(());
    };

    app_state.logger.info(format!(
        "Audio: {} ch, {} Hz",
        audio_stream_info.channels, audio_stream_info.sample_rate
    ));

    let mut services = ProcessingServices::new(&config, &data_dir, &app_state.logger);
    // Initialize the shared cost from the loaded value
    app_state.set_total_cost(services.price_estimator.total_cost);

    let typing_indicator = TypingIndicator::new(
        Arc::clone(&socket),
        Arc::clone(&app_state.config),
        app_state.logger.clone(),
    );

    let mut chatbox = Chatbox::new(Arc::clone(&socket));

    let mut test_recording = TestRecording::new(
        &app_state,
        audio_stream_info.channels,
        audio_stream_info.sample_rate,
    );
    // Keep playback stream alive until playback completes
    let mut playback_stream: Option<cpal::Stream> = None;
    let mut playback_errors = PlaybackErrors::default();

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
                        if finish_test_recording(&mut test_recording, &mut playback_stream, &playback_errors, &app_state).await.is_break() {
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
                if finish_test_recording(&mut test_recording, &mut playback_stream, &playback_errors, &app_state).await.is_break() {
                    break;
                }
            }
            // Errors of the playback stream, reported on the audio thread
            _ = playback_errors.log_next(&app_state.logger) => {}
            event = audio_events.recv() => {
                // Ignore speech while translation is off. SetEnabled(false)
                // turns off a typing indicator that is still on.
                if !app_state.enabled.load(Ordering::Relaxed) {
                    continue;
                }

                // A part of a long recording: the typing indicator is to stay
                // on after process_audio turns it off.
                let recording_goes_on = matches!(event, AudioEvent::AudioPart(_));
                let (audio, extent) = match event {
                    AudioEvent::StartRecording => {
                        typing_indicator.start_typing().await;
                        continue;
                    }
                    AudioEvent::StopRecording => {
                        typing_indicator.stop_typing().await;
                        continue;
                    }
                    // Logged above. AudioEvents follows an input error
                    // with StopRecording.
                    AudioEvent::EventsDropped(_) | AudioEvent::InputError(_) => continue,
                    AudioEvent::AudioData(audio, extent) => (audio, extent),
                    AudioEvent::AudioPart(audio) => (audio, Extent::Part),
                };
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
                    extent,
                    &current_config,
                    &mut chatbox,
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

    // Shutdown can stop processing between StartRecording and the end of
    // process_audio. Do not leave VRChat showing the typing indicator.
    typing_indicator.stop_typing().await;

    Ok(())
}
