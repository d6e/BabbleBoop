use babble_boop::api_client::OPENAI_BASE_URL;
use babble_boop::app_state::{run_logging_failure, AppCommand, AppState, LogEntry};
use babble_boop::audio_playback::AudioOutput;
use babble_boop::audio_recording::{start_audio_recording, SharedAudioState};
use babble_boop::config::Config;
use babble_boop::data_dir;
use babble_boop::gui::{run_error_dialog, run_gui};
use babble_boop::processing_loop::{hold_audio_stream, run_processing_loop, AudioInput};
use babble_boop::types::AudioEvent;

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

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
    let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
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
    // The processing loop owns its copy of the settings; the GUI sends it
    // the saved settings with UpdateConfig.
    let loop_config = loaded.config.clone();
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
                loop_config,
                cmd_rx,
                data.dir,
                OPENAI_BASE_URL,
                || start_audio_input(&app_state_clone),
                AudioOutput::open_default,
            ))
        });
        app_state_clone.mark_processing_stopped();
        // Dropping the runtime waits for blocking tasks without a limit, and
        // a cancelled request can leave a DNS lookup running on one.
        rt.shutdown_timeout(Duration::from_secs(1));
    });

    // Run GUI on main thread
    if let Err(e) = run_gui(
        app_state,
        loaded.config,
        log_rx,
        first_run,
        loaded.warnings,
        config_file,
    ) {
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

/// Start the audio input stream of cpal on a new thread. The processing
/// loop calls this after its other startup steps.
fn start_audio_input(app_state: &Arc<AppState>) -> AudioInput {
    let (tx, events) = mpsc::channel::<AudioEvent>(100);
    let shared_audio = SharedAudioState::new(app_state);
    let audio_app_state = Arc::clone(app_state);
    let audio_logger = app_state.logger.clone();
    let (init_tx, started) = oneshot::channel();
    // This thread only owns the stream; the callback runs on a thread of
    // the audio backend. A panic here would stop capture with no sign in
    // the GUI, so it goes to the activity log.
    // The thread is not joined: if the driver never returns, the thread
    // stays blocked until the process exits.
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
    AudioInput { started, events }
}
