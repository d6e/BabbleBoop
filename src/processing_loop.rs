//! The processing loop: it receives the commands from the GUI and the
//! events from the audio callback, and sends the translations and the
//! typing indicator to VRChat. `main.rs` runs it on the processing thread
//! and gives it the audio input and output devices.

use crate::api_client::build_api_client;
use crate::app_state::{AppCommand, AppState, Logger};
use crate::audio_playback::{convert_for_output, PlaybackOutput};
use crate::audio_processing::process_audio;
use crate::audio_recording::AudioStreamInfo;
use crate::chatbox::Chatbox;
use crate::config::Config;
use crate::data_dir::DataDir;
use crate::models;
use crate::price_estimator::PriceEstimator;
use crate::rate_limiter::RateLimiter;
use crate::recorder::MAX_RECORDING;
use crate::recording_manager::RecordingManager;
use crate::shutdown::Shutdown;
use crate::types::{AudioEvent, CapturedAudio, Extent};
use crate::typing_indicator::TypingIndicator;
use crate::upload_audio::encode_upload_wav;
use std::error::Error;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

/// The audio input as the processing loop receives it. `main.rs` makes it
/// from the input stream of cpal.
pub struct AudioInput {
    /// The stream format when the input stream started, or the error that
    /// stopped it from starting
    pub started: oneshot::Receiver<Result<AudioStreamInfo, String>>,
    /// The events from the audio callback
    pub events: mpsc::Receiver<AudioEvent>,
}

/// Run the processing loop until shutdown is requested or the command
/// channel closes. `start_audio` starts the audio input, and
/// `open_output` opens the output device for each test recording
/// playback. Returns an error if the loop cannot start.
pub async fn run_processing_loop<O: PlaybackOutput>(
    app_state: Arc<AppState>,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
    data_dir: DataDir,
    start_audio: impl FnOnce() -> AudioInput,
    open_output: impl Fn() -> Result<O, Box<dyn Error + Send + Sync>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
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

    // Before the audio input starts, so a failure here does not leave
    // audio capture running with nothing to receive it.
    let api_client =
        build_api_client().map_err(|e| format!("cannot create the HTTP client: {}", e))?;

    app_state.logger.info("Starting audio recording...");
    app_state.logger.info(format!(
        "Translating to: {}",
        config.translation.target_language
    ));

    let AudioInput { started, events } = start_audio();
    let mut audio_events = AudioEvents::new(events, app_state.logger.clone());

    let Some(audio_stream_info) =
        wait_for_audio_start(started, &app_state.shutdown, AUDIO_START_TIMEOUT).await?
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
    let mut playback_stream: Option<O::Playback> = None;
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
                        if finish_test_recording(&mut test_recording, &mut playback_stream, &playback_errors, &app_state, &open_output).await.is_break() {
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
                if finish_test_recording(&mut test_recording, &mut playback_stream, &playback_errors, &app_state, &open_output).await.is_break() {
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
                    // Logged above. The callback follows a discarded
                    // recording with StopRecording.
                    AudioEvent::RecordingDiscarded => continue,
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
                    services.recording_manager.as_mut(),
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

/// Stop the test recording and play it back on an output that
/// `open_output` opens. The playback is kept in `playback` until the next
/// playback or the end of the loop, and reports its errors to
/// `playback_errors`. Breaks if shutdown stopped the work.
async fn finish_test_recording<O: PlaybackOutput>(
    test_recording: &mut TestRecording,
    playback: &mut Option<O::Playback>,
    playback_errors: &PlaybackErrors,
    app_state: &AppState,
    open_output: &impl Fn() -> Result<O, Box<dyn Error + Send + Sync>>,
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
    let output = match open_output() {
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

/// Longest test recording. The Stop button in the GUI ends it earlier.
pub const TEST_RECORDING_LIMIT: Duration = Duration::from_secs(30);

/// Number of interleaved samples in `TEST_RECORDING_LIMIT`.
fn test_recording_samples(channels: u16, sample_rate: u32) -> usize {
    TEST_RECORDING_LIMIT.as_secs() as usize * sample_rate as usize * usize::from(channels)
}

/// The test microphone recording. While it runs, the audio callback copies
/// the input into the test buffer instead of the recorder. The Stop button
/// in the GUI or `TEST_RECORDING_LIMIT` ends it.
pub struct TestRecording {
    active: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    channels: u16,
    sample_rate: u32,
    /// When the running test recording reaches the limit
    deadline: Option<Instant>,
}

impl TestRecording {
    /// Test recordings of the input stream with this format.
    pub fn new(app_state: &AppState, channels: u16, sample_rate: u32) -> Self {
        Self {
            active: Arc::clone(&app_state.test_mode_active),
            buffer: Arc::clone(&app_state.test_recording_buffer),
            channels,
            sample_rate,
            deadline: None,
        }
    }

    /// Start a test recording. A test recording that runs starts again.
    pub fn start(&mut self) {
        // The callback adds samples only up to this capacity
        let reserved = Vec::with_capacity(test_recording_samples(self.channels, self.sample_rate));
        let previous = std::mem::replace(&mut *self.lock_buffer(), reserved);
        drop(previous);
        self.deadline = Some(Instant::now() + TEST_RECORDING_LIMIT);
        self.active.store(true, Ordering::SeqCst);
    }

    /// Stop the test recording and return what it recorded. Returns `None`
    /// if no test recording runs.
    pub fn stop(&mut self) -> Option<CapturedAudio> {
        self.deadline.take()?;
        self.active.store(false, Ordering::SeqCst);
        // Leaves a buffer with no capacity, so the callback adds nothing
        // if it still sees test mode on.
        let samples = std::mem::take(&mut *self.lock_buffer());
        Some(CapturedAudio {
            samples,
            channels: self.channels,
            sample_rate: self.sample_rate,
        })
    }

    /// Resolves when the running test recording reaches the limit. Never
    /// resolves while no test recording runs.
    pub async fn limit_reached(&self) {
        match self.deadline {
            Some(deadline) => tokio::time::sleep_until(deadline).await,
            None => std::future::pending().await,
        }
    }

    /// The callback only copies samples while it holds the lock, so a
    /// poisoned lock still holds a valid buffer.
    fn lock_buffer(&self) -> MutexGuard<'_, Vec<f32>> {
        self.buffer.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Processing loop state that depends on the settings.
pub struct ProcessingServices {
    pub rate_limiter: RateLimiter,
    pub price_estimator: PriceEstimator,
    pub recording_manager: Option<RecordingManager>,
    /// Where the recording manager saves recordings
    recordings_dir: PathBuf,
}

impl ProcessingServices {
    /// Services that keep the total cost and the recordings in `data_dir`.
    pub fn new(config: &Config, data_dir: &DataDir, logger: &Logger) -> Self {
        log_model_warnings(config, logger);
        let recordings_dir = data_dir.recordings_dir();
        Self {
            rate_limiter: RateLimiter::new(config.rate_limit.requests_per_minute),
            price_estimator: PriceEstimator::new(
                data_dir.cost_file(),
                &config.openai.model,
                &config.openai.transcription_model,
            ),
            recording_manager: recording_manager(config, &recordings_dir),
            recordings_dir,
        }
    }

    /// Apply settings saved in the GUI.
    pub fn apply_config(&mut self, config: &Config, logger: &Logger) {
        // Keep the requests already counted; a new limiter would reset them.
        self.rate_limiter
            .set_max_requests(config.rate_limit.requests_per_minute);
        self.price_estimator
            .set_models(&config.openai.model, &config.openai.transcription_model);
        self.recording_manager = recording_manager(config, &self.recordings_dir);
        log_model_warnings(config, logger);
    }
}

/// Tell the user when a model is scheduled to shut down, or when the cost
/// display cannot be accurate. The GUI accepts any model name.
fn log_model_warnings(config: &Config, logger: &Logger) {
    let (model, transcription_model) = (&config.openai.model, &config.openai.transcription_model);
    for warning in models::shutdown_warnings(model, transcription_model)
        .into_iter()
        .chain(PriceEstimator::unknown_pricing(model, transcription_model))
    {
        logger.info(warning);
    }
}

fn recording_manager(config: &Config, recordings_dir: &Path) -> Option<RecordingManager> {
    config
        .keep_audio_files
        .then(|| RecordingManager::new(recordings_dir.to_path_buf(), config.max_audio_files))
}

/// Handle the translation toggle from the GUI.
///
/// Disabling turns the typing indicator off. While translation is off the
/// loop ignores audio events, so the StopRecording of an utterance that
/// started before would not turn it off. The GUI stores `enabled` before it
/// sends this command, so no StartRecording handled after this can turn the
/// indicator on again.
pub async fn apply_enabled(enabled: bool, typing_indicator: &TypingIndicator, logger: &Logger) {
    if enabled {
        logger.info("Translation enabled");
    } else {
        typing_indicator.stop_typing().await;
        logger.info("Translation disabled");
    }
}

/// The events from the audio callback, as the processing loop receives
/// them.
///
/// An input error or the end of the input also ends the recording that
/// runs: a StopRecording follows, so the loop turns the typing indicator
/// off. The callback does not send one, as it may not run again:
/// - After a panic, `PanicGuard` in `audio_recording.rs` runs nothing of
///   the callback.
/// - On WASAPI, cpal 0.15.3 ends the stream thread after the first stream
///   error (`src/host/wasapi/stream.rs` lines 343 to 360 and 388 to 390),
///   and the channel closes.
/// - On CoreAudio, cpal pauses the stream when the device is disconnected
///   and then reports the error (`src/host/coreaudio/macos/mod.rs` lines
///   465 to 468).
/// - On ALSA, cpal reports an error and polls the device again
///   (`src/host/alsa/mod.rs` lines 586 to 596), so while the error does
///   not clear, cpal calls the error callback and not the data callback.
///
/// If the input goes on after a stream error, the indicator is off for the
/// rest of the recording, or until a part of it is processed.
pub struct AudioEvents {
    /// `None` after every sender is gone
    rx: Option<mpsc::Receiver<AudioEvent>>,
    /// An input error was returned, and the StopRecording that follows it
    /// was not
    stop_pending: bool,
    logger: Logger,
}

impl AudioEvents {
    pub fn new(rx: mpsc::Receiver<AudioEvent>, logger: Logger) -> Self {
        Self {
            rx: Some(rx),
            stop_pending: false,
            logger,
        }
    }

    /// The next event, after its line in the activity log. When every
    /// sender is gone, logs an error once, returns StopRecording and then
    /// never resolves. The senders go when the audio thread of cpal ends,
    /// which on WASAPI follows the first stream error, possibly before its
    /// report fits in the channel.
    ///
    /// Cancel safe: `mpsc::Receiver::recv` is (tokio 1.48.0,
    /// `src/sync/mpsc/bounded.rs` lines 199 to 204), and this future
    /// changes the state only in the poll that returns the event.
    pub async fn recv(&mut self) -> AudioEvent {
        if std::mem::take(&mut self.stop_pending) {
            return AudioEvent::StopRecording;
        }
        if let Some(rx) = &mut self.rx {
            if let Some(event) = rx.recv().await {
                log_audio_event(&event, &self.logger);
                self.stop_pending = matches!(event, AudioEvent::InputError(_));
                return event;
            }
            self.rx = None;
            self.logger
                .error("Audio input stopped. Restart BabbleBoop to record again.");
            return AudioEvent::StopRecording;
        }
        std::future::pending().await
    }
}

/// Write the activity log line for an event from the audio callback. The
/// callback runs on the audio thread and does not log itself.
pub fn log_audio_event(event: &AudioEvent, logger: &Logger) {
    match event {
        AudioEvent::StartRecording => logger.info("Sound detected, recording..."),
        AudioEvent::AudioPart(_) => logger.info(format!(
            "Recording reached {} s, processing it while recording goes on...",
            MAX_RECORDING.as_secs()
        )),
        AudioEvent::AudioData(..) => logger.info("Silence detected, processing..."),
        AudioEvent::StopRecording => {}
        AudioEvent::RecordingDiscarded => {
            logger.info("Test Microphone started, the recording in progress is discarded")
        }
        AudioEvent::EventsDropped(count) => logger.error(format!(
            "Lost {} audio events because the processing queue was full",
            count
        )),
        AudioEvent::InputError(message) => logger.error(message.as_str()),
    }
}

/// Encode a recording for upload on a blocking thread. The processing loop
/// runs in `block_on` on the processing thread, so encoding in the loop
/// would stop the loop, and its check for shutdown, until the encoding ends.
pub async fn encode_for_upload(audio: CapturedAudio) -> Result<Vec<u8>, String> {
    match tokio::task::spawn_blocking(move || encode_upload_wav(&audio)).await {
        Ok(Ok(wav)) => Ok(wav),
        Ok(Err(e)) => Err(format!("cannot encode the recording: {}", e)),
        Err(e) => Err(format!("encoding the recording failed: {}", e)),
    }
}

/// Longest wait for the audio input stream to start. Opening a device
/// usually takes less than a second, but a driver can take a few seconds
/// (for example a Bluetooth headset that changes to its microphone
/// profile). A driver that never returns gives this error instead of an
/// application that records nothing and shows no reason.
pub const AUDIO_START_TIMEOUT: Duration = Duration::from_secs(15);

/// Wait until the audio thread reports whether the input stream started.
/// Returns the stream format, or `None` if shutdown is requested first.
/// The error is the startup error for the activity log.
///
/// The wait does not block the runtime, and it ends on shutdown, so a
/// driver that never returns cannot stop the application from closing.
pub async fn wait_for_audio_start<T>(
    started: oneshot::Receiver<Result<T, String>>,
    shutdown: &Shutdown,
    timeout: Duration,
) -> Result<Option<T>, String> {
    match shutdown
        .run_until(tokio::time::timeout(timeout, started))
        .await
    {
        None => Ok(None),
        Some(Ok(Ok(Ok(info)))) => Ok(Some(info)),
        Some(Ok(Ok(Err(e)))) => Err(format!("cannot start audio input: {}", e)),
        // The audio thread panicked. Its own log line gives the reason; it
        // can come before or after the line of this error.
        Some(Ok(Err(_))) => Err("cannot start audio input: the audio thread stopped".to_string()),
        Some(Err(_)) => Err(format!(
            "cannot start audio input: timed out after {} s",
            timeout.as_secs()
        )),
    }
}

/// Keep the audio input stream until shutdown is requested or the
/// processing thread ends. The processing thread also ends when it stops
/// waiting for the stream to start, so no stream records that nothing
/// reads.
pub fn hold_audio_stream<S>(stream: S, app_state: &AppState) {
    while !app_state.is_shutdown_requested() && !app_state.is_processing_stopped() {
        std::thread::sleep(Duration::from_millis(100));
    }
    drop(stream);
}

/// Room for playback errors that the processing loop did not log yet. The
/// reporter sends only an error that differs from the last one.
const PLAYBACK_ERROR_QUEUE: usize = 16;

/// Stream errors of the test playback. cpal reports them on an audio
/// thread, which does not log (see `StreamErrorReporter`); the processing
/// loop receives them here and logs them.
pub struct PlaybackErrors {
    tx: mpsc::Sender<String>,
    rx: mpsc::Receiver<String>,
}

impl Default for PlaybackErrors {
    fn default() -> Self {
        let (tx, rx) = mpsc::channel(PLAYBACK_ERROR_QUEUE);
        Self { tx, rx }
    }
}

impl PlaybackErrors {
    /// Where a new playback stream sends its errors.
    pub fn sender(&self) -> mpsc::Sender<String> {
        self.tx.clone()
    }

    /// Log the next playback error. Does not resolve while no error comes,
    /// as this holds a sender. Cancel safe: `mpsc::Receiver::recv` is.
    pub async fn log_next(&mut self, logger: &Logger) {
        if let Some(message) = self.rx.recv().await {
            logger.error(message);
        }
    }
}

/// Convert a test recording for the output device on a blocking thread.
/// Resampling 30 s of 48 kHz stereo to 44.1 kHz takes about 0.1 s in a
/// release build and 3 s in a debug build.
pub async fn convert_for_playback(
    audio: CapturedAudio,
    channels: u16,
    sample_rate: u32,
) -> Result<Vec<f32>, String> {
    tokio::task::spawn_blocking(move || convert_for_output(&audio, channels, sample_rate))
        .await
        .map_err(|e| format!("converting the test recording failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::LogEntry;
    use rosc::{OscPacket, OscType};
    use std::future::Future;
    use std::sync::atomic::AtomicUsize;

    /// The input and output format of the tests. The output plays the
    /// input format, so a test recording plays unchanged.
    const SAMPLE_RATE: u32 = 16_000;

    /// A message that VRChat received from the loop.
    #[derive(Debug, PartialEq)]
    enum Osc {
        Typing(bool),
        Input(String),
    }

    /// An output device that plays nothing. It sends the samples of each
    /// playback to the test and counts the playbacks that are kept alive.
    #[derive(Clone)]
    struct FakeOutput {
        played: mpsc::UnboundedSender<Vec<f32>>,
        alive: Arc<AtomicUsize>,
    }

    struct FakePlayback(Arc<AtomicUsize>);

    impl Drop for FakePlayback {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl PlaybackOutput for FakeOutput {
        type Playback = FakePlayback;

        fn channels(&self) -> u16 {
            1
        }

        fn sample_rate(&self) -> u32 {
            SAMPLE_RATE
        }

        fn play(
            &self,
            samples: Vec<f32>,
            _errors: mpsc::Sender<String>,
        ) -> Result<FakePlayback, Box<dyn Error + Send + Sync>> {
            self.played.send(samples)?;
            self.alive.fetch_add(1, Ordering::SeqCst);
            Ok(FakePlayback(Arc::clone(&self.alive)))
        }
    }

    /// The processing loop with fakes for its devices: the audio input is
    /// a channel that the test fills, and the output is a `FakeOutput`.
    struct LoopUnderTest {
        app_state: Arc<AppState>,
        cmd_rx: mpsc::Receiver<AppCommand>,
        input: AudioInput,
        output: FakeOutput,
        audio_started: Arc<AtomicBool>,
        ended: oneshot::Sender<()>,
    }

    impl LoopUnderTest {
        /// Run the loop together with `test` until both end. Returns the
        /// result of the loop.
        async fn run_with(self, test: impl Future<Output = ()>) -> Result<(), String> {
            let Self {
                app_state,
                cmd_rx,
                input,
                output,
                audio_started,
                ended,
            } = self;
            let data_dir = DataDir::new(
                std::env::temp_dir()
                    .join(format!("babble_boop_no_data_dir_{}", std::process::id())),
            );
            let processing = async {
                let result = run_processing_loop(
                    app_state,
                    cmd_rx,
                    data_dir,
                    move || {
                        audio_started.store(true, Ordering::SeqCst);
                        input
                    },
                    move || Ok(output.clone()),
                )
                .await;
                ended
                    .send(())
                    .expect("the test does not wait for the end of the loop");
                result.map_err(|e| e.to_string())
            };
            let (result, ()) = tokio::join!(processing, test);
            result
        }
    }

    /// What the test does to the loop and what it sees of it.
    struct Driver {
        app_state: Arc<AppState>,
        commands: mpsc::Sender<AppCommand>,
        events: mpsc::Sender<AudioEvent>,
        started: Option<oneshot::Sender<Result<AudioStreamInfo, String>>>,
        vrchat: UdpSocket,
        log: mpsc::Receiver<LogEntry>,
        played: mpsc::UnboundedReceiver<Vec<f32>>,
        playbacks_alive: Arc<AtomicUsize>,
        audio_started: Arc<AtomicBool>,
        ended: oneshot::Receiver<()>,
        /// Longest wait for an effect of the loop. Only a failing test
        /// waits this long.
        wait: Duration,
    }

    /// A processing loop that sends to a local VRChat socket, with the
    /// settings that `configure` changes.
    async fn processing_loop(configure: impl FnOnce(&mut Config)) -> (LoopUnderTest, Driver) {
        let vrchat = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.input_port = 0;
        config.osc.output_port = vrchat.local_addr().unwrap().port();
        configure(&mut config);
        let (cmd_tx, cmd_rx) = mpsc::channel(32);
        let (log_tx, log) = mpsc::channel(100);
        let app_state = Arc::new(AppState::new(config, cmd_tx.clone(), log_tx));
        let (events_tx, events) = mpsc::channel(100);
        let (started_tx, started) = oneshot::channel();
        let (played_tx, played) = mpsc::unbounded_channel();
        let playbacks_alive = Arc::new(AtomicUsize::new(0));
        let audio_started = Arc::new(AtomicBool::new(false));
        let (ended_tx, ended) = oneshot::channel();
        let processing = LoopUnderTest {
            app_state: Arc::clone(&app_state),
            cmd_rx,
            input: AudioInput { started, events },
            output: FakeOutput {
                played: played_tx,
                alive: Arc::clone(&playbacks_alive),
            },
            audio_started: Arc::clone(&audio_started),
            ended: ended_tx,
        };
        let driver = Driver {
            app_state,
            commands: cmd_tx,
            events: events_tx,
            started: Some(started_tx),
            vrchat,
            log,
            played,
            playbacks_alive,
            audio_started,
            ended,
            wait: Duration::from_secs(5),
        };
        (processing, driver)
    }

    impl Driver {
        /// Report that the audio input started, as the audio thread does.
        fn start_audio(&mut self) {
            self.report_audio_start(Ok(AudioStreamInfo {
                sample_rate: SAMPLE_RATE,
                channels: 1,
            }));
        }

        fn report_audio_start(&mut self, result: Result<AudioStreamInfo, String>) {
            let started = self
                .started
                .take()
                .expect("the audio start is reported once");
            assert!(
                started.send(result).is_ok(),
                "the loop does not wait for the audio start"
            );
        }

        async fn command(&self, command: AppCommand) {
            self.commands.send(command).await.unwrap();
        }

        async fn event(&self, event: AudioEvent) {
            self.events.send(event).await.unwrap();
        }

        /// Wait for an activity log line that contains `text`, and skip the
        /// lines before it.
        async fn logged(&mut self, text: &str) {
            let mut seen = Vec::new();
            let found = tokio::time::timeout(self.wait, async {
                while let Some(entry) = self.log.recv().await {
                    if entry.message.contains(text) {
                        return true;
                    }
                    seen.push(entry.message);
                }
                false
            })
            .await;
            assert_eq!(found, Ok(true), "{:?} not logged after {:?}", text, seen);
        }

        /// The next message that VRChat receives.
        async fn received(&self) -> Osc {
            let mut buf = [0u8; 1024];
            let len = tokio::time::timeout(self.wait, self.vrchat.recv(&mut buf))
                .await
                .expect("VRChat received no message")
                .unwrap();
            let OscPacket::Message(message) = rosc::decoder::decode_udp(&buf[..len]).unwrap().1
            else {
                panic!("VRChat received a bundle");
            };
            match (message.addr.as_str(), message.args.first()) {
                ("/chatbox/typing", Some(OscType::Bool(typing))) => Osc::Typing(*typing),
                ("/chatbox/input", Some(OscType::String(text))) => Osc::Input(text.clone()),
                _ => panic!("unexpected OSC message {:?}", message),
            }
        }

        /// The samples of the next playback on the output.
        async fn played(&mut self) -> Vec<f32> {
            tokio::time::timeout(self.wait, self.played.recv())
                .await
                .expect("nothing was played")
                .unwrap()
        }

        /// Wait for the loop to end.
        async fn ended(&mut self) {
            tokio::time::timeout(self.wait, &mut self.ended)
                .await
                .expect("the processing loop did not end")
                .unwrap();
        }

        /// Request shutdown and wait for the loop to end.
        async fn shut_down(&mut self) {
            self.app_state.shutdown.request();
            self.ended().await;
        }
    }

    /// `seconds` of a constant mono signal.
    fn sound(seconds: f32) -> CapturedAudio {
        CapturedAudio {
            samples: vec![0.25; (SAMPLE_RATE as f32 * seconds) as usize],
            channels: 1,
            sample_rate: SAMPLE_RATE,
        }
    }

    /// Settings that skip every whole recording before the transcription
    /// request.
    fn skip_short_recordings(config: &mut Config) {
        config.audio.min_transcription_duration = 10.0;
    }

    #[tokio::test]
    async fn test_a_short_recording_turns_typing_on_and_off_without_a_request() {
        let (processing, mut d) = processing_loop(skip_short_recordings).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                d.event(AudioEvent::AudioData(sound(0.5), Extent::Whole))
                    .await;
                d.event(AudioEvent::StopRecording).await;
                // On at the start, off when process_audio skips the
                // recording, off at the stop
                assert_eq!(d.received().await, Osc::Typing(true));
                assert_eq!(d.received().await, Osc::Typing(false));
                assert_eq!(d.received().await, Osc::Typing(false));
                d.logged("Audio too short").await;
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        // Off at shutdown
        assert_eq!(d.received().await, Osc::Typing(false));
    }

    #[tokio::test]
    async fn test_speech_while_translation_is_off_does_not_turn_typing_on() {
        let (processing, mut d) = processing_loop(skip_short_recordings).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                // The GUI stores the toggle before it sends the command
                d.app_state.enabled.store(false, Ordering::Relaxed);
                d.command(AppCommand::SetEnabled(false)).await;
                assert_eq!(d.received().await, Osc::Typing(false));
                d.event(AudioEvent::StartRecording).await;
                d.event(AudioEvent::AudioData(sound(0.5), Extent::Whole))
                    .await;
                // The loop logs an event when it receives it, so it handled
                // the StartRecording before
                d.logged("Silence detected").await;
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        // Off at shutdown, and not on before
        assert_eq!(d.received().await, Osc::Typing(false));
    }

    #[tokio::test]
    async fn test_a_discarded_recording_is_logged_and_turns_typing_off() {
        let (processing, mut d) = processing_loop(skip_short_recordings).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                d.event(AudioEvent::RecordingDiscarded).await;
                d.event(AudioEvent::StopRecording).await;
                assert_eq!(d.received().await, Osc::Typing(true));
                assert_eq!(d.received().await, Osc::Typing(false));
                d.logged("Test Microphone started, the recording in progress is discarded")
                    .await;
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.received().await, Osc::Typing(false));
    }

    #[tokio::test]
    async fn test_an_audio_input_error_is_logged_and_turns_typing_off() {
        let (processing, mut d) = processing_loop(skip_short_recordings).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                d.event(AudioEvent::InputError(
                    "Audio input error: unplugged".into(),
                ))
                .await;
                assert_eq!(d.received().await, Osc::Typing(true));
                assert_eq!(d.received().await, Osc::Typing(false));
                d.logged("Audio input error: unplugged").await;
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.received().await, Osc::Typing(false));
    }

    #[tokio::test]
    async fn test_an_audio_start_error_ends_the_loop_with_the_error() {
        let (processing, mut d) = processing_loop(|_| {}).await;
        let result = processing
            .run_with(async {
                d.report_audio_start(Err("No input device available".into()));
                d.ended().await;
            })
            .await;

        assert_eq!(
            result,
            Err("cannot start audio input: No input device available".to_string())
        );
        assert!(!d.app_state.is_shutdown_requested());
    }

    #[tokio::test]
    async fn test_a_busy_osc_port_ends_the_loop_before_the_audio_input_starts() {
        let busy = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let port = busy.local_addr().unwrap().port();
        let (processing, mut d) = processing_loop(|config| config.osc.input_port = port).await;
        let result = processing.run_with(d.ended()).await;

        let error = result.unwrap_err();
        assert!(
            error.starts_with(&format!("cannot open OSC port 127.0.0.1:{}", port)),
            "{}",
            error
        );
        assert!(!d.audio_started.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_a_stopped_test_recording_plays_on_the_output() {
        let (processing, mut d) = processing_loop(|_| {}).await;
        let recorded: Vec<f32> = (0..160).map(|n| n as f32 / 160.0).collect();
        let result = processing
            .run_with(async {
                d.start_audio();
                d.command(AppCommand::StartTestRecording).await;
                d.logged("Test recording started").await;
                // What the audio callback does in test mode
                d.app_state
                    .test_recording_buffer
                    .lock()
                    .unwrap()
                    .extend_from_slice(&recorded);
                d.command(AppCommand::StopTestRecording).await;
                assert_eq!(d.played().await, recorded);
                // The loop keeps the playback after it handles the next
                // command
                d.command(AppCommand::SetEnabled(true)).await;
                d.logged("Translation enabled").await;
                assert_eq!(d.playbacks_alive.load(Ordering::SeqCst), 1);
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.playbacks_alive.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_a_test_recording_plays_when_it_reaches_its_limit() {
        let (processing, mut d) = processing_loop(|_| {}).await;
        // The paused clock moves on to the limit while the test waits
        d.wait = TEST_RECORDING_LIMIT * 2;
        let recorded = vec![0.5; 160];
        let result = processing
            .run_with(async {
                d.start_audio();
                d.command(AppCommand::StartTestRecording).await;
                d.logged("Test recording started").await;
                let started = Instant::now();
                d.app_state
                    .test_recording_buffer
                    .lock()
                    .unwrap()
                    .extend_from_slice(&recorded);
                assert_eq!(d.played().await, recorded);
                assert_eq!(started.elapsed(), TEST_RECORDING_LIMIT);
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
    }

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_during_an_utterance_ends_the_loop_and_turns_typing_off() {
        let (processing, mut d) = processing_loop(|config| {
            config.audio.min_transcription_duration = 0.0;
            // The rate limiter waits a minute before the transcription
            // request. Every wait of the test is shorter, so the test
            // sends no request, also when it fails.
            config.rate_limit.requests_per_minute = 0;
        })
        .await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                assert_eq!(d.received().await, Osc::Typing(true));
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                d.logged("Silence detected").await;
                // The paused clock moves only when every task waits for a
                // timer, and not while the recording is encoded on a
                // blocking thread. So this sleep ends when the loop waits
                // for the rate limiter in process_audio.
                tokio::time::sleep(Duration::from_secs(1)).await;
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.received().await, Osc::Typing(false));
    }

    #[tokio::test]
    async fn test_shutdown_comes_before_the_events_that_wait() {
        let (processing, mut d) = processing_loop(|_| {}).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                // The loop waits in its select after it logs this
                d.command(AppCommand::SetEnabled(true)).await;
                d.logged("Translation enabled").await;
                // Both are ready the next time the loop runs
                d.app_state.shutdown.request();
                d.event(AudioEvent::StartRecording).await;
                d.ended().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        // Off at shutdown; the StartRecording was not handled
        assert_eq!(d.received().await, Osc::Typing(false));
    }
}
