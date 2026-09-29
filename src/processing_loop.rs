//! The processing loop: it receives the commands from the GUI and the
//! events from the audio callback, and sends the translations and the
//! typing indicator to VRChat. `main.rs` runs it on the processing thread
//! and gives it the audio input and output devices.

use crate::api_client::{ApiError, OpenAi};
use crate::app_state::{AppCommand, AppState, AudioShared, Logger};
use crate::audio_playback::{convert_for_output, PlaybackOutput};
use crate::audio_recording::AudioStreamInfo;
use crate::chatbox::Chatbox;
use crate::config::Config;
use crate::data_dir::DataDir;
use crate::pipeline::{Pipeline, ProcessingServices};
use crate::recorder::MAX_RECORDING;
use crate::shutdown::Shutdown;
use crate::types::{AudioEvent, CapturedAudio, Extent};
use crate::typing_indicator::TypingIndicator;
use crate::upload_audio::encode_upload_wav;
use std::error::Error;
use std::ops::ControlFlow;
use std::sync::atomic::Ordering;
use std::sync::{Arc, MutexGuard, PoisonError};
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
/// channel closes. The loop owns its settings: it starts with `config`,
/// and only `AppCommand::UpdateConfig` replaces them. The loop sends its
/// API requests to `api_base_url`, such as `OPENAI_BASE_URL`.
/// `start_audio` starts the audio input, and `open_output` opens the
/// output device for each test recording playback. Returns an error if the
/// loop cannot start.
pub async fn run_processing_loop<O: PlaybackOutput>(
    app_state: Arc<AppState>,
    mut config: Config,
    mut cmd_rx: mpsc::Receiver<AppCommand>,
    data_dir: DataDir,
    api_base_url: &str,
    start_audio: impl FnOnce() -> AudioInput,
    open_output: impl Fn() -> Result<O, Box<dyn Error + Send + Sync>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Where the socket in use is bound. After a failed rebind this differs
    // from the address in the settings.
    let mut socket_address = osc_socket_address(&config);
    let socket = UdpSocket::bind(&socket_address)
        .await
        .map_err(|e| format!("cannot open OSC port {}: {}", socket_address, e))?;
    let socket = Arc::new(socket);

    // Before the audio input starts, so a failure here does not leave
    // audio capture running with nothing to receive it.
    let api =
        OpenAi::new(api_base_url).map_err(|e| format!("cannot create the HTTP client: {}", e))?;

    app_state.logger.info("Starting audio recording...");
    app_state.logger.info(format!(
        "Translating to: {}",
        config.translation.target_language
    ));

    // The audio callback reads them from its first buffer
    app_state.audio.params.update(&config.audio);
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

    let services = ProcessingServices::new(&config, &data_dir, &app_state.logger);
    // Initialize the shared cost from the loaded value
    app_state.set_total_cost(services.price_estimator.total_cost);

    let typing_indicator = TypingIndicator::new(Arc::clone(&socket), app_state.logger.clone());
    let chatbox = Chatbox::new(socket);
    let mut pipeline = Pipeline::new(
        Arc::clone(&app_state),
        api,
        chatbox,
        typing_indicator,
        services,
    );

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
                        apply_enabled(enabled, &config, &pipeline.typing_indicator, &app_state.logger).await;
                    }
                    // The only place where the settings change. The loop
                    // handles a command between utterances, so each
                    // utterance uses one set of settings from start to end.
                    Some(AppCommand::UpdateConfig(new_config)) => {
                        app_state.logger.info("Config updated");
                        if osc_destination(&new_config) != osc_destination(&config) {
                            // Typing can be on at the old destination, and
                            // the StopRecording goes to the new one
                            pipeline.typing_indicator.stop_typing(&config).await;
                        }
                        let new_socket_address = osc_socket_address(&new_config);
                        if new_socket_address != socket_address {
                            match rebind_osc_socket(&socket_address, &new_socket_address, &app_state).await {
                                ControlFlow::Continue(Some(socket)) => {
                                    pipeline.set_socket(socket);
                                    socket_address = new_socket_address;
                                }
                                ControlFlow::Continue(None) => {}
                                ControlFlow::Break(()) => break,
                            }
                        }
                        app_state.audio.params.update(&new_config.audio);
                        pipeline.services.apply_config(&new_config, &app_state.logger);
                        config = new_config;
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
                    None => {
                        // The command channel only closes when the GUI side
                        // is gone (the sender is dropped), so shut down.
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
                // on after Pipeline::process turns it off.
                let recording_goes_on = matches!(event, AudioEvent::AudioPart(_));
                let (audio, extent) = match event {
                    AudioEvent::StartRecording => {
                        pipeline.typing_indicator.start_typing(&config).await;
                        continue;
                    }
                    AudioEvent::StopRecording => {
                        pipeline.typing_indicator.stop_typing(&config).await;
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
                // Shutdown drops the work, including the chatbox
                // display pause and rate limiter wait.
                let result = app_state
                    .shutdown
                    .run_until(pipeline.process(audio_data, extent, &config))
                    .await;
                match result {
                    Some(Ok(())) => {}
                    Some(Err(e)) => log_processing_error(&*e, &app_state.logger),
                    None => break,
                }
                if recording_goes_on {
                    pipeline.typing_indicator.start_typing(&config).await;
                }
            }
        }
    }

    // Shutdown can stop processing between StartRecording and the end of
    // Pipeline::process. Do not leave VRChat showing the typing indicator.
    pipeline.typing_indicator.stop_typing(&config).await;

    Ok(())
}

/// The local address of the OSC socket in `config`, which the loop sends
/// from.
fn osc_socket_address(config: &Config) -> String {
    format!("{}:{}", config.osc.address, config.osc.input_port)
}

/// Where the loop sends the chatbox messages and the typing indicator.
fn osc_destination(config: &Config) -> (&str, u16) {
    (&config.osc.address, config.osc.output_port)
}

/// Open an OSC socket at `new_address` to use in place of the socket at
/// `old_address`. Returns the new socket, or `None` if it cannot be
/// opened: the error goes to the activity log and the old socket stays in
/// use. Breaks if shutdown is requested first, as a host name in the
/// address can wait for DNS.
async fn rebind_osc_socket(
    old_address: &str,
    new_address: &str,
    app_state: &AppState,
) -> ControlFlow<(), Option<Arc<UdpSocket>>> {
    let logger = &app_state.logger;
    let Some(bound) = app_state
        .shutdown
        .run_until(UdpSocket::bind(new_address))
        .await
    else {
        return ControlFlow::Break(());
    };
    match bound {
        Ok(socket) => {
            logger.info(format!("OSC port is now {}", new_address));
            ControlFlow::Continue(Some(Arc::new(socket)))
        }
        Err(e) => {
            logger.error(format!(
                "Cannot open OSC port {}: {}. The old port {} stays in use.",
                new_address, e, old_address
            ));
            ControlFlow::Continue(None)
        }
    }
}

/// Log an error of `Pipeline::process`. An error of the API shows its message
/// for the user in the activity log, and its details on stderr; any other
/// error shows as it is.
pub(crate) fn log_processing_error(error: &(dyn Error + 'static), logger: &Logger) {
    match error.downcast_ref::<ApiError>() {
        Some(api_error) => logger.error_with_details(api_error.to_string(), &api_error.details()),
        None => logger.error(format!("Error: {}", error)),
    }
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
    audio: Arc<AudioShared>,
    channels: u16,
    sample_rate: u32,
    /// When the running test recording reaches the limit
    deadline: Option<Instant>,
}

impl TestRecording {
    /// Test recordings of the input stream with this format.
    pub fn new(app_state: &AppState, channels: u16, sample_rate: u32) -> Self {
        Self {
            audio: Arc::clone(&app_state.audio),
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
        self.audio.test_mode.store(true, Ordering::SeqCst);
    }

    /// Stop the test recording and return what it recorded. Returns `None`
    /// if no test recording runs.
    pub fn stop(&mut self) -> Option<CapturedAudio> {
        self.deadline.take()?;
        self.audio.test_mode.store(false, Ordering::SeqCst);
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
        self.audio
            .test_buffer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Handle the translation toggle from the GUI.
///
/// Disabling turns the typing indicator at the address in `config` off.
/// While translation is off the loop ignores audio events, so the
/// StopRecording of an utterance that started before would not turn it
/// off. The GUI stores `enabled` before it sends this command, so no
/// StartRecording handled after this can turn the indicator on again.
pub async fn apply_enabled(
    enabled: bool,
    config: &Config,
    typing_indicator: &TypingIndicator,
    logger: &Logger,
) {
    if enabled {
        logger.info("Translation enabled");
    } else {
        typing_indicator.stop_typing(config).await;
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
    use crate::api_client::test_server::{Response, TestServer};
    use crate::app_state::{LogEntry, LogLevel};
    use crate::models;
    use crate::price_estimator::{PriceEstimator, TokenCounts};
    use rosc::{OscPacket, OscType};
    use std::future::Future;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Mutex;

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
        /// The settings that the loop starts with
        config: Config,
        cmd_rx: mpsc::Receiver<AppCommand>,
        /// The API that the loop sends its requests to
        api_base_url: String,
        input: AudioInput,
        output: FakeOutput,
        audio_started: Arc<AtomicBool>,
        /// Runs when the loop starts the audio input
        on_audio_start: Option<Box<dyn FnOnce()>>,
        ended: oneshot::Sender<()>,
    }

    impl LoopUnderTest {
        /// Run the loop together with `test` until both end. Returns the
        /// result of the loop.
        async fn run_with(self, test: impl Future<Output = ()>) -> Result<(), String> {
            let Self {
                app_state,
                config,
                cmd_rx,
                api_base_url,
                input,
                output,
                audio_started,
                on_audio_start,
                ended,
            } = self;
            let data_dir = DataDir::new(
                std::env::temp_dir()
                    .join(format!("babble_boop_no_data_dir_{}", std::process::id())),
            );
            let processing = async {
                let result = run_processing_loop(
                    app_state,
                    config,
                    cmd_rx,
                    data_dir,
                    &api_base_url,
                    move || {
                        audio_started.store(true, Ordering::SeqCst);
                        if let Some(on_audio_start) = on_audio_start {
                            on_audio_start();
                        }
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
        /// The settings that the loop starts with
        config: Config,
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
        let app_state = Arc::new(AppState::new(cmd_tx.clone(), log_tx));
        let (events_tx, events) = mpsc::channel(100);
        let (started_tx, started) = oneshot::channel();
        let (played_tx, played) = mpsc::unbounded_channel();
        let playbacks_alive = Arc::new(AtomicUsize::new(0));
        let audio_started = Arc::new(AtomicBool::new(false));
        let (ended_tx, ended) = oneshot::channel();
        let processing = LoopUnderTest {
            app_state: Arc::clone(&app_state),
            config: config.clone(),
            cmd_rx,
            // Port 1 of the local host, where nothing listens, so a request
            // fails at once. A test with a server sets its URL.
            api_base_url: "http://127.0.0.1:1/v1".to_string(),
            input: AudioInput { started, events },
            output: FakeOutput {
                played: played_tx,
                alive: Arc::clone(&playbacks_alive),
            },
            audio_started: Arc::clone(&audio_started),
            on_audio_start: None,
            ended: ended_tx,
        };
        let driver = Driver {
            app_state,
            config,
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
        /// lines before it. Returns the line.
        async fn logged(&mut self, text: &str) -> LogEntry {
            let mut seen = Vec::new();
            let found = tokio::time::timeout(self.wait, async {
                while let Some(entry) = self.log.recv().await {
                    if entry.message.contains(text) {
                        return Some(entry);
                    }
                    seen.push(entry.message);
                }
                None
            })
            .await;
            match found {
                Ok(Some(entry)) => entry,
                _ => panic!("{:?} not logged after {:?}", text, seen),
            }
        }

        /// The next message that VRChat receives.
        async fn received(&self) -> Osc {
            self.received_on(&self.vrchat).await.0
        }

        /// The next message that `vrchat` receives, and the address that
        /// sent it.
        async fn received_on(&self, vrchat: &UdpSocket) -> (Osc, SocketAddr) {
            let mut buf = [0u8; 1024];
            let (len, sender) = tokio::time::timeout(self.wait, vrchat.recv_from(&mut buf))
                .await
                .expect("VRChat received no message")
                .unwrap();
            let OscPacket::Message(message) = rosc::decoder::decode_udp(&buf[..len]).unwrap().1
            else {
                panic!("VRChat received a bundle");
            };
            let osc = match (message.addr.as_str(), message.args.first()) {
                ("/chatbox/typing", Some(OscType::Bool(typing))) => Osc::Typing(*typing),
                ("/chatbox/input", Some(OscType::String(text))) => Osc::Input(text.clone()),
                _ => panic!("unexpected OSC message {:?}", message),
            };
            (osc, sender)
        }

        /// Send the settings of the start with the changes of `change`.
        async fn update_config(&self, change: impl FnOnce(&mut Config)) {
            let mut config = self.config.clone();
            change(&mut config);
            self.command(AppCommand::UpdateConfig(config)).await;
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
                // On at the start, off when Pipeline::process skips the
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
                    .audio
                    .test_buffer
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
                    .audio
                    .test_buffer
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
                // for the rate limiter in Pipeline::process.
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

    /// The translation that the API server of the tests returns, and the
    /// tokens that it reports.
    const TRANSLATION: &str = "Bonjour";
    const TOKENS: TokenCounts = TokenCounts {
        input: 30,
        output: 2,
    };

    /// A server that transcribes every recording as "Hello" and translates
    /// every text as `TRANSLATION`.
    async fn translating_server() -> TestServer {
        let chat_completion = serde_json::json!({
            "choices": [{
                "message": { "content": TRANSLATION },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": TOKENS.input,
                "completion_tokens": TOKENS.output
            }
        });
        TestServer::start(vec![
            ("audio/transcriptions", Response::ok(r#"{"text": "Hello"}"#)),
            (
                "chat/completions",
                Response::ok(chat_completion.to_string()),
            ),
        ])
        .await
    }

    /// A processing loop that sends its requests to `server`, with no
    /// minimum duration and no display pause of the chatbox.
    async fn processing_loop_with_api(server: &TestServer) -> (LoopUnderTest, Driver) {
        let (mut processing, driver) = processing_loop(|config| {
            config.openai.api_key = "sk-test".to_string();
            config.audio.min_transcription_duration = 0.0;
            config.osc.display_time = 0;
        })
        .await;
        processing.api_base_url = server.base_url.clone();
        (processing, driver)
    }

    /// The cost of `utterances` recordings of `sound(seconds)` that are
    /// transcribed and translated by `translating_server`.
    fn translated_cost(utterances: u32, seconds: f32) -> f64 {
        let config = Config::default();
        let prices = PriceEstimator::new(
            PathBuf::new(),
            &config.openai.model,
            &config.openai.transcription_model,
        );
        let wav = encode_upload_wav(&sound(seconds)).unwrap();
        let duration = crate::pipeline::calculate_audio_duration(&wav).unwrap();
        let one =
            prices.estimate_transcription_cost(duration) + prices.estimate_translation_cost(TOKENS);
        (0..utterances).fold(0.0, |total, _| total + one)
    }

    /// The paths of the next `count` requests to `server`.
    async fn requested_paths(server: &mut TestServer, count: usize) -> Vec<String> {
        let mut paths = Vec::new();
        for _ in 0..count {
            paths.push(server.request().await.path);
        }
        paths
    }

    #[tokio::test]
    async fn test_an_utterance_is_transcribed_translated_and_sent_to_the_chatbox() {
        let mut server = translating_server().await;
        let (processing, mut d) = processing_loop_with_api(&server).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                d.event(AudioEvent::StopRecording).await;
                assert_eq!(d.received().await, Osc::Typing(true));
                assert_eq!(d.received().await, Osc::Input(TRANSLATION.to_string()));
                // Off when the translation is sent, off at the stop
                assert_eq!(d.received().await, Osc::Typing(false));
                assert_eq!(d.received().await, Osc::Typing(false));
                d.logged("Transcription: Hello").await;
                d.logged("Translation: Bonjour").await;
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.received().await, Osc::Typing(false));
        assert_eq!(d.app_state.get_total_cost(), translated_cost(1, 1.0));
        assert!(translated_cost(1, 1.0) > 0.0);
        assert_eq!(
            requested_paths(&mut server, 2).await,
            ["/v1/audio/transcriptions", "/v1/chat/completions"]
        );
        assert_eq!(
            server.received().len(),
            0,
            "more requests than one utterance needs"
        );
    }

    #[tokio::test]
    async fn test_a_part_of_a_long_recording_is_sent_and_typing_goes_on() {
        let mut server = translating_server().await;
        let (processing, mut d) = processing_loop_with_api(&server).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                d.event(AudioEvent::AudioPart(sound(1.0))).await;
                assert_eq!(d.received().await, Osc::Typing(true));
                assert_eq!(d.received().await, Osc::Input(TRANSLATION.to_string()));
                assert_eq!(d.received().await, Osc::Typing(false));
                // The recording goes on after its part
                assert_eq!(d.received().await, Osc::Typing(true));
                d.logged("processing it while recording goes on").await;

                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                d.event(AudioEvent::StopRecording).await;
                assert_eq!(d.received().await, Osc::Input(TRANSLATION.to_string()));
                assert_eq!(d.received().await, Osc::Typing(false));
                assert_eq!(d.received().await, Osc::Typing(false));
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.received().await, Osc::Typing(false));
        assert_eq!(d.app_state.get_total_cost(), translated_cost(2, 1.0));
        assert_eq!(
            requested_paths(&mut server, 4).await,
            [
                "/v1/audio/transcriptions",
                "/v1/chat/completions",
                "/v1/audio/transcriptions",
                "/v1/chat/completions"
            ]
        );
    }

    #[tokio::test]
    async fn test_an_api_error_shows_its_message_and_sends_nothing_to_the_chatbox() {
        let body = serde_json::json!({
            "error": {
                "message": "Incorrect API key provided: sk-test.",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        });
        let mut server = TestServer::start(vec![(
            "audio/transcriptions",
            Response::error(reqwest::StatusCode::UNAUTHORIZED, body.to_string()),
        )])
        .await;
        let (processing, mut d) = processing_loop_with_api(&server).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                d.event(AudioEvent::StopRecording).await;
                let error = d.logged("API key").await;
                assert_eq!(
                    (error.level, error.message.as_str()),
                    (
                        LogLevel::Error,
                        "Invalid API key. Check your OpenAI API key in settings."
                    )
                );
                // On at the start, off at the stop, and no chatbox message
                // between them
                assert_eq!(d.received().await, Osc::Typing(true));
                assert_eq!(d.received().await, Osc::Typing(false));
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.received().await, Osc::Typing(false));
        assert_eq!(d.app_state.get_total_cost(), 0.0);
        assert_eq!(
            requested_paths(&mut server, 1).await,
            ["/v1/audio/transcriptions"]
        );
        assert_eq!(server.received().len(), 0, "a translation was requested");
    }

    #[tokio::test]
    async fn test_an_error_that_is_not_from_the_api_is_logged_as_it_is() {
        // The body is not a transcription
        let mut server = TestServer::start(vec![(
            "audio/transcriptions",
            Response::ok(r#"{"text": null}"#),
        )])
        .await;
        let (processing, mut d) = processing_loop_with_api(&server).await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                let error = d.logged("Error: ").await;
                assert_eq!(error.level, LogLevel::Error);
                assert!(
                    error.message.starts_with("Error: invalid type: null"),
                    "{}",
                    error.message
                );
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(
            requested_paths(&mut server, 1).await,
            ["/v1/audio/transcriptions"]
        );
    }

    /// The US dollars for `TOKENS` at the prices of the chat model `model`.
    fn token_cost(model: &str) -> f64 {
        let prices = models::chat_model(model).unwrap();
        (TOKENS.input as f64 * prices.input_price + TOKENS.output as f64 * prices.output_price)
            / 1_000_000.0
    }

    /// The model of each translation request in `requests`.
    fn translation_models(requests: &[crate::api_client::test_server::Request]) -> Vec<String> {
        requests
            .iter()
            .filter(|request| request.path == "/v1/chat/completions")
            .map(|request| {
                let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
                body["model"].as_str().unwrap().to_string()
            })
            .collect()
    }

    #[tokio::test]
    async fn test_a_new_model_applies_to_the_request_and_the_cost_of_the_next_utterance() {
        const NEW_MODEL: &str = "gpt-6-sol";
        let mut server = translating_server().await;
        let (processing, mut d) = processing_loop_with_api(&server).await;
        let old_model = d.config.openai.model.clone();
        assert!(token_cost(NEW_MODEL) > token_cost(&old_model));
        let one_utterance = translated_cost(1, 1.0);
        let mut costs = Vec::new();
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                assert_eq!(d.received().await, Osc::Input(TRANSLATION.to_string()));
                assert_eq!(d.received().await, Osc::Typing(false));
                costs.push(d.app_state.get_total_cost());

                d.update_config(|config| config.openai.model = NEW_MODEL.to_string())
                    .await;
                d.logged("Config updated").await;
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                assert_eq!(d.received().await, Osc::Input(TRANSLATION.to_string()));
                assert_eq!(d.received().await, Osc::Typing(false));
                costs.push(d.app_state.get_total_cost());
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        let mut requests = Vec::new();
        for _ in 0..4 {
            requests.push(server.request().await);
        }
        assert_eq!(
            translation_models(&requests),
            [old_model.as_str(), NEW_MODEL]
        );
        // The second utterance costs its transcription, which did not
        // change, and its tokens at the prices of the new model
        let second_utterance = one_utterance - token_cost(&old_model) + token_cost(NEW_MODEL);
        assert_eq!(costs[0], one_utterance);
        assert!(
            (costs[1] - (one_utterance + second_utterance)).abs() < 1e-12,
            "{:?}, expected {} then {}",
            costs,
            one_utterance,
            one_utterance + second_utterance
        );
    }

    #[tokio::test]
    async fn test_a_new_output_port_moves_typing_and_the_chatbox_to_it() {
        let server = translating_server().await;
        let (processing, mut d) = processing_loop_with_api(&server).await;
        let new_vrchat = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let new_port = new_vrchat.local_addr().unwrap().port();
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                assert_eq!(d.received().await, Osc::Typing(true));

                d.update_config(|config| config.osc.output_port = new_port)
                    .await;
                // Off where it went on
                assert_eq!(d.received().await, Osc::Typing(false));
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                d.event(AudioEvent::StopRecording).await;
                for expected in [
                    Osc::Input(TRANSLATION.to_string()),
                    Osc::Typing(false),
                    Osc::Typing(false),
                ] {
                    assert_eq!(d.received_on(&new_vrchat).await.0, expected);
                }
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(d.received_on(&new_vrchat).await.0, Osc::Typing(false));
        let mut buf = [0u8; 1024];
        assert!(
            d.vrchat.try_recv(&mut buf).is_err(),
            "the old port received a message after the change"
        );
    }

    /// A local UDP port that nothing uses at the time of the call.
    async fn free_port() -> u16 {
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        socket.local_addr().unwrap().port()
    }

    #[tokio::test]
    async fn test_a_new_input_port_opens_a_new_socket_and_closes_the_old_one() {
        let server = translating_server().await;
        let (processing, mut d) = processing_loop_with_api(&server).await;
        let new_port = free_port().await;
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                let (typing, old_socket) = d.received_on(&d.vrchat).await;
                assert_eq!(typing, Osc::Typing(true));

                d.update_config(|config| config.osc.input_port = new_port)
                    .await;
                let line = d.logged("OSC port is now").await;
                assert_eq!(
                    (line.level, line.message),
                    (
                        LogLevel::Info,
                        format!("OSC port is now 127.0.0.1:{}", new_port)
                    )
                );
                // The typing indicator and the chatbox let go of the old
                // socket
                UdpSocket::bind(old_socket).await.unwrap();
                d.event(AudioEvent::AudioData(sound(1.0), Extent::Whole))
                    .await;
                d.event(AudioEvent::StopRecording).await;
                for expected in [
                    Osc::Input(TRANSLATION.to_string()),
                    Osc::Typing(false),
                    Osc::Typing(false),
                ] {
                    let (osc, sender) = d.received_on(&d.vrchat).await;
                    assert_eq!((osc, sender.port()), (expected, new_port));
                }
                // Back to the port of the start
                d.update_config(|_| {}).await;
                d.logged("OSC port is now 127.0.0.1:0").await;
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
    }

    #[tokio::test]
    async fn test_a_busy_input_port_keeps_the_old_socket_until_it_is_free() {
        let (processing, mut d) = processing_loop(|_| {}).await;
        let busy = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let busy_port = busy.local_addr().unwrap().port();
        let result = processing
            .run_with(async {
                d.start_audio();
                d.event(AudioEvent::StartRecording).await;
                let (_, old_socket) = d.received_on(&d.vrchat).await;

                d.update_config(|config| config.osc.input_port = busy_port)
                    .await;
                let error = d.logged("Cannot open OSC port").await;
                assert_eq!(error.level, LogLevel::Error);
                assert!(
                    error
                        .message
                        .starts_with(&format!("Cannot open OSC port 127.0.0.1:{}: ", busy_port))
                        && error
                            .message
                            .ends_with(". The old port 127.0.0.1:0 stays in use."),
                    "{}",
                    error.message
                );
                d.event(AudioEvent::StopRecording).await;
                let (typing, sender) = d.received_on(&d.vrchat).await;
                assert_eq!((typing, sender), (Osc::Typing(false), old_socket));

                // Saved again once the port is free, the settings open it
                drop(busy);
                d.update_config(|config| config.osc.input_port = busy_port)
                    .await;
                d.logged(&format!("OSC port is now 127.0.0.1:{}", busy_port))
                    .await;
                d.event(AudioEvent::StartRecording).await;
                let (typing, sender) = d.received_on(&d.vrchat).await;
                assert_eq!((typing, sender.port()), (Osc::Typing(true), busy_port));
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
    }

    #[tokio::test]
    async fn test_the_audio_settings_apply_before_the_audio_input_starts_and_on_update() {
        let (mut processing, mut d) =
            processing_loop(|config| config.audio.noise_gate_threshold = 0.2).await;
        assert_ne!(Config::default().audio.noise_gate_threshold, 0.2);
        // What the audio callback reads when the input starts
        let threshold_at_start = Arc::new(Mutex::new(None));
        let (app_state, seen) = (Arc::clone(&d.app_state), Arc::clone(&threshold_at_start));
        processing.on_audio_start = Some(Box::new(move || {
            *seen.lock().unwrap() = Some(
                app_state
                    .audio
                    .params
                    .recorder_settings()
                    .noise_gate_threshold,
            );
        }));
        let result = processing
            .run_with(async {
                d.start_audio();
                d.update_config(|config| config.audio.noise_gate_threshold = 0.4)
                    .await;
                // The loop handles commands in order
                d.command(AppCommand::SetEnabled(true)).await;
                d.logged("Translation enabled").await;
                assert_eq!(
                    d.app_state
                        .audio
                        .params
                        .recorder_settings()
                        .noise_gate_threshold,
                    0.4
                );
                d.shut_down().await;
            })
            .await;

        assert_eq!(result, Ok(()));
        assert_eq!(*threshold_at_start.lock().unwrap(), Some(0.2));
    }
}
