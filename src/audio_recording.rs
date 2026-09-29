use crate::app_state::{panic_reason, AudioShared};
use crate::recorder::{peak_level, Recorder, RecorderEvent};
use crate::stream_errors::StreamErrorReporter;
use crate::types::{AudioEvent, CapturedAudio};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use std::error::Error;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

/// Queues events for the processing loop without blocking.
struct EventQueue {
    tx: mpsc::Sender<AudioEvent>,
    /// Events that did not fit in the channel and are not reported yet
    dropped: u32,
}

impl EventQueue {
    /// Queue an event, or count it as dropped if the channel is full.
    fn send(&mut self, event: AudioEvent) {
        if self.tx.try_send(event).is_err() {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    /// Report the dropped events if the channel has room again.
    fn report_dropped(&mut self) {
        if self.dropped > 0
            && self
                .tx
                .try_send(AudioEvent::EventsDropped(self.dropped))
                .is_ok()
        {
            self.dropped = 0;
        }
    }
}

/// Handles the input buffers of one stream, converted to f32.
///
/// This runs on the audio thread, which can be real time. It updates
/// atomics, copies samples and queues events with `try_send`. It publishes
/// the recorder status if its lock is free (`try_lock`); while the GUI
/// copies the status, the publish is skipped and the next buffer publishes
/// again. In test mode it copies the samples into the test buffer if its
/// lock is free (`try_lock`). Logging and encoding happen on the processing
/// side when it receives the events.
///
/// The callback can still allocate, free memory or take a lock:
/// - `Recorder` allocates a 5 s buffer when it sends a part, and when a
///   recording starts with no buffer (the first recording, or one after a
///   recording that ended with sound). During speech longer than 5 s it
///   doubles the buffer (10, 20, then 30 s), which copies the samples: at
///   most 3 times in a part. After a quiet part or a recording that ended
///   without sound, it frees a buffer that grew and allocates a new 5 s one.
///   It does the same when test mode starts during a recording and the
///   recorder discards it.
/// - `build_input_stream` grows its f32 buffer on the first callback and
///   when the backend delivers a larger buffer than before.
/// - `try_send` can allocate a new block of the channel list (tokio 1.48.0,
///   `src/sync/mpsc/list.rs` line 134 calls `Block::grow`,
///   `src/sync/mpsc/block.rs` line 351). Waking the processing loop locks
///   a mutex if its thread is parked (`src/runtime/park.rs` line 202; the
///   loop runs in `Runtime::block_on` in `main.rs`).
/// - An event that `try_send` rejects is dropped here, and so are its
///   samples.
/// - A `try_lock` of the status or the test buffer does not wait. But if
///   another thread starts to wait for that lock, the unlock here wakes it
///   with a system call (Rust 1.97 on Linux and Windows,
///   `library/std/src/sys/sync/mutex/futex.rs` lines 89 to 95).
/// - `PanicGuard` formats a crash report after a panic.
struct InputHandler {
    shared: Arc<AudioShared>,
    recorder: Recorder,
    events: EventQueue,
    channels: u16,
    sample_rate: u32,
    /// Whether the last buffer was in test mode
    in_test_mode: bool,
}

impl InputHandler {
    fn new(
        shared: Arc<AudioShared>,
        tx: mpsc::Sender<AudioEvent>,
        channels: u16,
        sample_rate: u32,
        now: Instant,
    ) -> Self {
        Self {
            shared,
            recorder: Recorder::for_stream(now, channels, sample_rate),
            events: EventQueue { tx, dropped: 0 },
            channels,
            sample_rate,
            in_test_mode: false,
        }
    }

    fn process(&mut self, data: &[f32], now: Instant) {
        self.events.report_dropped();
        self.shared.level.store(peak_level(data));

        // If test mode is active, write raw samples to the test buffer and skip normal processing
        if self.shared.test_mode.load(Ordering::Relaxed) {
            if !self.in_test_mode {
                self.in_test_mode = true;
                // The recorder does not run during the test. A recording
                // that went on after it would join the audio before and
                // after the test.
                if self.recorder.discard() {
                    self.events.send(AudioEvent::RecordingDiscarded);
                    self.events.send(AudioEvent::StopRecording);
                }
            }
            // On each buffer, as the publish skips while the lock is held
            self.shared.publish(self.recorder.status(now));
            // The processing side holds the lock only to swap the buffer.
            // If it does so now, this buffer is lost; waiting could make the
            // audio thread miss its deadline.
            if let Ok(mut buffer) = self.shared.test_buffer.try_lock() {
                append_within_capacity(&mut buffer, data, self.channels);
            }
            return;
        }
        self.in_test_mode = false;

        let settings = self.shared.params.recorder_settings();
        let Self {
            recorder,
            events,
            channels,
            sample_rate,
            ..
        } = self;
        let captured = |samples| CapturedAudio {
            samples,
            channels: *channels,
            sample_rate: *sample_rate,
        };
        recorder.process(data, &settings, now, |event| match event {
            RecorderEvent::Started => events.send(AudioEvent::StartRecording),
            RecorderEvent::LimitReached(samples) => {
                events.send(AudioEvent::AudioPart(captured(samples)))
            }
            RecorderEvent::Ended(samples, extent) => {
                if !samples.is_empty() {
                    events.send(AudioEvent::AudioData(captured(samples), extent));
                }
                events.send(AudioEvent::StopRecording);
            }
        });
        self.shared.publish(self.recorder.status(now));
    }
}

/// Append the whole frames of `data` that fit in the capacity of `buffer`.
/// The processing side reserves the capacity when a test recording starts,
/// so the audio thread does not allocate and the buffer has a fixed limit.
fn append_within_capacity(buffer: &mut Vec<f32>, data: &[f32], channels: u16) {
    let room = buffer.capacity() - buffer.len();
    let room = room - room % usize::from(channels.max(1));
    buffer.extend_from_slice(data.get(..room).unwrap_or(data));
}

/// Sample formats the input stream accepts.
trait InputSample: cpal::SizedSample {
    fn to_f32(self) -> f32;
}

impl InputSample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl InputSample for i16 {
    fn to_f32(self) -> f32 {
        self as f32 / i16::MAX as f32
    }
}

/// Runs the audio callback and catches a panic in it. A panic would
/// otherwise end the backend's audio thread with a message on stderr only
/// (ALSA, WASAPI), or abort the program where the backend calls the
/// callback through an `extern "C"` function (CoreAudio; Rust aborts on a
/// panic that unwinds out of one since 1.81).
struct PanicGuard {
    tx: mpsc::Sender<AudioEvent>,
    failed: bool,
    /// The crash report while it does not fit in the channel
    unsent_report: Option<String>,
}

impl PanicGuard {
    fn new(tx: mpsc::Sender<AudioEvent>) -> Self {
        Self {
            tx,
            failed: false,
            unsent_report: None,
        }
    }

    /// Run `body` unless an earlier run panicked. On a panic, report it
    /// to the processing side; later runs then do nothing, as the state the
    /// body left can be inconsistent, except to send the report again
    /// while it does not fit in the channel.
    fn run(&mut self, body: impl FnOnce()) {
        if self.failed {
            self.send_report();
            return;
        }
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
            self.failed = true;
            self.unsent_report = Some(format!(
                "Audio input crashed: {}. Restart BabbleBoop to record again.",
                panic_reason(&*payload)
            ));
            self.send_report();
        }
    }

    /// Send the crash report. Keep it if the channel is full; drop it if
    /// the processing side is gone.
    fn send_report(&mut self) {
        let Some(message) = self.unsent_report.take() else {
            return;
        };
        if let Err(TrySendError::Full(AudioEvent::InputError(message))) =
            self.tx.try_send(AudioEvent::InputError(message))
        {
            self.unsent_report = Some(message);
        }
    }
}

/// Reports the stream errors of the input stream as `InputError` events.
/// On WASAPI cpal 0.15.3 calls the error callback once and then ends the
/// stream thread, which drops this reporter (`run_input` in
/// `src/host/wasapi/stream.rs`). An error that did not fit is then lost,
/// and `AudioEvents` in the processing loop reports the closed channel.
fn input_error_reporter(tx: mpsc::Sender<AudioEvent>) -> StreamErrorReporter<AudioEvent> {
    StreamErrorReporter::new(tx, "Audio input error", AudioEvent::InputError)
}

fn build_input_stream<T: InputSample>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut handler: InputHandler,
    tx: mpsc::Sender<AudioEvent>,
) -> Result<Stream, cpal::BuildStreamError> {
    let mut samples = Vec::new();
    let mut guard = PanicGuard::new(tx.clone());
    let mut errors = input_error_reporter(tx);
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            guard.run(|| {
                samples.clear();
                samples.extend(data.iter().map(|&s| s.to_f32()));
                handler.process(&samples, Instant::now());
            });
        },
        move |err| errors.report(err),
        None,
    )
}

/// Information about the audio stream configuration
pub struct AudioStreamInfo {
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn start_audio_recording(
    shared: Arc<AudioShared>,
    tx: mpsc::Sender<AudioEvent>,
) -> Result<(Stream, AudioStreamInfo), Box<dyn Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let device_config = device.default_input_config()?;

    let sample_format = device_config.sample_format();
    let stream_info = AudioStreamInfo {
        sample_rate: device_config.sample_rate().0,
        channels: device_config.channels(),
    };

    let handler = InputHandler::new(
        shared,
        tx.clone(),
        stream_info.channels,
        stream_info.sample_rate,
        Instant::now(),
    );
    let config = device_config.config();
    let stream: Stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(&device, &config, handler, tx)?,
        cpal::SampleFormat::I16 => build_input_stream::<i16>(&device, &config, handler, tx)?,
        _ => return Err(format!("Unsupported sample format: {:?}", sample_format).into()),
    };

    stream.play()?;

    Ok((stream, stream_info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{AppCommand, AppState};
    use crate::config::AudioConfig;
    use crate::recorder::RecorderStatus;
    use crate::test_support::{check_against_minimum, MinimumCheck};
    use crate::types::{CapturedAudio, Extent};
    use std::time::Duration;

    const LOUD: [f32; 4] = [0.0, 0.5, -0.5, 0.0];
    const QUIET: [f32; 4] = [0.0, 0.05, -0.05, 0.0];
    /// Two buffers of `QUIET`: 4 stereo frames at 48 kHz
    const TWO_QUIET_BUFFERS: f32 = 4.0 / 48_000.0;

    struct Setup {
        app_state: AppState,
        handler: InputHandler,
        rx: mpsc::Receiver<AudioEvent>,
        now: Instant,
        // Keep the command and log channels open
        _channels: (
            mpsc::Receiver<AppCommand>,
            mpsc::Receiver<crate::app_state::LogEntry>,
        ),
    }

    impl Setup {
        /// A stereo 48 kHz handler whose recording ends after two quiet
        /// buffers, with events queued in a channel of `capacity`.
        fn new(capacity: usize) -> Self {
            let (cmd_tx, cmd_rx) = mpsc::channel(1);
            let (log_tx, log_rx) = mpsc::channel(10);
            let app_state = AppState::new(cmd_tx, log_tx);
            app_state.audio.params.update(&AudioConfig {
                silence_duration: TWO_QUIET_BUFFERS,
                noise_gate_threshold: 0.1,
                noise_gate_hold_time: 0.0,
                min_transcription_duration: 0.0,
            });
            let (tx, rx) = mpsc::channel(capacity);
            let now = Instant::now();
            let handler = InputHandler::new(Arc::clone(&app_state.audio), tx, 2, 48_000, now);
            Setup {
                app_state,
                handler,
                rx,
                now,
                _channels: (cmd_rx, log_rx),
            }
        }

        fn feed(&mut self, data: &[f32]) {
            self.now += Duration::from_millis(10);
            self.handler.process(data, self.now);
        }

        /// Turn on test mode with room for `capacity` samples, as the
        /// processing side does.
        fn start_test_mode(&self, capacity: usize) {
            *self.app_state.audio.test_buffer.lock().unwrap() = Vec::with_capacity(capacity);
            self.app_state
                .audio
                .test_mode
                .store(true, Ordering::Relaxed);
        }

        fn test_buffer(&self) -> Vec<f32> {
            self.app_state.audio.test_buffer.lock().unwrap().clone()
        }

        /// Feed `data` on another thread while this thread holds the
        /// status lock, as the GUI does while it copies the status.
        /// Returns whether the callback returned within 5 s.
        fn feed_while_status_held(&mut self, data: &[f32]) -> bool {
            self.now += Duration::from_millis(10);
            let (handler, audio, now) = (&mut self.handler, &self.app_state.audio, self.now);
            let held = audio.hold_status();
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            std::thread::scope(|scope| {
                scope.spawn(move || {
                    handler.process(data, now);
                    done_tx.send(()).unwrap();
                });
                let returned = done_rx.recv_timeout(Duration::from_secs(5)).is_ok();
                // Let a blocked callback finish so the scope can end
                drop(held);
                returned
            })
        }

        fn events(&mut self) -> Vec<AudioEvent> {
            std::iter::from_fn(|| self.rx.try_recv().ok()).collect()
        }
    }

    #[test]
    fn test_16_bit_input_at_full_scale_is_full_scale_f32() {
        // The input callback gives these values to the input handler, so a
        // 16 bit device reaches the same levels as a 32 bit float device.
        let converted = [0, i16::MAX, -i16::MAX].map(InputSample::to_f32);
        assert_eq!(converted, [0.0, 1.0, -1.0]);
        // i16::MIN has no positive counterpart and is a little below -1.
        let min = i16::MIN.to_f32();
        assert!((min + 1.0).abs() < 1e-4, "i16::MIN gives {}", min);
    }

    #[test]
    fn test_a_recording_reaches_the_processing_side_as_raw_samples() {
        let mut s = Setup::new(10);
        s.feed(&LOUD);
        s.feed(&QUIET);
        s.feed(&QUIET);
        assert_eq!(
            s.events(),
            vec![
                AudioEvent::StartRecording,
                AudioEvent::AudioData(
                    CapturedAudio {
                        samples: [LOUD, QUIET].concat(),
                        channels: 2,
                        sample_rate: 48_000,
                    },
                    Extent::Whole
                ),
                AudioEvent::StopRecording,
            ]
        );
    }

    #[test]
    fn test_recording_state_is_published_for_the_gui() {
        let mut s = Setup::new(10);
        s.feed(&LOUD);
        assert!(s.app_state.audio.status().is_recording);
        assert!(s.app_state.audio.status().gate_open);
        s.feed(&QUIET);
        let quiet_time = s.app_state.audio.status().quiet_time;
        assert_eq!(quiet_time, 2.0 / 48_000.0);
        assert!(!s.app_state.audio.status().gate_open);
        let level = s.app_state.audio.level.load();
        assert_eq!(level, 0.05);
    }

    #[test]
    fn test_a_changed_silence_duration_applies_to_the_recording_in_progress() {
        let mut s = Setup::new(10);
        let mut audio = AudioConfig {
            silence_duration: 1.0,
            noise_gate_threshold: 0.1,
            noise_gate_hold_time: 0.0,
            min_transcription_duration: 0.0,
        };
        s.feed(&LOUD);
        s.app_state.audio.params.update(&audio);
        for _ in 0..3 {
            s.feed(&QUIET);
        }
        assert!(s.app_state.audio.status().is_recording);
        // Saved settings with a shorter duration: the next quiet buffer
        // makes 8 quiet frames, more than 4
        audio.silence_duration = TWO_QUIET_BUFFERS;
        s.app_state.audio.params.update(&audio);
        s.feed(&QUIET);
        assert!(!s.app_state.audio.status().is_recording);
        assert_eq!(s.events().last(), Some(&AudioEvent::StopRecording));
    }

    #[test]
    fn test_test_mode_diverts_the_samples_and_sends_nothing() {
        let mut s = Setup::new(10);
        s.start_test_mode(100);
        s.feed(&LOUD);
        s.feed(&QUIET);
        assert!(s.events().is_empty());
        assert!(!s.app_state.audio.status().is_recording);
        let buffer = s.app_state.audio.test_buffer.lock().unwrap().clone();
        assert_eq!(buffer, [LOUD, QUIET].concat());
        let level = s.app_state.audio.level.load();
        assert_eq!(level, 0.05);
    }

    #[test]
    fn test_test_mode_discards_the_recording_in_progress() {
        let mut s = Setup::new(10);
        s.feed(&LOUD);
        s.feed(&QUIET);
        s.start_test_mode(100);
        s.feed(&LOUD);
        s.feed(&QUIET);
        assert_eq!(
            s.events(),
            vec![
                AudioEvent::StartRecording,
                AudioEvent::RecordingDiscarded,
                AudioEvent::StopRecording,
            ]
        );
        // The GUI shows no recording, and the level meter stays live
        let state = &s.app_state;
        assert!(!state.audio.status().is_recording);
        assert_eq!(state.audio.status().quiet_time, 0.0);
        let duration = state.audio.status().recording_duration;
        assert_eq!(duration, 0.0);
        let level = state.audio.level.load();
        assert_eq!(level, 0.05);

        s.app_state.audio.test_mode.store(false, Ordering::Relaxed);
        s.feed(&QUIET);
        assert!(s.events().is_empty());
        s.feed(&LOUD);
        s.feed(&QUIET);
        s.feed(&QUIET);
        assert_eq!(
            s.events(),
            vec![
                AudioEvent::StartRecording,
                AudioEvent::AudioData(
                    CapturedAudio {
                        samples: [LOUD, QUIET].concat(),
                        channels: 2,
                        sample_rate: 48_000,
                    },
                    Extent::Whole
                ),
                AudioEvent::StopRecording,
            ]
        );
    }

    #[test]
    fn test_a_second_test_mode_discards_the_recording_in_progress_again() {
        let mut s = Setup::new(10);
        for _ in 0..2 {
            s.feed(&LOUD);
            s.start_test_mode(100);
            s.feed(&LOUD);
            s.app_state.audio.test_mode.store(false, Ordering::Relaxed);
        }
        let discarded = || {
            [
                AudioEvent::StartRecording,
                AudioEvent::RecordingDiscarded,
                AudioEvent::StopRecording,
            ]
        };
        assert_eq!(
            s.events(),
            discarded()
                .into_iter()
                .chain(discarded())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_test_mode_stops_adding_samples_when_the_buffer_is_full() {
        let mut s = Setup::new(10);
        s.start_test_mode(8);
        for _ in 0..3 {
            s.feed(&LOUD);
        }
        assert_eq!(s.test_buffer(), [LOUD, LOUD].concat());
        // The callback did not allocate a larger buffer
        assert_eq!(s.app_state.audio.test_buffer.lock().unwrap().capacity(), 8);
    }

    #[test]
    fn test_test_mode_adds_only_whole_frames() {
        let mut s = Setup::new(10);
        // Room for three and a half stereo frames
        s.start_test_mode(7);
        s.feed(&LOUD);
        s.feed(&LOUD);
        assert_eq!(s.test_buffer(), [&LOUD[..], &LOUD[..2]].concat());
    }

    #[test]
    fn test_test_mode_does_not_wait_for_a_locked_buffer() {
        let mut s = Setup::new(10);
        s.start_test_mode(100);
        let locked = s.app_state.audio.test_buffer.lock().unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                s.handler.process(&LOUD, s.now);
                done_tx.send(()).unwrap();
            });
            let returned = done_rx.recv_timeout(Duration::from_secs(5)).is_ok();
            // Let a blocked callback finish so the scope can end
            drop(locked);
            assert!(returned, "the callback waited for the test buffer lock");
        });
        // The samples of that buffer are lost, the next buffer is kept
        s.feed(&QUIET);
        assert_eq!(s.test_buffer(), QUIET.to_vec());
    }

    #[test]
    fn test_the_callback_skips_the_status_while_the_gui_holds_it() {
        let mut s = Setup::new(10);
        assert!(
            s.feed_while_status_held(&LOUD),
            "the callback waited for the status lock"
        );
        // The recording started, but the GUI still has the old status
        assert_eq!(s.events(), vec![AudioEvent::StartRecording]);
        assert_eq!(s.app_state.audio.status(), RecorderStatus::default());
        // The level meter does not need the lock
        assert_eq!(s.app_state.audio.level.load(), 0.5);
        // The next buffer publishes again
        s.feed(&QUIET);
        let status = s.app_state.audio.status();
        assert!(status.is_recording);
        assert_eq!(status.quiet_time, 2.0 / 48_000.0);
    }

    #[test]
    fn test_a_status_skipped_when_test_mode_starts_is_published_later() {
        let mut s = Setup::new(10);
        s.feed(&LOUD);
        s.start_test_mode(100);
        assert!(
            s.feed_while_status_held(&LOUD),
            "the callback waited for the status lock"
        );
        assert!(s.app_state.audio.status().is_recording);
        // The recording was discarded; the next test buffer shows it
        s.feed(&LOUD);
        assert_eq!(s.app_state.audio.status(), RecorderStatus::default());
    }

    #[test]
    fn test_changed_gate_settings_apply_to_the_next_buffer() {
        let mut s = Setup::new(10);
        let mut audio = AudioConfig {
            silence_duration: TWO_QUIET_BUFFERS,
            noise_gate_threshold: 0.6,
            noise_gate_hold_time: 0.0,
            min_transcription_duration: 0.0,
        };
        s.app_state.audio.params.update(&audio);
        s.feed(&LOUD);
        assert!(s.events().is_empty(), "0.5 is below the threshold");

        audio.noise_gate_threshold = 0.1;
        audio.noise_gate_hold_time = 1.0;
        s.app_state.audio.params.update(&audio);
        s.feed(&LOUD);
        assert_eq!(s.events(), vec![AudioEvent::StartRecording]);
        // 10 ms after the loud buffer, the gate holds for 0.99 s more
        s.feed(&QUIET);
        let status = s.app_state.audio.status();
        assert!(status.gate_open);
        assert!(
            (status.hold_remaining - 0.99).abs() < 1e-4,
            "{}",
            status.hold_remaining
        );
    }

    #[test]
    fn test_events_lost_on_a_full_queue_are_reported_once_there_is_room() {
        let mut s = Setup::new(1);
        s.feed(&LOUD); // StartRecording fills the queue
        s.feed(&QUIET);
        s.feed(&QUIET); // AudioData and StopRecording do not fit
        assert_eq!(s.events(), vec![AudioEvent::StartRecording]);
        s.feed(&QUIET);
        assert_eq!(s.events(), vec![AudioEvent::EventsDropped(2)]);
        s.feed(&QUIET);
        assert!(s.events().is_empty());
    }

    #[test]
    fn test_speech_longer_than_30_seconds_is_sent_in_parts() {
        let mut s = Setup::new(10);
        // One second of stereo audio at 48 kHz
        let second = vec![0.5; 2 * 48_000];
        for _ in 0..31 {
            s.feed(&second);
        }
        let events = s.events();
        assert_eq!(events.len(), 2, "{:?}", events.first());
        assert_eq!(events[0], AudioEvent::StartRecording);
        let AudioEvent::AudioPart(part) = &events[1] else {
            panic!("expected AudioPart");
        };
        assert_eq!(part.samples.len(), 30 * 2 * 48_000);
        assert_eq!((part.channels, part.sample_rate), (2, 48_000));
        assert!(s.app_state.audio.status().is_recording);
        // The GUI shows that the minimum does not apply to the rest
        assert!(s.app_state.audio.status().split);
        s.feed(&QUIET);
        s.feed(&QUIET);
        assert!(!s.app_state.audio.status().split);
    }

    #[test]
    fn test_silence_after_a_part_ends_the_recording_without_audio() {
        let mut s = Setup::new(10);
        // One second of stereo audio at 48 kHz
        let second = vec![0.5; 2 * 48_000];
        for _ in 0..30 {
            s.feed(&second);
        }
        s.feed(&QUIET);
        s.feed(&QUIET);
        let events = s.events();
        assert!(
            matches!(
                events.as_slice(),
                [
                    AudioEvent::StartRecording,
                    AudioEvent::AudioPart(_),
                    AudioEvent::StopRecording
                ]
            ),
            "{:?}",
            events.get(2..)
        );
        assert!(!s.app_state.audio.status().is_recording);
    }

    /// The recorded audio that `Pipeline::process` receives after the events
    /// of one recording.
    fn recorded_audio(events: Vec<AudioEvent>) -> (CapturedAudio, Extent) {
        let mut audio = events.into_iter().filter_map(|event| match event {
            AudioEvent::AudioData(audio, extent) => Some((audio, extent)),
            _ => None,
        });
        let recorded = audio.next().expect("no AudioData event");
        assert!(audio.next().is_none(), "more than one AudioData event");
        recorded
    }

    #[tokio::test]
    async fn test_the_end_of_a_long_speech_is_not_skipped_as_too_short() {
        let mut s = Setup::new(10);
        // One second of stereo audio at 48 kHz
        let second = vec![0.5; 2 * 48_000];
        // 31 s of speech: a part of 30 s, then the last second
        for _ in 0..31 {
            s.feed(&second);
        }
        s.feed(&QUIET);
        s.feed(&QUIET);
        let (last, extent) = recorded_audio(s.events());
        assert_eq!(
            check_against_minimum(last, extent, 2.0).await,
            MinimumCheck::Transcribed
        );
    }

    #[tokio::test]
    async fn test_a_short_recording_is_skipped_as_too_short() {
        let mut s = Setup::new(10);
        s.feed(&vec![0.5; 2 * 48_000]);
        s.feed(&QUIET);
        s.feed(&QUIET);
        let (recording, extent) = recorded_audio(s.events());
        assert_eq!(
            check_against_minimum(recording, extent, 2.0).await,
            MinimumCheck::Skipped
        );
    }

    #[test]
    fn test_a_panic_in_the_callback_is_reported_once_and_stops_processing() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut guard = PanicGuard::new(tx);
        guard.run(|| panic!("index out of bounds"));
        let mut ran = false;
        guard.run(|| ran = true);

        assert!(!ran, "the callback ran again after a panic");
        assert_eq!(
            std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>(),
            vec![AudioEvent::InputError(
                "Audio input crashed: index out of bounds. Restart BabbleBoop to record again."
                    .to_string()
            )]
        );
    }

    #[test]
    fn test_a_crash_report_that_did_not_fit_is_sent_once_there_is_room() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(AudioEvent::StartRecording).unwrap();
        let mut guard = PanicGuard::new(tx);
        guard.run(|| panic!("index out of bounds"));
        // The processing side takes the event that filled the channel
        assert_eq!(rx.try_recv().unwrap(), AudioEvent::StartRecording);
        let mut ran = false;
        guard.run(|| ran = true);
        guard.run(|| ran = true);

        assert!(!ran, "the callback ran again after a panic");
        assert_eq!(
            std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>(),
            vec![AudioEvent::InputError(
                "Audio input crashed: index out of bounds. Restart BabbleBoop to record again."
                    .to_string()
            )]
        );
    }

    #[test]
    fn test_the_callback_runs_while_it_does_not_panic() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut guard = PanicGuard::new(tx);
        let mut runs = 0;
        guard.run(|| runs += 1);
        guard.run(|| runs += 1);
        assert_eq!(runs, 2);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_a_repeated_stream_error_is_reported_once() {
        let (tx, mut rx) = mpsc::channel(10);
        let mut reporter = input_error_reporter(tx);
        reporter.report("device unplugged");
        reporter.report("device unplugged");
        reporter.report("buffer overrun");
        reporter.report("device unplugged");
        let messages: Vec<AudioEvent> = std::iter::from_fn(|| rx.try_recv().ok()).collect();
        assert_eq!(
            messages,
            ["device unplugged", "buffer overrun", "device unplugged"]
                .map(|e| AudioEvent::InputError(format!("Audio input error: {}", e)))
        );
    }

    #[test]
    fn test_a_stream_error_that_did_not_fit_is_sent_again() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.try_send(AudioEvent::StartRecording).unwrap();
        let mut reporter = input_error_reporter(tx);
        reporter.report("device unplugged");
        assert_eq!(rx.try_recv().unwrap(), AudioEvent::StartRecording);
        reporter.report("device unplugged");
        assert_eq!(
            rx.try_recv().unwrap(),
            AudioEvent::InputError("Audio input error: device unplugged".to_string())
        );
    }
}
