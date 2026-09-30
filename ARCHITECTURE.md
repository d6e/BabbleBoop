# Architecture

This document describes the code as it stands on this branch. Read it before you change a
thread boundary, a channel, or the shutdown path.

## Threads

| Thread | Owns |
|---|---|
| GUI (main thread) | The `eframe`/`egui` window (`BabbleBoopApp` in `gui.rs`), the config editor state, and sending `AppCommand`s. |
| Processing thread | A tokio runtime (`main.rs`) that runs `run_processing_loop` (`processing_loop.rs`). Owns session state: the current `Config`, the OSC socket, the `Pipeline`, and the test recording state. |
| Audio input thread | Starts the cpal input stream and blocks in `hold_audio_stream` until shutdown is requested or the processing thread ends. The stream's own real time callback runs `InputHandler::process` (`audio_recording.rs`), which is not this thread but a callback the audio backend invokes. |
| cpal output callback | Plays back a Test Microphone recording. `AudioOutput::play` (`audio_playback.rs`) hands cpal a callback that copies converted samples into the output buffer; a `Stream` from `play` keeps it alive. |
| `spawn_blocking` tasks | Run on the tokio blocking pool, one task per call: WAV encoding (`encode_for_upload`), test recording resampling (`convert_for_playback`), the total cost load at startup (`load_services`) and save after each request (`PriceEstimator::add_cost`), and the debug recording save (`RecordingManager::save_recording`). The DNS lookups of a host name (in the OSC address, and the API host) also run there, started by tokio and reqwest. Each finishes and returns; none is a long lived thread. |

## Data flow of one utterance

1. The audio callback (`InputHandler::process`, `audio_recording.rs`) feeds each buffer through `Recorder`, which applies the noise gate and produces `RecorderEvent`s (`Started`, `LimitReached`, `Ended`).
2. The callback turns each `RecorderEvent` into one or two `AudioEvent`s (`types.rs`; `Ended` gives `AudioData` if it has samples, then `StopRecording`) and queues them on the audio event channel with `try_send`.
3. `run_processing_loop` (`processing_loop.rs`) receives the event through `AudioEvents`. `StartRecording` and `StopRecording` just toggle the typing indicator. `AudioData` (a whole recording, or the last part of one that hit the length limit) and `AudioPart` (an earlier part) carry a `CapturedAudio`.
4. The loop encodes the captured samples to a WAV upload on the blocking pool: `encode_for_upload` (`processing_loop.rs`) calls `spawn_blocking` with `encode_upload_wav` (`upload_audio.rs`).
5. The loop calls `Pipeline::process` (`pipeline.rs`), which runs the utterance:
   - Transcribes the upload (`transcription::transcribe_audio`).
   - Adds the request cost to the running total and saves the total to `total_cost.txt` (`accept_transcription`, using `PriceEstimator::add_cost`), off the async task via `spawn_blocking`.
   - Saves the debug recording if enabled (`ProcessingServices::recording_manager`, `RecordingManager::save_recording`), off the async task via `spawn_blocking`.
   - Translates the transcription (`translation::ask_chatgpt`).
   - Adds the translation cost the same way, then delivers the translation to the VRChat chatbox (`deliver_translation`, using `Chatbox`).
6. `Chatbox` and `TypingIndicator` (`chatbox.rs`, `typing_indicator.rs`) send OSC packets over a shared `tokio::net::UdpSocket` to VRChat.

## How threads communicate

| Mechanism | Writer | Reader | Why |
|---|---|---|---|
| Atomics in `AudioShared` (`params`, `level`, `test_mode`) | Audio callback (`level`); processing loop (`params.update`, `test_mode` on test start and stop) | Audio callback (`params`, `test_mode`); GUI (`level`, `test_mode`) | The callback must not wait for another thread; atomics need no lock. |
| `AudioShared.status: Mutex<RecorderStatus>` | Audio callback (`publish`, `try_lock`) | GUI (`status()`, blocking `lock`) | The callback publishes the whole status as one snapshot, so the GUI never shows fields of two different buffers. The callback skips a publish rather than wait for the lock; the GUI holds the lock only to copy the value. |
| `AudioShared.test_buffer: Mutex<Vec<f32>>` | Audio callback (`try_lock`, append) | Processing loop (puts an empty buffer in with `TestRecording::start` and takes the samples out with `stop`, blocking `lock`) | Same reasoning: the callback must not wait to append test samples. |
| Bounded tokio mpsc: audio event channel (`AudioEvent`, capacity 100) | Audio callback, via `try_send` | Processing loop (`AudioEvents::recv`) | Crosses from a real time callback to async code without blocking; a full channel counts the event as dropped and is reported later as `EventsDropped`. The lost events can hold a `StopRecording`, so the loop turns the typing indicator off on `EventsDropped` if it turned it on (`TypingIndicator::is_typing`). |
| Bounded tokio mpsc: command channel (`AppCommand`, capacity 32) | GUI (`send_command`, `try_send`) | Processing loop (`cmd_rx.recv`) | The GUI thread must not wait for the processing loop; a failed send surfaces as a status message instead of freezing the window. |
| Bounded tokio mpsc: log channel (`LogEntry`, capacity 100) | `Logger::send` (`try_send`), called from any thread | GUI (`log_rx`) | Same non-blocking rule; the entry is dropped rather than stalling the writer if the channel is full. |
| `oneshot::channel` for audio start (`AudioInput.started`) | Audio input thread, once, after the stream opens or fails | Processing loop (`wait_for_audio_start`) | One value, once: the stream's format or the startup error. |
| `watch::channel` in `Shutdown` | GUI (`on_exit`) and `main.rs` after `run_gui` returns (`Shutdown::request`) | Processing loop (`requested`, `run_until`), audio input thread (`hold_audio_stream`, `is_requested`) | A broadcastable, always-current boolean: every reader sees the latest request, not a queued one. |
| `GuiWaker` (wraps `egui::Context`, `request_repaint`) | `Logger::send`, `AppState::set_total_cost`, `AppState::mark_processing_stopped` | GUI | `egui` repaints only on input or on request; without this a log entry, a cost update, or the processing thread stopping stays invisible until the next mouse move. |

The level meter, the recorder status panel, and the state of the Test Microphone button (`test_mode`) are the exception: while the Audio Settings section is open, the GUI reads them again after a timed repaint (`request_repaint_after`) in its own update loop, not through `GuiWaker`. `audio_settings_ui` (`gui.rs`) says when a minimized window skips it.

## Rules for contributors

- The audio callback (`InputHandler::process` and anything it calls) shares state with other threads only through atomics, `try_send`, and `try_lock`. Never log, wait, print, or touch `egui` from it: any of those can miss the backend's deadline and glitch or drop audio. The callback still allocates and frees memory, can lock a mutex inside tokio, and can wake a thread with a system call, in known places, such as the recorder buffers, the conversion buffer, the event channel, and the unlock of a lock that another thread waits for. The `InputHandler` doc comment (`audio_recording.rs`) lists all of these places on the normal path. Do not add to that list. A panic on the audio thread, which `PanicGuard` catches, also runs the panic machinery of std on that thread (the default panic hook and the unwinding), which takes locks, allocates memory, and prints the panic message to stderr. The doc comment names the panic path as one item and does not list all that std does in it.
- The GUI thread does not wait for the processing loop. Commands to the processing loop go through `try_send`, and a failed send is shown as a status message, not retried in a loop. While the window is open, the GUI thread blocks in two places: Save Settings writes `config.toml` on the GUI thread (`Config::save`, from `save_config` in `gui.rs`), so a slow disk freezes the window until the write ends, and reading the recorder status (`AudioShared::status`) takes its lock, which the audio callback holds only to copy one value.
- The processing loop owns session state and the `Config`. Only the `AppCommand::UpdateConfig` arm in `run_processing_loop` replaces the config; every utterance runs against the config version it started with.
- Work that can block runs through `tokio::task::spawn_blocking`, not inline on the loop's async task: CPU heavy work (WAV encoding, resampling) and file system access (the total cost load and save, the debug recording save). A slow disk would otherwise stop the loop and its check for shutdown. One exception: `finish_test_recording` opens the output device (`open_output`), starts the playback stream (`PlaybackOutput::play`), and drops the previous playback stream inline. On ALSA and WASAPI that drop joins the stream thread of cpal (cpal 0.15.3, `src/host/alsa/mod.rs` and `src/host/wasapi/stream.rs`, `Drop for Stream`). A slow audio driver stops the loop until those calls return. Nothing checks this rule automatically; it is enforced by code reading.
- Every long `await` in the processing loop is wrapped in `app_state.shutdown.run_until(..)`, or sits directly in the `tokio::select!` alongside `shutdown.requested()`, so a shutdown request cancels it instead of leaving the loop stuck. Exceptions: the typing indicator sends (`start_typing` and `stop_typing` in the arms of the loop and in `apply_enabled`) and the bind of the OSC socket at startup are not wrapped. They give the address as a `host:port` string, and for a host name tokio does a DNS lookup on the blocking pool (tokio 1.48.0, `src/net/addr.rs`), so a slow DNS server holds the loop until the lookup ends. The last `stop_typing`, after the loop, runs after shutdown on purpose, so VRChat does not keep showing the typing indicator.
- Code outside the GUI that changes something the GUI shows (a log entry, the total cost, the processing-stopped flag) wakes it with `GuiWaker`, since `egui` does not repaint on its own for those changes.
- Shutdown goes through `Shutdown`: `request()` to ask for it, `is_requested()`/`requested()`/`run_until()` to observe or respect it. The only other stop signal is the processing stopped flag (`AppState::mark_processing_stopped`), which `main.rs` sets when the processing loop returns or panics, or when its tokio runtime cannot start. `hold_audio_stream` also drops the audio input stream on that flag, so no stream records that nothing reads.

## Testing

Tests live next to the module they exercise, in a `#[cfg(test)] mod tests` block (for example `src/rate_limiter.rs`, `src/recording_manager.rs`). Helpers shared by more than one module's tests live in `src/test_support.rs` (`#[cfg(test)] mod test_support`), among them `LogCapture`, an `AppState` builder, an OSC receiver, a `TempDir` with `Drop` cleanup, and signal generators. There is no `tests/` integration test directory.

`processing_loop.rs`'s own `#[cfg(test)] mod tests` has a full loop test harness: `processing_loop(configure)` builds a `LoopUnderTest` and a `Driver` around a real `run_processing_loop`, with fakes for every external boundary:

- A fake audio input: an `AudioInput` built from tokio channels the test drives directly (`Driver::events`, `Driver::started`).
- A fake playback output (`FakeOutput`/`FakePlayback`) that records what it was asked to play instead of opening a real device.
- A local UDP socket standing in for VRChat, read with `test_support::recv_osc`.
- A local OpenAI test server (`api_client::test_server::TestServer`), reached through the loop's `api_base_url` parameter, so transcription and translation calls in a test never reach the real API.

Gate command, run before every commit: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`.

Some tests and helpers are built only on Unix or Linux, so also run `cargo clippy --target x86_64-pc-windows-gnu --all-targets -- -D warnings` (after `rustup target add x86_64-pc-windows-gnu`) to check the Windows build; CI runs the same clippy on Windows.

## Where files live at run time

BabbleBoop keeps `config.toml`, `total_cost.txt`, and the `recordings` folder in one data folder chosen at startup; see [Where BabbleBoop keeps its files](README.md#where-babbleboop-keeps-its-files) in the README for the lookup order.
