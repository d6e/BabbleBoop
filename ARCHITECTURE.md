# Architecture

This document describes the code as it stands on this branch. Read it before you change a
thread boundary, a channel, or the shutdown path.

## Threads

| Thread | Owns |
|---|---|
| GUI (main thread) | The `eframe`/`egui` window (`BabbleBoopApp` in `gui.rs`), the config editor state, and sending `AppCommand`s. |
| Processing thread | A tokio runtime (`main.rs`) that runs `run_processing_loop` (`processing_loop.rs`). Owns session state: the current `Config`, the OSC socket, the `Pipeline`, and the test recording state. |
| Audio input thread | Starts the cpal input stream and blocks in `hold_audio_stream` until shutdown. The stream's own real time callback runs `InputHandler::process` (`audio_recording.rs`), which is not this thread but a callback the audio backend invokes. |
| cpal output callback | Plays back a Test Microphone recording. `AudioOutput::play` (`audio_playback.rs`) hands cpal a callback that copies converted samples into the output buffer; a `Stream` from `play` keeps it alive. |
| `spawn_blocking` tasks | Run on the tokio blocking pool, one task per call: WAV encoding (`encode_for_upload`), test recording resampling (`convert_for_playback`), the total cost load at startup (`load_services`) and save after each request (`PriceEstimator::add_cost`), and the debug recording save (`RecordingManager::save_recording`). Each finishes and returns; none is a long lived thread. |

## Data flow of one utterance

1. The audio callback (`InputHandler::process`, `audio_recording.rs`) feeds each buffer through `Recorder`, which applies the noise gate and produces `RecorderEvent`s (`Started`, `LimitReached`, `Ended`).
2. The callback turns each `RecorderEvent` into an `AudioEvent` (`types.rs`) and queues it on the audio event channel with `try_send`.
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
| Atomics in `AudioShared` (`params`, `level`, `test_mode`) | Audio callback (`level`, reads of `test_mode`); processing loop (`params.update`, `test_mode` on test start/stop) | Both | The callback must never block; atomics need no lock. |
| `AudioShared.status: Mutex<RecorderStatus>`, `try_lock`/`try_recv`-style | Audio callback (`publish`, `try_lock`) | GUI and processing loop (`status()`, blocking `lock`) | The callback publishes the whole status as one snapshot so a reader never sees a mix of two buffers; the lock only guards a copy of a `Copy` value, so it skips a publish rather than wait. |
| `AudioShared.test_buffer: Mutex<Vec<f32>>` | Audio callback (`try_lock`, append) | Processing loop (`TestRecording::start`/`stop`, blocking `lock`) | Same reasoning: the callback must not wait to append test samples. |
| Bounded tokio mpsc: audio event channel (`AudioEvent`, capacity 100) | Audio callback, via `try_send` | Processing loop (`AudioEvents::recv`) | Crosses from a real time callback to async code without blocking; a full channel counts the event as dropped and is reported later as `EventsDropped`. |
| Bounded tokio mpsc: command channel (`AppCommand`, capacity 32) | GUI (`send_command`, `try_send`) | Processing loop (`cmd_rx.recv`) | The GUI thread must never block; a failed send surfaces as a status message instead of freezing the window. |
| Bounded tokio mpsc: log channel (`LogEntry`, capacity 100) | `Logger::send` (`try_send`), called from any thread | GUI (`log_rx`) | Same non-blocking rule; the entry is dropped rather than stalling the writer if the channel is full. |
| `oneshot::channel` for audio start (`AudioInput.started`) | Audio input thread, once, after the stream opens or fails | Processing loop (`wait_for_audio_start`) | One value, once: the stream's format or the startup error. |
| `watch::channel` in `Shutdown` | Any thread (`Shutdown::request`) | GUI, processing loop, `Shutdown::run_until`/`requested` | A broadcastable, always-current boolean: every reader sees the latest request, not a queued one. |
| `GuiWaker` (wraps `egui::Context`, `request_repaint`) | `Logger::send`, `AppState::set_total_cost`, `AppState::mark_processing_stopped` | GUI | `egui` repaints only on input or on request; without this a log entry, a cost update, or the processing thread stopping stays invisible until the next mouse move. |

The level meter and the recorder status panel are the exception: they are read continuously with a timed repaint (`request_repaint_after`) in the GUI's own update loop, not through `GuiWaker`.

## Rules for contributors

- The audio callback (`InputHandler::process` and anything it calls) may only touch atomics, `try_send`, and `try_lock`. Never log, wait, print, or touch `egui` from it: any of those can miss the backend's deadline and glitch or drop audio.
- The GUI thread never blocks. Commands to the processing loop go through `try_send`, and a failed send is shown as a status message, not retried in a loop.
- The processing loop owns session state and the `Config`. Only the `AppCommand::UpdateConfig` arm in `run_processing_loop` replaces the config; every utterance runs against the config version it started with.
- Work that can block runs through `tokio::task::spawn_blocking`, not inline on the loop's async task: CPU heavy work (WAV encoding, resampling) and file system access (the total cost load and save, the debug recording save). A slow disk would otherwise stop the loop and its check for shutdown. Nothing checks this rule automatically; it is enforced by code reading.
- Every long `await` in the processing loop is wrapped in `app_state.shutdown.run_until(..)`, or sits directly in the `tokio::select!` alongside `shutdown.requested()`, so a shutdown request cancels it instead of leaving the loop stuck.
- Code outside the GUI that changes something the GUI shows (a log entry, the total cost, the processing-stopped flag) wakes it with `GuiWaker`, since `egui` does not repaint on its own for those changes.
- Shutdown goes only through `Shutdown`: `request()` to ask for it, `is_requested()`/`requested()`/`run_until()` to observe or respect it. No other stop signal exists.

## Testing

Tests live next to the module they exercise, in a `#[cfg(test)] mod tests` block (for example `src/rate_limiter.rs`, `src/recording_manager.rs`). Helpers shared by more than one module's tests live in `src/test_support.rs` (`#[cfg(test)] mod test_support`): `LogCapture`, an `AppState` builder, an OSC receiver, a `TempDir` with `Drop` cleanup, and signal generators. There is no `tests/` integration test directory.

`processing_loop.rs`'s own `#[cfg(test)] mod tests` has a full loop test harness: `processing_loop(configure)` builds a `LoopUnderTest` and a `Driver` around a real `run_processing_loop`, with fakes for every external boundary:

- A fake audio input: an `AudioInput` built from tokio channels the test drives directly (`Driver::events`, `Driver::started`).
- A fake playback output (`FakeOutput`/`FakePlayback`) that records what it was asked to play instead of opening a real device.
- A local UDP socket standing in for VRChat, read with `test_support::recv_osc`.
- A local OpenAI test server (`api_client::test_server::TestServer`), reached through the loop's `api_base_url` parameter, so transcription and translation calls in a test never reach the real API.

Gate command, run before every commit: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`.

Some tests and helpers are built only on Unix or Linux, so also run `cargo clippy --target x86_64-pc-windows-gnu --all-targets -- -D warnings` (after `rustup target add x86_64-pc-windows-gnu`) to check the Windows build; CI runs the same clippy on Windows.

## Where files live at run time

BabbleBoop keeps `config.toml`, `total_cost.txt`, and the `recordings` folder in one data folder chosen at startup; see [Where BabbleBoop keeps its files](README.md#where-babbleboop-keeps-its-files) in the README for the lookup order.
