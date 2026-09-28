//! Decides when a recording starts and ends from the input level. The audio
//! callback feeds it each buffer; it has no device, clock or channel of its
//! own, so it can be tested with plain sample buffers.

use std::time::Instant;

/// Settings the GUI can change while the stream runs. The callback reads
/// them for each buffer.
#[derive(Clone, Copy, Debug)]
pub struct RecorderSettings {
    /// Peak level above which the noise gate opens
    pub noise_gate_threshold: f32,
    /// Seconds the gate stays open after the level drops below the threshold
    pub noise_gate_hold_time: f32,
    /// Number of buffers with the gate closed that end a recording
    pub silence_threshold: u32,
}

/// What the callback must pass on to the processing side.
#[derive(Debug, PartialEq)]
pub enum RecorderEvent {
    /// The gate opened and a recording started.
    Started,
    /// Silence ended the recording. Holds its samples, interleaved.
    Ended(Vec<f32>),
}

/// State the GUI shows in the audio settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecorderStatus {
    pub is_recording: bool,
    pub silent_frames: u32,
    pub gate_open: bool,
    /// Seconds until the gate closes if the level stays low
    pub hold_remaining: f32,
    /// Seconds since the recording started, 0 when not recording
    pub recording_duration: f32,
}

/// Peak absolute sample value of a buffer.
pub fn peak_level(samples: &[f32]) -> f32 {
    samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max)
}

struct NoiseGate {
    last_active: Instant,
    is_active: bool,
    hold_remaining: f32,
}

impl NoiseGate {
    fn new(now: Instant) -> Self {
        NoiseGate {
            last_active: now,
            is_active: false,
            hold_remaining: 0.0,
        }
    }

    /// Update the gate for a buffer with peak `level`. Returns whether it is open.
    fn process(&mut self, level: f32, settings: &RecorderSettings, now: Instant) -> bool {
        if level > settings.noise_gate_threshold {
            self.last_active = now;
            self.is_active = true;
            self.hold_remaining = 0.0;
        } else if self.is_active {
            let elapsed = now.duration_since(self.last_active).as_secs_f32();
            if elapsed > settings.noise_gate_hold_time {
                self.is_active = false;
                self.hold_remaining = 0.0;
            } else {
                self.hold_remaining = settings.noise_gate_hold_time - elapsed;
            }
        } else {
            self.hold_remaining = 0.0;
        }
        self.is_active
    }
}

pub struct Recorder {
    gate: NoiseGate,
    is_recording: bool,
    silent_frames: u32,
    recording_start: Option<Instant>,
    samples: Vec<f32>,
}

impl Recorder {
    pub fn new(now: Instant) -> Self {
        Recorder {
            gate: NoiseGate::new(now),
            is_recording: false,
            silent_frames: 0,
            recording_start: None,
            samples: Vec::new(),
        }
    }

    /// Process one input buffer received at `now`, and pass each resulting
    /// event to `emit`.
    pub fn process(
        &mut self,
        data: &[f32],
        settings: &RecorderSettings,
        now: Instant,
        mut emit: impl FnMut(RecorderEvent),
    ) {
        if self.gate.process(peak_level(data), settings, now) {
            if !self.is_recording {
                self.is_recording = true;
                self.recording_start = Some(now);
                emit(RecorderEvent::Started);
            }
            self.samples.extend_from_slice(data);
            self.silent_frames = 0;
        } else if self.is_recording {
            self.silent_frames += 1;
            if self.silent_frames >= settings.silence_threshold {
                self.is_recording = false;
                self.silent_frames = 0;
                self.recording_start = None;
                emit(RecorderEvent::Ended(std::mem::take(&mut self.samples)));
            } else {
                // Keep recording during short pauses
                self.samples.extend_from_slice(data);
            }
        }
    }

    pub fn status(&self, now: Instant) -> RecorderStatus {
        RecorderStatus {
            is_recording: self.is_recording,
            silent_frames: self.silent_frames,
            gate_open: self.gate.is_active,
            hold_remaining: self.gate.hold_remaining,
            recording_duration: self
                .recording_start
                .map(|start| now.duration_since(start).as_secs_f32())
                .unwrap_or(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const SETTINGS: RecorderSettings = RecorderSettings {
        noise_gate_threshold: 0.1,
        noise_gate_hold_time: 0.5,
        silence_threshold: 3,
    };
    const LOUD: [f32; 4] = [0.0, 0.5, -0.5, 0.0];
    const QUIET: [f32; 4] = [0.0, 0.05, -0.05, 0.0];
    /// Time between two buffers in these tests
    const BUFFER: Duration = Duration::from_millis(100);

    /// Feeds buffers to a recorder, one `BUFFER` apart, and keeps the events.
    struct Harness {
        recorder: Recorder,
        now: Instant,
        events: Vec<RecorderEvent>,
    }

    impl Harness {
        fn new() -> Self {
            let now = Instant::now();
            Harness {
                recorder: Recorder::new(now),
                now,
                events: Vec::new(),
            }
        }

        fn feed(&mut self, data: &[f32]) -> RecorderStatus {
            self.feed_with(data, &SETTINGS)
        }

        fn feed_with(&mut self, data: &[f32], settings: &RecorderSettings) -> RecorderStatus {
            self.now += BUFFER;
            let events = &mut self.events;
            self.recorder
                .process(data, settings, self.now, |event| events.push(event));
            self.recorder.status(self.now)
        }

        /// Wait until the hold time is over, without feeding buffers.
        fn wait_past_hold(&mut self) {
            self.now += Duration::from_secs_f32(SETTINGS.noise_gate_hold_time) + BUFFER;
        }
    }

    #[test]
    fn test_quiet_input_does_not_start_a_recording() {
        let mut h = Harness::new();
        let status = h.feed(&QUIET);
        assert!(!status.is_recording);
        assert!(!status.gate_open);
        assert!(h.events.is_empty());
    }

    #[test]
    fn test_loud_input_starts_a_recording_once() {
        let mut h = Harness::new();
        h.feed(&LOUD);
        let status = h.feed(&LOUD);
        assert_eq!(h.events, vec![RecorderEvent::Started]);
        assert!(status.is_recording);
        assert!(status.gate_open);
        assert_eq!(status.recording_duration, BUFFER.as_secs_f32());
    }

    #[test]
    fn test_gate_stays_open_for_the_hold_time() {
        let mut h = Harness::new();
        h.feed(&LOUD);
        let status = h.feed(&QUIET);
        assert!(status.gate_open);
        assert_eq!(status.silent_frames, 0);
        let expected_hold = SETTINGS.noise_gate_hold_time - BUFFER.as_secs_f32();
        assert!((status.hold_remaining - expected_hold).abs() < 1e-4);
    }

    #[test]
    fn test_silence_ends_the_recording_with_all_its_samples() {
        let mut h = Harness::new();
        h.feed(&LOUD);
        h.feed(&QUIET); // Inside the hold time: recorded
        h.wait_past_hold();
        let first_silent = h.feed(&QUIET);
        assert!(!first_silent.gate_open);
        assert_eq!(first_silent.silent_frames, 1);
        assert!(first_silent.is_recording);
        h.feed(&QUIET);
        let ended = h.feed(&QUIET);

        // The buffer that reaches the threshold is not recorded; the silent
        // buffers before it are.
        let expected: Vec<f32> = [&LOUD[..], &QUIET, &QUIET, &QUIET].concat();
        assert_eq!(
            h.events,
            vec![RecorderEvent::Started, RecorderEvent::Ended(expected)]
        );
        assert!(!ended.is_recording);
        assert_eq!(ended.silent_frames, 0);
        assert_eq!(ended.recording_duration, 0.0);
    }

    #[test]
    fn test_sound_during_the_silence_count_resets_it() {
        let mut h = Harness::new();
        h.feed(&LOUD);
        h.wait_past_hold();
        h.feed(&QUIET);
        h.feed(&QUIET);
        let status = h.feed(&LOUD);
        assert_eq!(status.silent_frames, 0);
        h.wait_past_hold();
        h.feed(&QUIET);
        h.feed(&QUIET);
        assert_eq!(h.events, vec![RecorderEvent::Started]);
        h.feed(&QUIET);
        assert_eq!(h.events.len(), 2);
    }

    #[test]
    fn test_changed_settings_apply_to_the_next_buffer() {
        let mut h = Harness::new();
        let deaf = RecorderSettings {
            noise_gate_threshold: 0.9,
            ..SETTINGS
        };
        assert!(!h.feed_with(&LOUD, &deaf).is_recording);
        assert!(h.feed(&LOUD).is_recording);
    }

    #[test]
    fn test_a_new_recording_starts_after_one_ends() {
        let mut h = Harness::new();
        h.feed(&LOUD);
        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        h.feed(&LOUD);
        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        let second: Vec<f32> = [&LOUD[..], &QUIET, &QUIET].concat();
        assert_eq!(h.events.len(), 4);
        assert_eq!(h.events[2], RecorderEvent::Started);
        assert_eq!(h.events[3], RecorderEvent::Ended(second));
    }
}
