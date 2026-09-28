//! Decides when a recording starts and ends from the input level. The audio
//! callback feeds it each buffer; it has no device, clock or channel of its
//! own, so it can be tested with plain sample buffers.

use crate::types::Extent;
use std::time::{Duration, Instant};

/// Longest recording sent as one upload. Longer speech is sent in parts of
/// this length while recording goes on. The upload of a 30 s part is a
/// 16 kHz mono 16 bit WAV file (`upload_audio`) of about 0.96 MB
/// (30 x 16000 x 2 bytes and a 44 byte header), far below the 25 MB file
/// limit of the transcription API. The first translation of a long speech
/// also appears after 30 s instead of after the speech ends.
pub const MAX_RECORDING: Duration = Duration::from_secs(30);

/// Number of interleaved samples in `MAX_RECORDING`.
pub fn max_recording_samples(channels: u16, sample_rate: u32) -> usize {
    MAX_RECORDING.as_secs() as usize * sample_rate as usize * usize::from(channels)
}

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
    /// Silence ended the recording. Holds its samples since the last part,
    /// interleaved, and whether the recording reached the length limit
    /// before. Holds no samples if none of them is above the gate
    /// threshold.
    Ended(Vec<f32>, Extent),
    /// The recording reached the maximum length. Holds its samples so far;
    /// the recording goes on with a new part. A part with no sample above
    /// the gate threshold is not sent.
    LimitReached(Vec<f32>),
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
    /// Whether the recording reached the length limit. Its last part is
    /// then transcribed whatever its length.
    pub split: bool,
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
    /// Whether `samples` holds a sample above the gate threshold
    has_sound: bool,
    /// Whether the recording reached the length limit
    split: bool,
    max_samples: usize,
}

impl Recorder {
    /// `max_samples` is the length at which a recording is split. It must
    /// be a whole number of frames, so that a split does not fall inside a
    /// frame.
    pub fn new(now: Instant, max_samples: usize) -> Self {
        Recorder {
            gate: NoiseGate::new(now),
            is_recording: false,
            silent_frames: 0,
            recording_start: None,
            samples: Vec::new(),
            has_sound: false,
            split: false,
            max_samples: max_samples.max(1),
        }
    }

    /// Add samples to the recording, and send a part each time it reaches
    /// the maximum length. `loud` tells whether `data` holds a sample above
    /// `threshold`.
    fn record(
        &mut self,
        mut data: &[f32],
        loud: bool,
        threshold: f32,
        now: Instant,
        emit: &mut impl FnMut(RecorderEvent),
    ) {
        while !data.is_empty() {
            let room = self.max_samples - self.samples.len();
            let (part, rest) = data.split_at(room.min(data.len()));
            self.samples.extend_from_slice(part);
            // A split can leave all the loud samples of a buffer in one part
            if loud && !self.has_sound {
                self.has_sound = peak_level(part) > threshold;
            }
            data = rest;
            if self.samples.len() == self.max_samples {
                self.recording_start = Some(now);
                self.split = true;
                if self.has_sound {
                    self.has_sound = false;
                    let next = self.new_buffer();
                    let samples = std::mem::replace(&mut self.samples, next);
                    emit(RecorderEvent::LimitReached(samples));
                } else {
                    self.samples.clear();
                }
            }
        }
    }

    /// Buffer for a new recording. It holds a whole part, so that the audio
    /// callback does not copy the samples each time the buffer grows.
    fn new_buffer(&self) -> Vec<f32> {
        Vec::with_capacity(self.max_samples)
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
        let level = peak_level(data);
        let loud = level > settings.noise_gate_threshold;
        let threshold = settings.noise_gate_threshold;
        if self.gate.process(level, settings, now) {
            if !self.is_recording {
                self.is_recording = true;
                self.recording_start = Some(now);
                // A recording that ended without sound keeps its buffer
                if self.samples.capacity() == 0 {
                    self.samples = self.new_buffer();
                }
                emit(RecorderEvent::Started);
            }
            self.record(data, loud, threshold, now, &mut emit);
            self.silent_frames = 0;
        } else if self.is_recording {
            self.silent_frames += 1;
            if self.silent_frames >= settings.silence_threshold {
                self.is_recording = false;
                self.silent_frames = 0;
                self.recording_start = None;
                let extent = if std::mem::take(&mut self.split) {
                    Extent::Part
                } else {
                    Extent::Whole
                };
                if self.has_sound {
                    self.has_sound = false;
                    emit(RecorderEvent::Ended(
                        std::mem::take(&mut self.samples),
                        extent,
                    ));
                } else {
                    self.samples.clear();
                    emit(RecorderEvent::Ended(Vec::new(), extent));
                }
            } else {
                // Keep recording during short pauses
                self.record(data, loud, threshold, now, &mut emit);
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
            split: self.split,
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
            Self::with_limit(1000)
        }

        /// A recorder that splits recordings at `max_samples`.
        fn with_limit(max_samples: usize) -> Self {
            let now = Instant::now();
            Harness {
                recorder: Recorder::new(now, max_samples),
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
            vec![
                RecorderEvent::Started,
                RecorderEvent::Ended(expected, Extent::Whole)
            ]
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
        assert_eq!(h.events[3], RecorderEvent::Ended(second, Extent::Whole));
    }

    #[test]
    fn test_a_long_recording_is_split_at_the_limit() {
        let mut h = Harness::with_limit(4 * LOUD.len());
        for _ in 0..8 {
            h.feed(&LOUD);
        }
        let status = h.feed(&LOUD);
        let part: Vec<f32> = [LOUD, LOUD, LOUD, LOUD].concat();
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::LimitReached(part.clone()),
                RecorderEvent::LimitReached(part),
            ]
        );
        // The recording goes on, and its duration counts from the split
        assert!(status.is_recording);
        assert_eq!(status.recording_duration, BUFFER.as_secs_f32());
        assert!(status.split);

        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        let last: Vec<f32> = [&LOUD[..], &QUIET, &QUIET].concat();
        assert_eq!(h.events.len(), 4);
        assert_eq!(h.events[3], RecorderEvent::Ended(last, Extent::Part));
    }

    #[test]
    fn test_a_recording_after_a_split_one_is_whole() {
        let mut h = Harness::with_limit(4 * LOUD.len());
        for _ in 0..5 {
            h.feed(&LOUD);
        }
        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        assert!(!h.feed(&LOUD).split);
        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        let last = [&LOUD[..], &QUIET, &QUIET].concat();
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::LimitReached([LOUD, LOUD, LOUD, LOUD].concat()),
                RecorderEvent::Ended(last.clone(), Extent::Part),
                RecorderEvent::Started,
                RecorderEvent::Ended(last, Extent::Whole),
            ]
        );
    }

    #[test]
    fn test_a_buffer_across_the_limit_is_split_between_parts() {
        // Stereo: the limit of 7 frames falls inside the fourth buffer
        let mut h = Harness::with_limit(14);
        for _ in 0..4 {
            h.feed(&LOUD);
        }
        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        let first: Vec<f32> = [&LOUD[..], &LOUD, &LOUD, &LOUD[..2]].concat();
        let rest: Vec<f32> = [&LOUD[2..], &QUIET, &QUIET].concat();
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::LimitReached(first),
                RecorderEvent::Ended(rest, Extent::Part),
            ]
        );
    }

    #[test]
    fn test_the_pause_before_the_end_counts_toward_the_limit() {
        let mut h = Harness::with_limit(2 * LOUD.len());
        h.feed(&LOUD);
        h.wait_past_hold();
        h.feed(&QUIET);
        h.feed(&QUIET);
        h.feed(&QUIET);
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::LimitReached([LOUD, QUIET].concat()),
                // The rest holds no sound above the threshold: not sent
                RecorderEvent::Ended(Vec::new(), Extent::Part),
            ]
        );
    }

    #[test]
    fn test_the_quiet_rest_after_a_split_is_not_sent() {
        // The split falls at the end of the last loud buffer
        let mut h = Harness::with_limit(3 * LOUD.len());
        for _ in 0..3 {
            h.feed(&LOUD);
        }
        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::LimitReached([LOUD, LOUD, LOUD].concat()),
                RecorderEvent::Ended(Vec::new(), Extent::Part),
            ]
        );
    }

    #[test]
    fn test_the_quiet_samples_of_a_loud_buffer_after_a_split_are_not_sent() {
        // Stereo: the limit of 7 frames falls after the loud samples of
        // the fourth buffer
        let loud_start = [0.5, -0.5, 0.0, 0.0];
        let mut h = Harness::with_limit(14);
        for _ in 0..3 {
            h.feed(&LOUD);
        }
        h.feed(&loud_start);
        h.wait_past_hold();
        for _ in 0..SETTINGS.silence_threshold {
            h.feed(&QUIET);
        }
        let first: Vec<f32> = [&LOUD[..], &LOUD, &LOUD, &loud_start[..2]].concat();
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::LimitReached(first),
                RecorderEvent::Ended(Vec::new(), Extent::Part),
            ]
        );
    }

    #[test]
    fn test_a_quiet_part_of_a_long_pause_is_not_sent() {
        // A pause longer than a part
        let patient = RecorderSettings {
            silence_threshold: 10,
            ..SETTINGS
        };
        let mut h = Harness::with_limit(2 * LOUD.len());
        h.feed_with(&LOUD, &patient);
        h.feed_with(&LOUD, &patient);
        h.wait_past_hold();
        h.feed_with(&QUIET, &patient);
        let status = h.feed_with(&QUIET, &patient);
        // The recording goes on, and its duration counts from the end of
        // the quiet part
        assert!(status.is_recording);
        assert_eq!(status.recording_duration, 0.0);
        h.feed_with(&LOUD, &patient);
        h.wait_past_hold();
        for _ in 0..patient.silence_threshold {
            h.feed_with(&QUIET, &patient);
        }
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::LimitReached([LOUD, LOUD].concat()),
                RecorderEvent::LimitReached([LOUD, QUIET].concat()),
                RecorderEvent::Ended(Vec::new(), Extent::Part),
            ]
        );
    }
}
