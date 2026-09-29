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

/// Length of the buffer a recording or a part reserves when it starts.
/// Longer speech doubles the buffer, up to `MAX_RECORDING`. The buffer goes
/// with the samples to the processing side, so its capacity stays close to
/// the length of the speech: a 1 s utterance holds 5 s (about 1.9 MB at
/// 48 kHz stereo), not 30 s (about 11.5 MB).
const INITIAL_RESERVE: Duration = Duration::from_secs(5);

/// Number of interleaved samples in `length` of audio, in whole seconds.
fn samples_in(length: Duration, channels: u16, sample_rate: u32) -> usize {
    length.as_secs() as usize * sample_rate as usize * usize::from(channels)
}

/// Settings the GUI can change while the stream runs. The callback reads
/// them for each buffer.
#[derive(Clone, Copy, Debug)]
pub struct RecorderSettings {
    /// Peak level above which the noise gate opens
    pub noise_gate_threshold: f32,
    /// Seconds the gate stays open after the level drops below the threshold
    pub noise_gate_hold_time: f32,
    /// Seconds of input with the gate closed that end a recording
    pub silence_duration: f32,
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
    /// Seconds of input since the gate closed, 0 while it is open or when
    /// not recording
    pub quiet_time: f32,
    pub gate_open: bool,
    /// Seconds until the gate closes if the level stays low
    pub hold_remaining: f32,
    /// Seconds since the recording started or since its last split, 0 when
    /// not recording
    pub recording_duration: f32,
    /// Whether the recording reached the length limit. The minimum
    /// transcription duration then does not apply to its last part.
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
    channels: usize,
    sample_rate: u32,
    /// Frames received since the gate closed. The recording ends when they
    /// reach `silence_duration`. Counting frames, not buffers or clock
    /// time, makes the silence as long as the setting with every buffer
    /// size.
    quiet_frames: usize,
    recording_start: Option<Instant>,
    samples: Vec<f32>,
    /// Whether `samples` holds a sample above the gate threshold
    has_sound: bool,
    /// Whether the recording reached the length limit
    split: bool,
    initial_samples: usize,
    max_samples: usize,
}

impl Recorder {
    /// A recorder for a stream with `channels` and `sample_rate`. It splits
    /// recordings at `MAX_RECORDING`.
    pub fn for_stream(now: Instant, channels: u16, sample_rate: u32) -> Self {
        Self::new(
            now,
            channels,
            sample_rate,
            samples_in(INITIAL_RESERVE, channels, sample_rate),
            samples_in(MAX_RECORDING, channels, sample_rate),
        )
    }

    /// `channels` and `sample_rate` give the length of the input in
    /// seconds. `initial_samples` is the capacity a recording or part
    /// reserves when it starts. `max_samples` is the length at which a
    /// recording is split. It must be a whole number of frames, so that a
    /// split does not fall inside a frame.
    pub fn new(
        now: Instant,
        channels: u16,
        sample_rate: u32,
        initial_samples: usize,
        max_samples: usize,
    ) -> Self {
        let max_samples = max_samples.max(1);
        Recorder {
            gate: NoiseGate::new(now),
            is_recording: false,
            channels: usize::from(channels.max(1)),
            sample_rate: sample_rate.max(1),
            quiet_frames: 0,
            recording_start: None,
            samples: Vec::new(),
            has_sound: false,
            split: false,
            initial_samples: initial_samples.min(max_samples),
            max_samples,
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
            self.reserve(part.len());
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
                    self.discard_samples();
                }
            }
        }
    }

    /// Make room for `additional` samples. The buffer at least doubles, so
    /// that the callback grows it at most 3 times in a part (5 s to 10 s,
    /// 20 s, then 30 s), and each growth can copy the samples. It never
    /// holds more than a part, and after it grows, it holds less than twice
    /// its samples.
    fn reserve(&mut self, additional: usize) {
        let needed = self.samples.len() + additional;
        let capacity = self.samples.capacity();
        if needed > capacity {
            let target = needed.max(2 * capacity).min(self.max_samples);
            self.samples.reserve_exact(target - self.samples.len());
        }
    }

    /// Buffer for a new recording or part.
    fn new_buffer(&self) -> Vec<f32> {
        Vec::with_capacity(self.initial_samples)
    }

    /// Drop the samples of a quiet part and keep its buffer for the next
    /// part or recording, unless the buffer grew. The next part would then
    /// carry that capacity to the processing side.
    fn discard_samples(&mut self) {
        if self.samples.capacity() > self.initial_samples {
            self.samples = self.new_buffer();
        } else {
            self.samples.clear();
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
        let level = peak_level(data);
        let loud = level > settings.noise_gate_threshold;
        let threshold = settings.noise_gate_threshold;
        if self.gate.process(level, settings, now) {
            if !self.is_recording {
                self.is_recording = true;
                self.recording_start = Some(now);
                // A recording that ended without sound keeps a buffer
                if self.samples.capacity() == 0 {
                    self.samples = self.new_buffer();
                }
                emit(RecorderEvent::Started);
            }
            self.record(data, loud, threshold, now, &mut emit);
            self.quiet_frames = 0;
        } else if self.is_recording {
            self.quiet_frames += data.len() / self.channels;
            if self.quiet_frames >= self.frames_in(settings.silence_duration) {
                self.is_recording = false;
                self.quiet_frames = 0;
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
                    self.discard_samples();
                    emit(RecorderEvent::Ended(Vec::new(), extent));
                }
            } else {
                // Keep recording during short pauses
                self.record(data, loud, threshold, now, &mut emit);
            }
        }
    }

    /// Number of frames in `seconds` of input, to the nearest frame. f32
    /// holds most decimal values only approximately: 0.7 is 0.69999999,
    /// which is 13.99999 frames at 20 Hz, not 14. A value that is not a
    /// number gives 0 frames: the recording then ends at the first quiet
    /// buffer instead of never.
    fn frames_in(&self, seconds: f32) -> usize {
        let frames = (f64::from(seconds) * f64::from(self.sample_rate)).round();
        // `as` saturates: NaN and negative values give 0
        frames as usize
    }

    pub fn status(&self, now: Instant) -> RecorderStatus {
        RecorderStatus {
            is_recording: self.is_recording,
            quiet_time: self.quiet_frames as f32 / self.sample_rate as f32,
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
        silence_duration: 0.3,
    };
    /// Buffers of 2 stereo frames at 20 Hz: 0.1 s each
    const CHANNELS: u16 = 2;
    const SAMPLE_RATE: u32 = 20;
    const LOUD: [f32; 4] = [0.0, 0.5, -0.5, 0.0];
    const QUIET: [f32; 4] = [0.0, 0.05, -0.05, 0.0];
    /// Time between two buffers in these tests, the length of `LOUD` and
    /// `QUIET`
    const BUFFER: Duration = Duration::from_millis(100);
    /// Quiet buffers that end a recording with `SETTINGS`: 0.3 s
    const QUIET_BUFFERS_TO_END: usize = 3;

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

        /// A recorder that splits recordings at `max_samples`. Each
        /// recording or part starts with room for one buffer, so the
        /// buffer grows while it records.
        fn with_limit(max_samples: usize) -> Self {
            Self::with_recorder(|now| {
                Recorder::new(now, CHANNELS, SAMPLE_RATE, LOUD.len(), max_samples)
            })
        }

        fn with_recorder(recorder: impl FnOnce(Instant) -> Recorder) -> Self {
            let now = Instant::now();
            Harness {
                recorder: recorder(now),
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
            self.now +=
                Duration::try_from_secs_f32(SETTINGS.noise_gate_hold_time).unwrap() + BUFFER;
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
        assert_eq!(status.quiet_time, 0.0);
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
        assert_eq!(first_silent.quiet_time, BUFFER.as_secs_f32());
        assert!(first_silent.is_recording);
        h.feed(&QUIET);
        let ended = h.feed(&QUIET);

        // The buffer that reaches the silence duration is not recorded; the
        // quiet buffers before it are.
        let expected: Vec<f32> = [&LOUD[..], &QUIET, &QUIET, &QUIET].concat();
        assert_eq!(
            h.events,
            vec![
                RecorderEvent::Started,
                RecorderEvent::Ended(expected, Extent::Whole)
            ]
        );
        assert!(!ended.is_recording);
        assert_eq!(ended.quiet_time, 0.0);
        assert_eq!(ended.recording_duration, 0.0);
    }

    #[test]
    fn test_sound_during_the_silence_resets_the_quiet_time() {
        let mut h = Harness::new();
        h.feed(&LOUD);
        h.wait_past_hold();
        h.feed(&QUIET);
        h.feed(&QUIET);
        let status = h.feed(&LOUD);
        assert_eq!(status.quiet_time, 0.0);
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
        for _ in 0..QUIET_BUFFERS_TO_END {
            h.feed(&QUIET);
        }
        h.feed(&LOUD);
        h.wait_past_hold();
        for _ in 0..QUIET_BUFFERS_TO_END {
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
        for _ in 0..QUIET_BUFFERS_TO_END {
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
        for _ in 0..QUIET_BUFFERS_TO_END {
            h.feed(&QUIET);
        }
        assert!(!h.feed(&LOUD).split);
        h.wait_past_hold();
        for _ in 0..QUIET_BUFFERS_TO_END {
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
        for _ in 0..QUIET_BUFFERS_TO_END {
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
        for _ in 0..QUIET_BUFFERS_TO_END {
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
        for _ in 0..QUIET_BUFFERS_TO_END {
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
            silence_duration: 1.0,
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
        // 1 s of quiet buffers
        for _ in 0..10 {
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

    /// Buffers of 441 frames: 30 s at 48 kHz is not a whole number of them,
    /// so the split falls inside a buffer.
    const STREAM_BUFFER_FRAMES: usize = 441;

    /// Loud samples that differ from each other, so that a lost or
    /// reordered sample changes the recording.
    fn loud_samples(start: usize, len: usize) -> Vec<f32> {
        (start..start + len)
            .map(|i| 0.2 + (i * 7919 % 10007) as f32 / 10007.0 * 0.7)
            .collect()
    }

    /// Record `millis` ms of loud input from a stream, then end the
    /// recording with silence. Returns the samples the recorder received
    /// and, for each part, the number of times the capacity of the
    /// recorder's buffer increased. Checks after each buffer that the
    /// recorder holds no more than 30 s, and after each increase that the
    /// buffer holds less than twice its samples.
    fn record_from_stream(
        h: &mut Harness,
        channels: u16,
        sample_rate: u32,
        millis: usize,
    ) -> (Vec<f32>, Vec<usize>) {
        let per_second = sample_rate as usize * usize::from(channels);
        let buffer_len = STREAM_BUFFER_FRAMES * usize::from(channels);
        let limit = 30 * per_second;
        let mut fed = Vec::new();
        let mut growths = vec![0];
        while fed.len() < per_second * millis / 1000 {
            let data = loud_samples(fed.len(), buffer_len);
            let held = h.recorder.samples.len();
            let capacity = h.recorder.samples.capacity();
            let events_before = h.events.len();
            h.feed(&data);
            fed.extend_from_slice(&data);
            let grown = h.recorder.samples.capacity();
            if held > 0 && grown > capacity {
                let len = h.recorder.samples.len();
                assert!(
                    grown < 2 * len,
                    "grew from {capacity} to {grown} for {len} samples"
                );
                if let Some(part) = growths.last_mut() {
                    *part += 1;
                }
            }
            assert!(grown <= limit);
            for event in &h.events[events_before..] {
                if matches!(event, RecorderEvent::LimitReached(_)) {
                    growths.push(0);
                }
            }
        }
        h.wait_past_hold();
        let quiet = vec![0.0; buffer_len];
        for _ in 0..per_second {
            h.feed(&quiet);
            if matches!(h.events.last(), Some(RecorderEvent::Ended(..))) {
                return (fed, growths);
            }
            // The buffer that ends the recording is not recorded
            fed.extend_from_slice(&quiet);
        }
        panic!("the recording did not end");
    }

    fn short_recording_reserves_about_five_seconds(channels: u16, sample_rate: u32) {
        let mut h = Harness::with_recorder(|now| Recorder::for_stream(now, channels, sample_rate));
        let (fed, _) = record_from_stream(&mut h, channels, sample_rate, 1000);
        let [RecorderEvent::Started, RecorderEvent::Ended(samples, Extent::Whole)] = &h.events[..]
        else {
            panic!("unexpected events {:?}", h.events.len());
        };
        assert_eq!(samples, &fed);
        let per_second = sample_rate as usize * usize::from(channels);
        assert!(
            (4 * per_second..=5 * per_second).contains(&samples.capacity()),
            "capacity {} for {} samples",
            samples.capacity(),
            samples.len()
        );
    }

    #[test]
    fn test_a_short_recording_reserves_about_five_seconds_in_stereo() {
        short_recording_reserves_about_five_seconds(2, 48000);
    }

    #[test]
    fn test_a_short_recording_reserves_about_five_seconds_with_three_channels() {
        short_recording_reserves_about_five_seconds(3, 44100);
    }

    fn long_recording_grows_to_thirty_seconds(channels: u16, sample_rate: u32) {
        let mut h = Harness::with_recorder(|now| Recorder::for_stream(now, channels, sample_rate));
        let (fed, growths) = record_from_stream(&mut h, channels, sample_rate, 42_000);
        let [RecorderEvent::Started, RecorderEvent::LimitReached(first), RecorderEvent::Ended(rest, Extent::Part)] =
            &h.events[..]
        else {
            panic!("unexpected events {:?}", h.events.len());
        };
        let limit = 30 * sample_rate as usize * usize::from(channels);
        assert_eq!(first.len(), limit);
        assert!(first.capacity() <= limit);
        // The second part grew from its start to about 12 s
        assert!(rest.capacity() < 2 * rest.len());
        // The buffer doubles: 5, 10, 20 and 30 s in the first part (3
        // growths), 5, 10 and 20 s in the second (2 growths)
        assert_eq!(growths.len(), 2);
        assert!(
            growths.iter().all(|&n| n <= 3),
            "growths per part {growths:?}"
        );
        assert_eq!([&first[..], rest].concat(), fed);
    }

    #[test]
    fn test_a_long_recording_grows_to_thirty_seconds_in_stereo() {
        long_recording_grows_to_thirty_seconds(2, 48000);
    }

    #[test]
    fn test_a_long_recording_grows_to_thirty_seconds_with_three_channels() {
        long_recording_grows_to_thirty_seconds(3, 44100);
    }

    /// A recording that ends just after 5 s holds at most 10 s, and one that
    /// ends just after 10 s at most 20 s. The buffer that reaches the
    /// processing side holds less than twice its samples.
    fn growing_recording_holds_less_than_twice_its_samples(channels: u16, sample_rate: u32) {
        let per_second = sample_rate as usize * usize::from(channels);
        for (millis, reserved_seconds) in [(5_100, 10), (10_100, 20)] {
            let mut h =
                Harness::with_recorder(|now| Recorder::for_stream(now, channels, sample_rate));
            let (fed, _) = record_from_stream(&mut h, channels, sample_rate, millis);
            let [RecorderEvent::Started, RecorderEvent::Ended(samples, Extent::Whole)] =
                &h.events[..]
            else {
                panic!("unexpected events {:?}", h.events.len());
            };
            assert_eq!(samples, &fed);
            let capacity = samples.capacity();
            assert!(
                capacity <= reserved_seconds * per_second && capacity < 2 * samples.len(),
                "{millis} ms: capacity {capacity} for {} samples, {} per second",
                samples.len(),
                per_second
            );
        }
    }

    #[test]
    fn test_a_growing_recording_holds_less_than_twice_its_samples_in_stereo() {
        growing_recording_holds_less_than_twice_its_samples(2, 48000);
    }

    #[test]
    fn test_a_growing_recording_holds_less_than_twice_its_samples_with_three_channels() {
        growing_recording_holds_less_than_twice_its_samples(3, 44100);
    }

    /// Settings with a long hold time, so that a quiet part fits in one
    /// pause.
    const LONG_HOLD: RecorderSettings = RecorderSettings {
        noise_gate_hold_time: 2.0,
        ..SETTINGS
    };

    fn end_after_long_hold(h: &mut Harness) {
        h.now += Duration::from_secs(3);
        for _ in 0..QUIET_BUFFERS_TO_END {
            h.feed_with(&QUIET, &LONG_HOLD);
        }
    }

    #[test]
    fn test_a_part_after_a_quiet_part_does_not_keep_its_buffer() {
        let mut h = Harness::with_limit(8 * LOUD.len());
        for _ in 0..8 {
            h.feed_with(&LOUD, &LONG_HOLD);
        }
        // A whole quiet part inside the hold time
        for _ in 0..8 {
            h.feed_with(&QUIET, &LONG_HOLD);
        }
        h.feed_with(&LOUD, &LONG_HOLD);
        end_after_long_hold(&mut h);
        let last = [LOUD, QUIET, QUIET].concat();
        let Some(RecorderEvent::Ended(samples, Extent::Part)) = h.events.last() else {
            panic!("unexpected events {:?}", h.events);
        };
        assert_eq!(samples, &last);
        assert!(samples.capacity() <= 2 * last.len());
    }

    #[test]
    fn test_a_recording_after_a_quiet_end_does_not_keep_its_buffer() {
        let mut h = Harness::with_limit(8 * LOUD.len());
        for _ in 0..8 {
            h.feed_with(&LOUD, &LONG_HOLD);
        }
        // A quiet rest of 7 buffers, not sent
        for _ in 0..5 {
            h.feed_with(&QUIET, &LONG_HOLD);
        }
        end_after_long_hold(&mut h);
        assert_eq!(
            h.events.last(),
            Some(&RecorderEvent::Ended(Vec::new(), Extent::Part))
        );
        h.feed_with(&LOUD, &LONG_HOLD);
        end_after_long_hold(&mut h);
        let last = [LOUD, QUIET, QUIET].concat();
        let Some(RecorderEvent::Ended(samples, Extent::Whole)) = h.events.last() else {
            panic!("unexpected events {:?}", h.events);
        };
        assert_eq!(samples, &last);
        assert!(samples.capacity() <= 2 * last.len());
    }

    /// Record loud input from a stream, then feed quiet buffers of
    /// `buffer_frames` frames until the recording ends. Returns the number
    /// of quiet frames fed, with the buffer that ended the recording.
    fn quiet_frames_that_end_a_recording(
        channels: u16,
        sample_rate: u32,
        buffer_frames: usize,
        silence_duration: f32,
    ) -> usize {
        // The gate closes at the first quiet buffer
        let settings = RecorderSettings {
            noise_gate_hold_time: 0.0,
            silence_duration,
            ..SETTINGS
        };
        let buffer_len = buffer_frames * usize::from(channels);
        let mut h = Harness::with_recorder(|now| Recorder::for_stream(now, channels, sample_rate));
        h.feed_with(&loud_samples(0, buffer_len), &settings);
        let quiet = vec![0.0; buffer_len];
        let mut frames = 0;
        while !matches!(h.events.last(), Some(RecorderEvent::Ended(..))) {
            assert!(frames < 20 * sample_rate as usize, "no end after 20 s");
            h.feed_with(&quiet, &settings);
            frames += buffer_frames;
        }
        frames
    }

    /// The recording ends with the buffer during which the quiet input
    /// reaches the silence duration, whatever the buffer size, sample rate
    /// and channel count. The harness clock moves 0.1 s per buffer, so the
    /// end does not come from the clock.
    #[test]
    fn test_silence_duration_is_the_same_with_every_buffer_size() {
        // Channels, sample rate, silence duration, and its frames
        let streams = [
            (2, 48_000, 1.0, 48_000),
            (2, 48_000, 0.3, 14_400),
            (1, 44_100, 1.0, 44_100),
            (1, 44_100, 0.3, 13_230),
        ];
        for (channels, sample_rate, silence_duration, expected) in streams {
            for buffer_frames in [480, 441, 1024] {
                let fed = quiet_frames_that_end_a_recording(
                    channels,
                    sample_rate,
                    buffer_frames,
                    silence_duration,
                );
                assert!(
                    fed >= expected && fed - buffer_frames < expected,
                    "{channels} channels, {sample_rate} Hz, {buffer_frames} frame buffers, \
                    {silence_duration} s: ended after {fed} quiet frames, not with the \
                    buffer that reaches {expected}"
                );
            }
        }
    }

    #[test]
    fn test_silence_duration_is_rounded_to_the_nearest_frame() {
        // 0.7 in f32 is 0.69999999, so 0.7 s at 20 Hz is 13.99999 frames.
        // Buffers of one frame end the recording at the 14th.
        let settings = RecorderSettings {
            silence_duration: 0.7,
            noise_gate_hold_time: 0.0,
            ..SETTINGS
        };
        let mut h = Harness::new();
        h.feed_with(&LOUD, &settings);
        for _ in 0..13 {
            h.feed_with(&QUIET[..2], &settings);
        }
        assert_eq!(h.events, vec![RecorderEvent::Started]);
        h.feed_with(&QUIET[..2], &settings);
        assert!(matches!(h.events.last(), Some(RecorderEvent::Ended(..))));
    }

    #[test]
    fn test_quiet_time_shows_the_seconds_of_quiet_input() {
        let mut h = Harness::with_recorder(|now| Recorder::for_stream(now, 2, 48_000));
        let settings = RecorderSettings {
            noise_gate_hold_time: 0.0,
            ..SETTINGS
        };
        h.feed_with(&[0.5; 2 * 4800], &settings);
        // 0.1 s and 0.15 s of quiet stereo input
        h.feed_with(&[0.0; 2 * 4800], &settings);
        let status = h.feed_with(&[0.0; 2 * 7200], &settings);
        assert!(status.is_recording);
        assert_eq!(status.quiet_time, 0.25);
    }

    #[test]
    fn test_a_changed_silence_duration_applies_to_the_silence_in_progress() {
        let mut h = Harness::new();
        let patient = RecorderSettings {
            silence_duration: 1.0,
            ..SETTINGS
        };
        h.feed(&LOUD);
        h.wait_past_hold();
        h.feed(&QUIET);
        h.feed(&QUIET);
        // 0.3 s of silence does not end the recording with 1 s set
        assert!(h.feed_with(&QUIET, &patient).is_recording);
        assert_eq!(h.events, vec![RecorderEvent::Started]);
        // The shorter duration ends it at the next quiet buffer
        assert!(!h.feed(&QUIET).is_recording);
        let expected = [&LOUD[..], &QUIET, &QUIET, &QUIET].concat();
        assert_eq!(h.events[1], RecorderEvent::Ended(expected, Extent::Whole));
    }

    #[test]
    fn test_a_silence_duration_that_is_not_a_number_ends_at_the_first_quiet_buffer() {
        let mut h = Harness::new();
        let broken = RecorderSettings {
            silence_duration: f32::NAN,
            ..SETTINGS
        };
        h.feed_with(&LOUD, &broken);
        h.wait_past_hold();
        assert!(!h.feed_with(&QUIET, &broken).is_recording);
    }
}
