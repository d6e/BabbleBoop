/// Events from the audio callback to the processing loop.
#[derive(Debug, PartialEq)]
pub enum AudioEvent {
    StartRecording,
    /// The recording ended. `AudioEvents` in the processing loop also
    /// returns one after an input error and after the end of the input.
    StopRecording,
    /// A whole recording (`Extent::Whole`), or the last part of one that
    /// reached the length limit (`Extent::Part`). StopRecording follows.
    AudioData(CapturedAudio, Extent),
    /// A part of a recording that reached the length limit. The recording
    /// goes on.
    AudioPart(CapturedAudio),
    /// The callback could not queue this many events because the channel
    /// was full. Sent once the channel has room again.
    EventsDropped(u32),
    /// Test Microphone started during a recording. The callback dropped
    /// the recording without processing it. StopRecording follows.
    RecordingDiscarded,
    /// The audio input reported an error or stopped. Holds the message for
    /// the activity log.
    InputError(String),
}

/// Samples of one recording as the input device delivered them.
#[derive(Debug, PartialEq)]
pub struct CapturedAudio {
    /// Interleaved samples
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
}

impl CapturedAudio {
    /// How much of a recording these samples hold. A sample rate or
    /// channel count of 0 cannot happen from a real input stream; both are
    /// clamped to 1, as `Recorder::new` clamps the sample rate, instead of
    /// dividing by zero.
    pub fn duration(&self) -> std::time::Duration {
        let frames = (self.samples.len() / usize::from(self.channels.max(1))) as u64;
        let sample_rate = u64::from(self.sample_rate.max(1));
        let secs = frames / sample_rate;
        let subsec_frames = frames % sample_rate;
        let nanos = subsec_frames * 1_000_000_000 / sample_rate;
        std::time::Duration::new(secs, nanos as u32)
    }
}

/// How much of a recording some audio holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Extent {
    /// All of a recording that did not reach the length limit.
    Whole,
    /// A part of a recording that reached the length limit, including its
    /// last part. The recording holds at least `MAX_RECORDING` of samples
    /// (`Recorder` splits it when its samples reach that length), so the
    /// minimum transcription duration does not apply.
    Part,
}
