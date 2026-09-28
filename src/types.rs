/// Events from the audio callback to the processing loop.
#[derive(Debug, PartialEq)]
pub enum AudioEvent {
    StartRecording,
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
