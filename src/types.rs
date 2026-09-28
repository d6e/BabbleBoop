/// Events from the audio callback to the processing loop.
#[derive(Debug, PartialEq)]
pub enum AudioEvent {
    StartRecording,
    StopRecording,
    AudioData(CapturedAudio),
    /// The callback could not queue this many events because the channel
    /// was full. Sent once the channel has room again.
    EventsDropped(u32),
}

/// Samples of one recording as the input device delivered them.
#[derive(Debug, PartialEq)]
pub struct CapturedAudio {
    /// Interleaved samples
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
}
