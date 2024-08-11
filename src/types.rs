pub enum AudioEvent {
    StartRecording,
    StopRecording,
    AudioData(Vec<u8>),
}