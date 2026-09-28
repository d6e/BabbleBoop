//! Converts a recording into the WAV file that is uploaded for
//! transcription.

use crate::types::CapturedAudio;
use hound::WavWriter;
use std::io::Cursor;

/// Encode the recording as a WAV file.
pub fn encode_upload_wav(audio: &CapturedAudio) -> Result<Vec<u8>, hound::Error> {
    let mut wav_buffer = Vec::new();
    let mut writer = WavWriter::new(
        Cursor::new(&mut wav_buffer),
        hound::WavSpec {
            channels: audio.channels,
            sample_rate: audio.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for &sample in &audio.samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(wav_buffer)
}
