use crate::config::OpenAiConfig;
use crate::rate_limiter::RateLimiter;
use serde::Deserialize;
use std::error::Error;

pub async fn transcribe_audio(
    client: &reqwest::Client,
    audio_data: Vec<u8>,
    config: &OpenAiConfig,
    rate_limiter: &mut RateLimiter,
) -> Result<String, Box<dyn Error>> {
    if audio_data.is_empty() {
        return Err("Audio data is empty".into());
    }

    rate_limiter.wait().await;

    let part = reqwest::multipart::Part::bytes(audio_data)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", config.transcription_model.clone());

    let res = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", config.api_key))
        .multipart(form)
        .send()
        .await?;

    if !res.status().is_success() {
        let error_text = res.text().await?;
        return Err(format!("API request failed: {}", error_text).into());
    }

    parse_transcription(&res.text().await?)
}

/// Reads the text of a transcription response body. The text can be empty
/// or blank; `audio_processing::accept_transcription` decides what to do
/// with it, after it adds the cost.
pub(crate) fn parse_transcription(body: &str) -> Result<String, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct TranscriptionResponse {
        text: String,
    }

    let transcription: TranscriptionResponse = serde_json::from_str(body)?;
    Ok(transcription.text)
}
