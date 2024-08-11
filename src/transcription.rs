use crate::config::OpenAiConfig;
use crate::rate_limiter::RateLimiter;
use serde::Deserialize;
use std::error::Error;

pub async fn transcribe_audio(
    audio_data: Vec<u8>,
    config: &OpenAiConfig,
    rate_limiter: &mut RateLimiter,
) -> Result<String, Box<dyn Error>> {
    println!(
        "Starting audio transcription. Audio data size: {} bytes",
        audio_data.len()
    );

    if audio_data.is_empty() {
        return Err("Audio data is empty".into());
    }

    rate_limiter.wait().await;

    let client = reqwest::Client::new();
    let part = reqwest::multipart::Part::bytes(audio_data)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", "whisper-1");

    println!("Sending request to OpenAI Whisper API");
    let res = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", &config.api_key))
        .multipart(form)
        .send()
        .await?;

    if !res.status().is_success() {
        let error_text = res.text().await?;
        return Err(format!("API request failed: {}", error_text).into());
    }

    #[derive(Deserialize)]
    struct TranscriptionResponse {
        text: String,
    }

    let transcription: TranscriptionResponse = res.json().await?;
    println!("Transcription received: {}", transcription.text);

    if transcription.text.is_empty() {
        return Err("Received empty transcription from API".into());
    }

    Ok(transcription.text)
}