use crate::api_client::OpenAi;
use crate::config::OpenAiConfig;
use crate::rate_limiter::RateLimiter;
use serde::Deserialize;
use std::error::Error;

/// Transcribe the WAV file `audio_data`. An error of the API is an
/// `ApiError` in the box.
pub async fn transcribe_audio(
    api: &OpenAi,
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

    let body = api
        .post_multipart("audio/transcriptions", &config.api_key, form)
        .await?;
    parse_transcription(&body)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_client::test_server::{Response, TestServer};
    use crate::api_client::ApiError;
    use reqwest::StatusCode;

    fn openai_config() -> OpenAiConfig {
        OpenAiConfig {
            api_key: "sk-test".to_string(),
            transcription_model: "whisper-1".to_string(),
            ..OpenAiConfig::default()
        }
    }

    #[tokio::test]
    async fn test_the_audio_goes_to_the_transcriptions_path_with_the_key() {
        let mut server = TestServer::start(vec![(
            "audio/transcriptions",
            Response::ok(r#"{"text": "Hello"}"#),
        )])
        .await;
        let api = OpenAi::new(&server.base_url).unwrap();

        let text = transcribe_audio(
            &api,
            b"RIFF wav bytes".to_vec(),
            &openai_config(),
            &mut RateLimiter::new(50),
        )
        .await
        .unwrap();

        assert_eq!(text, "Hello");
        let request = server.request().await;
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/audio/transcriptions");
        assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
        let body = request.body_text();
        assert!(
            body.contains(
                "name=\"file\"; filename=\"audio.wav\"\r\n\
                 Content-Type: audio/wav\r\n\r\nRIFF wav bytes\r\n"
            ),
            "{}",
            body
        );
        assert!(
            body.contains("name=\"model\"\r\n\r\nwhisper-1\r\n"),
            "{}",
            body
        );
    }

    #[tokio::test]
    async fn test_an_error_status_is_an_api_error() {
        let server = TestServer::start(vec![(
            "audio/transcriptions",
            Response::error(StatusCode::TOO_MANY_REQUESTS, "{}"),
        )])
        .await;
        let api = OpenAi::new(&server.base_url).unwrap();

        let error = transcribe_audio(
            &api,
            b"RIFF".to_vec(),
            &openai_config(),
            &mut RateLimiter::new(50),
        )
        .await
        .unwrap_err();

        assert!(
            matches!(
                error.downcast_ref::<ApiError>(),
                Some(ApiError::Http { status, .. }) if *status == StatusCode::TOO_MANY_REQUESTS
            ),
            "{:?}",
            error
        );
    }
}
