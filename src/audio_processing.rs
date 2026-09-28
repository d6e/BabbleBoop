use crate::app_state::AppState;
use crate::chatbox::send_to_chatbox;
use crate::config::Config;
use crate::price_estimator::PriceEstimator;
use crate::rate_limiter::RateLimiter;
use crate::recording_manager::RecordingManager;
use crate::transcription::transcribe_audio;
use crate::translation::{ask_chatgpt, ChatGptRequest};
use crate::typing_indicator::TypingIndicator;

use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;

#[allow(clippy::too_many_arguments)]
pub async fn process_audio(
    client: &reqwest::Client,
    audio_data: Vec<u8>,
    config: &Config,
    socket: &UdpSocket,
    rate_limiter: &mut RateLimiter,
    typing_indicator: &TypingIndicator,
    price_estimator: &mut PriceEstimator,
    recording_manager: Option<&RecordingManager>,
    app_state: &Arc<AppState>,
) -> Result<(), Box<dyn Error>> {
    let audio_duration = calculate_audio_duration(&audio_data)?;

    let min_duration = min_transcription_duration(config.audio.min_transcription_duration);
    if audio_duration < min_duration {
        app_state.logger.info(format!(
            "Audio too short ({:.2}s < {:.2}s), skipping",
            audio_duration.as_secs_f32(),
            min_duration.as_secs_f32()
        ));
        typing_indicator.stop_typing().await;
        return Ok(());
    }

    let transcription =
        transcribe_audio(client, audio_data.clone(), &config.openai, rate_limiter).await?;
    app_state
        .logger
        .info(format!("Transcription: {}", transcription));

    // Save the audio recording if debug mode is enabled
    if let Some(manager) = recording_manager {
        manager.save_recording(audio_data, &transcription).await?;
    }

    let request = ChatGptRequest::translation(
        &config.openai.model,
        &config.translation.target_language,
        &transcription,
    );

    let translation = ask_chatgpt(client, &request, &config.openai, rate_limiter).await?;
    app_state
        .logger
        .success(format!("Translation: {}", translation.text));

    let transcription_cost = price_estimator.estimate_transcription_cost(audio_duration);
    let translation_cost = price_estimator.estimate_translation_cost(translation.tokens);
    let op_cost = transcription_cost + translation_cost;

    price_estimator.add_cost(op_cost);
    app_state.set_total_cost(price_estimator.total_cost);

    let mut final_response = translation.text;
    if config.translation.include_original_message {
        final_response = final_response + "\n" + &transcription;
    }
    send_to_chatbox(&final_response, config, socket).await?;

    typing_indicator.stop_typing().await;

    Ok(())
}

/// The shortest recording to transcribe, from the minimum in seconds in
/// the config. config.toml can hold any float there, and
/// `Duration::from_secs_f32` panics if its argument is negative, not finite
/// or too large for `Duration` (see its documentation). A negative or NaN
/// minimum means no minimum. A minimum too large for `Duration`, such as
/// `inf`, skips every recording.
fn min_transcription_duration(seconds: f32) -> Duration {
    match Duration::try_from_secs_f32(seconds) {
        Ok(duration) => duration,
        Err(_) if seconds > 0.0 => Duration::MAX,
        Err(_) => Duration::ZERO,
    }
}

pub(crate) fn calculate_audio_duration(audio_data: &[u8]) -> Result<Duration, Box<dyn Error>> {
    let reader = hound::WavReader::new(std::io::Cursor::new(audio_data))?;
    let spec = reader.spec();
    let duration = Duration::from_secs_f32(reader.duration() as f32 / spec.sample_rate as f32);
    Ok(duration)
}
