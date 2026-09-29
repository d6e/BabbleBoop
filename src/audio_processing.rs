use crate::app_state::AppState;
use crate::chatbox::Chatbox;
use crate::config::Config;
use crate::price_estimator::PriceEstimator;
use crate::rate_limiter::RateLimiter;
use crate::recording_manager::RecordingManager;
use crate::transcription::transcribe_audio;
use crate::translation::{ask_chatgpt, ChatGptRequest, Translation};
use crate::types::Extent;
use crate::typing_indicator::TypingIndicator;

use std::error::Error;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

#[allow(clippy::too_many_arguments)]
pub async fn process_audio(
    client: &reqwest::Client,
    audio_data: Vec<u8>,
    extent: Extent,
    config: &Config,
    chatbox: &mut Chatbox,
    rate_limiter: &mut RateLimiter,
    typing_indicator: &TypingIndicator,
    price_estimator: &mut PriceEstimator,
    recording_manager: Option<&mut RecordingManager>,
    app_state: &Arc<AppState>,
) -> Result<(), Box<dyn Error>> {
    let audio_duration = calculate_audio_duration(&audio_data)?;

    let min_duration = min_transcription_duration(config.audio.min_transcription_duration);
    if extent == Extent::Whole && audio_duration < min_duration {
        app_state.logger.info(format!(
            "Audio too short ({:.2}s < {:.2}s), skipping",
            audio_duration.as_secs_f32(),
            min_duration.as_secs_f32()
        ));
        typing_indicator.stop_typing().await;
        return Ok(());
    }

    let text = transcribe_audio(client, audio_data.clone(), &config.openai, rate_limiter).await?;
    let Some(transcription) = accept_transcription(
        text,
        audio_duration,
        typing_indicator,
        price_estimator,
        app_state,
    )
    .await
    else {
        return Ok(());
    };

    // Save the audio recording if debug mode is enabled. A failed save is
    // logged by the manager itself and does not stop the translation.
    if let Some(manager) = recording_manager {
        manager
            .save_recording(audio_data, &transcription, &app_state.logger)
            .await;
    }

    let request = ChatGptRequest::translation(
        &config.openai.model,
        &config.translation.target_language,
        &transcription,
    );

    let translation = ask_chatgpt(client, &request, &config.openai, rate_limiter).await?;
    deliver_translation(
        translation,
        &transcription,
        config,
        chatbox,
        typing_indicator,
        price_estimator,
        app_state,
    )
    .await
}

/// Add the cost of the transcription request to the total and return the
/// text to translate. The cost is added before the later steps, so it
/// stays in the total when one of them fails. The estimate depends only on
/// the audio duration, so an empty or blank text costs as much as speech.
/// For such a text, an error message goes to the activity log, the typing
/// indicator turns off, and the function returns `None`.
pub(crate) async fn accept_transcription(
    text: String,
    audio_duration: Duration,
    typing_indicator: &TypingIndicator,
    price_estimator: &mut PriceEstimator,
    app_state: &AppState,
) -> Option<String> {
    price_estimator.add_cost(
        price_estimator.estimate_transcription_cost(audio_duration),
        &app_state.logger,
    );
    app_state.set_total_cost(price_estimator.total_cost);

    if text.trim().is_empty() {
        app_state
            .logger
            .error("The transcription is empty, so nothing was translated");
        typing_indicator.stop_typing().await;
        return None;
    }
    app_state.logger.info(format!("Transcription: {}", text));
    Some(text)
}

/// Add the cost of the translation request to the total and send the
/// translation to the chatbox. A response without a translation, such as
/// a refusal, costs its tokens too; it goes to the activity log, not to the
/// chatbox. The translation waits for the display time of the previous
/// message, and is not sent if translation is off after that wait. A
/// message that has started to go out is sent to its last chunk.
pub(crate) async fn deliver_translation(
    translation: Translation,
    transcription: &str,
    config: &Config,
    chatbox: &mut Chatbox,
    typing_indicator: &TypingIndicator,
    price_estimator: &mut PriceEstimator,
    app_state: &AppState,
) -> Result<(), Box<dyn Error>> {
    let translation_cost = price_estimator.estimate_translation_cost(translation.tokens);
    price_estimator.add_cost(translation_cost, &app_state.logger);
    app_state.set_total_cost(price_estimator.total_cost);

    let text = match translation.text {
        Ok(text) => text,
        Err(no_translation) => {
            app_state.logger.error(no_translation.to_string());
            typing_indicator.stop_typing().await;
            return Ok(());
        }
    };
    app_state.logger.success(format!("Translation: {}", text));

    let mut final_response = text;
    if config.translation.include_original_message {
        final_response = final_response + "\n" + transcription;
    }
    // Translation can be switched off while this message waits for the
    // previous one or for the API. The processing loop checks the toggle
    // only when it takes the audio event, so check it again here.
    chatbox.wait_for_display(config).await;
    if !app_state.enabled.load(Ordering::Relaxed) {
        app_state
            .logger
            .info("Translation is off, so the translation was not sent to the chatbox");
        typing_indicator.stop_typing().await;
        return Ok(());
    }
    chatbox.send(&final_response, config).await?;

    typing_indicator.stop_typing().await;

    Ok(())
}

/// The shortest recording to transcribe, from the minimum in seconds in
/// the config. config.toml can hold any float there, and
/// `Duration::from_secs_f32` panics if its argument is negative, not finite
/// or too large for `Duration` (see its documentation). A negative or NaN
/// minimum means no minimum. A minimum too large for `Duration`, such as
/// `inf`, skips every whole recording. The minimum does not apply to the
/// parts of a recording that reached the length limit: `process_audio`
/// checks it only when `extent == Extent::Whole`, so the minimum never
/// stops those parts.
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
    let duration = Duration::try_from_secs_f32(reader.duration() as f32 / spec.sample_rate as f32)?;
    Ok(duration)
}
