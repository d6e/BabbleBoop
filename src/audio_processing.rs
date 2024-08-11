use crate::config::Config;
use crate::rate_limiter::RateLimiter;
use crate::price_estimator::PriceEstimator;
use crate::typing_indicator::TypingIndicator;
use crate::transcription::transcribe_audio;
use crate::translation::ask_chatgpt;
use crate::chatbox::send_to_chatbox;

use std::error::Error;
use std::time::Duration;
use tokio::net::UdpSocket;

pub async fn process_audio(
    audio_data: Vec<u8>,
    config: &Config,
    socket: &UdpSocket,
    rate_limiter: &mut RateLimiter,
    typing_indicator: &TypingIndicator,
    price_estimator: &mut PriceEstimator,
) -> Result<(), Box<dyn Error>> {
    let audio_duration = calculate_audio_duration(&audio_data)?;

    let min_duration = Duration::from_secs_f32(config.audio.min_transcription_duration);
    if audio_duration < min_duration {
        println!(
            "Audio too short ({:.2}s). Minimum duration is {:.2}s. Skipping transcription.",
            audio_duration.as_secs_f32(),
            min_duration.as_secs_f32()
        );
        typing_indicator.stop_typing().await;
        return Ok(());
    }

    let transcription = transcribe_audio(audio_data, &config.openai, rate_limiter).await?;
    println!("Transcription: {}", transcription);

    let translation_prompt = format!(
        "You are a language translation app for VRChat. Answer only in the target language. Do not quote the translation. target_language={} Text:\n\n{}",
        config.translation.target_language, transcription
    );

    let mut response = ask_chatgpt(&translation_prompt, &config.openai).await?;
    println!("Translation: {}", response);

    let transcription_cost = price_estimator.estimate_transcription_cost(audio_duration);
    let input_tokens = translation_prompt.len() / 4;
    let output_tokens = response.len() / 4;
    let translation_cost = price_estimator.estimate_translation_cost(input_tokens, output_tokens);
    let total_cost = transcription_cost + translation_cost;

    price_estimator.add_cost(total_cost);
    println!("Estimated cost for this operation: ${:.4}", total_cost);
    println!("Total cost so far: ${:.4}", price_estimator.total_cost);
    println!("---");

    if config.translation.include_original_message {
        response = response + "\n" + &transcription;
    }
    send_to_chatbox(&response, &config, socket).await?;

    typing_indicator.stop_typing().await;

    Ok(())
}

fn calculate_audio_duration(audio_data: &[u8]) -> Result<Duration, Box<dyn Error>> {
    let reader = hound::WavReader::new(std::io::Cursor::new(audio_data))?;
    let spec = reader.spec();
    let duration = Duration::from_secs_f32(reader.duration() as f32 / spec.sample_rate as f32);
    Ok(duration)
}