use crate::config::OpenAiConfig;
use crate::rate_limiter::RateLimiter;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Serialize)]
struct ChatGptRequest {
    model: String,
    messages: Vec<ChatGptMessage>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatGptMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Clone)]
struct ChatGptResponse {
    choices: Vec<ChatGptChoice>,
}

#[derive(Deserialize, Clone)]
struct ChatGptChoice {
    message: ChatGptMessage,
}

pub async fn ask_chatgpt(
    prompt: &str,
    config: &OpenAiConfig,
    rate_limiter: &mut RateLimiter,
) -> Result<String, Box<dyn Error>> {
    rate_limiter.wait().await;

    let client = reqwest::Client::new();

    let request_body = ChatGptRequest {
        model: config.model.clone(),
        messages: vec![ChatGptMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(&config.api_key)
        .json(&request_body)
        .send()
        .await?;

    if !res.status().is_success() {
        let error_text = res.text().await?;
        return Err(format!("ChatGPT API request failed: {}", error_text).into());
    }

    let res_body: ChatGptResponse = res.json().await?;
    let choice = res_body
        .choices
        .into_iter()
        .next()
        .ok_or("ChatGPT API returned empty choices array")?;
    Ok(choice.message.content)
}
