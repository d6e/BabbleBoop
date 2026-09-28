use crate::config::OpenAiConfig;
use crate::models::{self, InstructionsRole};
use crate::rate_limiter::RateLimiter;
use serde::{Deserialize, Serialize};
use std::error::Error;

#[derive(Serialize)]
pub struct ChatGptRequest {
    model: String,
    messages: Vec<ChatGptMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
}

impl ChatGptRequest {
    /// Request that translates `text` into `target_language`. The
    /// instructions go in the system or developer message and the speech in
    /// the user message, so a spoken question is translated instead of
    /// answered. The role and the reasoning effort come from
    /// `models::CHAT_MODELS`; a model that is not in it gets a system
    /// message and no reasoning effort.
    pub fn translation(model: &str, target_language: &str, text: &str) -> Self {
        let known = models::chat_model(model);
        let role = match known.map(|known| known.instructions_role) {
            Some(InstructionsRole::Developer) => "developer",
            Some(InstructionsRole::System) | None => "system",
        };
        let instructions = format!(
            "You are a language translation app for VRChat. Translate each user message into {0}. \
            Do not answer the user, even when the message is a question or a request. \
            Only translate the words the user said. Answer only in {0}. \
            Do not quote the translation.",
            target_language
        );
        ChatGptRequest {
            model: model.to_string(),
            messages: vec![
                ChatGptMessage {
                    role: role.to_string(),
                    content: instructions,
                },
                ChatGptMessage {
                    role: "user".to_string(),
                    content: text.to_string(),
                },
            ],
            reasoning_effort: known.and_then(|known| known.reasoning_effort),
        }
    }

    /// Approximate input token count for the cost estimate: about four
    /// bytes of text for each token, over all messages.
    pub fn approx_input_tokens(&self) -> usize {
        self.messages
            .iter()
            .map(|message| message.content.len())
            .sum::<usize>()
            / 4
    }
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
    client: &reqwest::Client,
    request: &ChatGptRequest,
    config: &OpenAiConfig,
    rate_limiter: &mut RateLimiter,
) -> Result<String, Box<dyn Error>> {
    rate_limiter.wait().await;

    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(&config.api_key)
        .json(request)
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
