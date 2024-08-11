use crate::config::OpenAiConfig;
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

pub async fn ask_chatgpt(prompt: &str, config: &OpenAiConfig) -> Result<String, Box<dyn Error>> {
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

    let res_body: ChatGptResponse = res.json().await?;
    Ok(res_body.choices[0].message.content.clone())
}
