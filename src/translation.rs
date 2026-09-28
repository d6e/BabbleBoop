use crate::config::OpenAiConfig;
use crate::models::{self, InstructionsRole};
use crate::price_estimator::TokenCounts;
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

    /// Reads the response body of this request. The token counts come from
    /// the `usage` that the API reports. A count that is missing is
    /// estimated at four bytes of text for each token, which does not
    /// include reasoning tokens. A response that the API returns without a
    /// usable translation, such as a refusal, is still a `Translation`: the
    /// API charges for its tokens.
    pub fn parse_response(&self, body: &str) -> Result<Translation, Box<dyn Error>> {
        let response: ChatGptResponse = serde_json::from_str(body)?;
        let (text, answer_len) = match response.choices.into_iter().next() {
            Some(choice) => {
                let answer_len = choice
                    .message
                    .content
                    .as_ref()
                    .or(choice.message.refusal.as_ref())
                    .map_or(0, String::len);
                (choice.into_text(), answer_len)
            }
            None => (Err(NoTranslation::NoChoices), 0),
        };
        let usage = response.usage.unwrap_or_default();
        let tokens = TokenCounts {
            input: usage
                .prompt_tokens
                .unwrap_or_else(|| self.approx_input_tokens()),
            output: usage.completion_tokens.unwrap_or(answer_len / 4),
        };
        Ok(Translation { text, tokens })
    }
}

/// The answer to a translation request and the tokens that the request
/// used.
pub struct Translation {
    /// The translated text, or why the response holds none to send.
    pub text: Result<String, NoTranslation>,
    pub tokens: TokenCounts,
}

/// Why a Chat Completions response holds no translation to send.
#[derive(Debug, PartialEq)]
pub enum NoTranslation {
    /// The model refused, with this message.
    Refused(String),
    /// The response has no choices.
    NoChoices,
    /// The content is null or blank. `finish_reason` is from the response,
    /// such as `content_filter`.
    Empty { finish_reason: Option<String> },
    /// The model reached the output token limit (`finish_reason: length`).
    /// The request sets no limit, so the text reached the limit of the
    /// model, which is much longer than a translation of a 30 s part of
    /// speech. Such a text is not a translation (for example, the model
    /// repeated itself), and its end is missing, so it is not sent.
    CutOff,
}

impl std::fmt::Display for NoTranslation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NoTranslation::Refused(refusal) => {
                write!(f, "The model refused to translate: {}", refusal)
            }
            NoTranslation::NoChoices => write!(
                f,
                "The model returned no translation (the response has no choices)"
            ),
            NoTranslation::Empty { finish_reason } => match finish_reason.as_deref() {
                Some("content_filter") => write!(
                    f,
                    "The model returned no translation: the content filter removed it \
                     (finish_reason: content_filter)"
                ),
                Some(reason) => write!(
                    f,
                    "The model returned no translation (finish_reason: {})",
                    reason
                ),
                None => write!(f, "The model returned no translation"),
            },
            NoTranslation::CutOff => write!(
                f,
                "The translation reached the output token limit and was cut off, \
                 so it was not sent (finish_reason: length)"
            ),
        }
    }
}

impl Error for NoTranslation {}

#[derive(Serialize, Clone)]
pub struct ChatGptMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
struct ChatGptResponse {
    choices: Vec<ChatGptChoice>,
    usage: Option<ChatGptUsage>,
}

/// Token usage that the API reports. `completion_tokens` includes the
/// reasoning tokens.
#[derive(Deserialize, Clone, Default)]
struct ChatGptUsage {
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
}

/// One choice of a Chat Completions response. The API reference
/// (developers.openai.com/api/reference/resources/chat) gives `content` as
/// string or null, `refusal` as an optional string or null, and
/// `finish_reason` as `stop`, `length`, `tool_calls`, `content_filter` or
/// `function_call`.
#[derive(Deserialize)]
struct ChatGptChoice {
    message: ChatGptAnswer,
    finish_reason: Option<String>,
}

/// The assistant message of a choice.
#[derive(Deserialize)]
struct ChatGptAnswer {
    content: Option<String>,
    refusal: Option<String>,
}

impl ChatGptChoice {
    /// The text to send, or why there is none. A refusal comes first,
    /// then a cut off text, then a blank one.
    fn into_text(self) -> Result<String, NoTranslation> {
        if let Some(refusal) = self
            .message
            .refusal
            .filter(|refusal| !refusal.trim().is_empty())
        {
            return Err(NoTranslation::Refused(refusal));
        }
        let content = self
            .message
            .content
            .filter(|content| !content.trim().is_empty());
        match (content, self.finish_reason) {
            (Some(_), Some(reason)) if reason == "length" => Err(NoTranslation::CutOff),
            (Some(content), _) => Ok(content),
            (None, finish_reason) => Err(NoTranslation::Empty { finish_reason }),
        }
    }
}

pub async fn ask_chatgpt(
    client: &reqwest::Client,
    request: &ChatGptRequest,
    config: &OpenAiConfig,
    rate_limiter: &mut RateLimiter,
) -> Result<Translation, Box<dyn Error>> {
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

    request.parse_response(&res.text().await?)
}
