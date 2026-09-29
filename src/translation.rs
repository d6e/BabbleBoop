use crate::api_client::OpenAi;
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
    /// The content filter removed part of the text
    /// (`finish_reason: content_filter` with content). What is left is not
    /// the whole translation, so it is not sent.
    Filtered,
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
                    "The content filter removed the translation, \
                     so nothing was sent (finish_reason: content_filter)"
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
            NoTranslation::Filtered => write!(
                f,
                "The content filter removed part of the translation, \
                 so it was not sent (finish_reason: content_filter)"
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
    /// then a cut off or filtered text, then a blank one.
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
            (Some(_), Some(reason)) if reason == "content_filter" => Err(NoTranslation::Filtered),
            (Some(content), _) => Ok(content),
            (None, finish_reason) => Err(NoTranslation::Empty { finish_reason }),
        }
    }
}

/// Send `request` to Chat Completions. An error of the API is an
/// `ApiError` in the box.
pub async fn ask_chatgpt(
    api: &OpenAi,
    request: &ChatGptRequest,
    config: &OpenAiConfig,
    rate_limiter: &mut RateLimiter,
) -> Result<Translation, Box<dyn Error>> {
    rate_limiter.wait().await;

    let body = api
        .post_json("chat/completions", &config.api_key, request)
        .await?;
    request.parse_response(&body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_client::test_server::{Response, TestServer};
    use crate::test_support::{chat_completion_body, chat_completion_body_with, REASONING_USAGE};

    #[tokio::test]
    async fn test_the_request_goes_to_the_chat_completions_path_with_the_key() {
        let mut server = TestServer::start(vec![(
            "chat/completions",
            Response::ok(
                r#"{"choices": [{"message": {"content": "Bonjour"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 2}}"#,
            ),
        )])
        .await;
        let api = OpenAi::new(&server.base_url).unwrap();
        let config = OpenAiConfig {
            api_key: "sk-test".to_string(),
            model: "gpt-4o-mini".to_string(),
            ..OpenAiConfig::default()
        };
        let request = ChatGptRequest::translation(&config.model, "French", "Hello");

        let translation = ask_chatgpt(&api, &request, &config, &mut RateLimiter::new(50))
            .await
            .unwrap();

        assert_eq!(translation.text, Ok("Bonjour".to_string()));
        assert_eq!(
            (translation.tokens.input, translation.tokens.output),
            (30, 2)
        );
        let received = server.request().await;
        assert_eq!(received.method, "POST");
        assert_eq!(received.path, "/v1/chat/completions");
        assert_eq!(received.header("authorization"), Some("Bearer sk-test"));
        assert_eq!(received.header("content-type"), Some("application/json"));
        let body: serde_json::Value = serde_json::from_slice(&received.body).unwrap();
        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["messages"][0]["role"], "system");
        assert!(
            body["messages"][0]["content"]
                .as_str()
                .unwrap()
                .contains("Translate each user message into French"),
            "{}",
            body
        );
        assert_eq!(
            body["messages"][1],
            serde_json::json!({ "role": "user", "content": "Hello" })
        );
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    }

    // ===========================================================================
    // Test: Translation request shape
    // ===========================================================================

    #[test]
    fn test_translation_request_sends_speech_as_user_message() {
        use serde_json::json;

        let spoken = "What time is it?";
        let request = ChatGptRequest::translation("gpt-4o-mini", "Japanese", spoken);
        let body = serde_json::to_value(&request).unwrap();

        assert_eq!(body["model"], "gpt-4o-mini");
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        let instructions = messages[0]["content"].as_str().unwrap();
        assert!(instructions.contains("Japanese"));
        assert!(!instructions.contains(spoken));
        assert_eq!(messages[1], json!({"role": "user", "content": spoken}));
    }

    #[test]
    fn test_translation_request_options_follow_the_model() {
        // (model, instructions role, reasoning_effort)
        for (model, role, effort) in [
            ("gpt-6-luna", "developer", Some("none")),
            ("gpt-6-sol", "developer", Some("none")),
            // Shutdown replacements; without `none` they reason at `medium`.
            ("gpt-5.6-luna", "developer", Some("none")),
            ("gpt-5.6-terra", "developer", Some("none")),
            ("gpt-5.6-sol", "developer", Some("none")),
            ("gpt-5.4-nano", "developer", None),
            ("gpt-4.1-mini", "system", None),
            ("gpt-4o-mini", "system", None),
            ("my-finetuned-model", "system", None),
        ] {
            let request = ChatGptRequest::translation(model, "Japanese", "Hello");
            let body = serde_json::to_value(&request).unwrap();
            assert_eq!(body["messages"][0]["role"], role, "{}", model);
            assert_eq!(
                body.get("reasoning_effort").map(|v| v.as_str().unwrap()),
                effort,
                "{}",
                model
            );
        }
    }

    #[test]
    fn test_translation_request_token_estimate_counts_all_messages() {
        let silent = ChatGptRequest::translation("gpt-4o-mini", "Japanese", "");
        let spoken = ChatGptRequest::translation("gpt-4o-mini", "Japanese", &"a".repeat(400));

        // The instructions count, and so does the speech at 4 bytes a token.
        assert!(silent.approx_input_tokens() > 0);
        assert_eq!(
            spoken.approx_input_tokens() - silent.approx_input_tokens(),
            100
        );
    }

    #[test]
    fn test_translation_tokens_come_from_reported_usage() {
        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "What time is it?");
        let body = chat_completion_body("Quelle heure est-il ?", Some(REASONING_USAGE));

        let translation = request.parse_response(&body).unwrap();

        assert_eq!(translation.text, Ok("Quelle heure est-il ?".to_string()));
        // completion_tokens includes the 320 reasoning tokens.
        assert_eq!(
            translation.tokens,
            TokenCounts {
                input: 58,
                output: 331
            }
        );
    }

    #[test]
    fn test_translation_tokens_are_estimated_without_usage() {
        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "What time is it?");
        for body in [
            chat_completion_body("Quelle heure est-il ?", None),
            chat_completion_body("Quelle heure est-il ?", Some("null")),
        ] {
            let translation = request.parse_response(&body).unwrap();

            assert_eq!(translation.text, Ok("Quelle heure est-il ?".to_string()));
            // Four bytes a token: 21 bytes of translation give 5 tokens.
            assert_eq!(
                translation.tokens,
                TokenCounts {
                    input: request.approx_input_tokens(),
                    output: 5
                }
            );
        }
    }

    #[test]
    fn test_translation_cost_counts_reported_reasoning_tokens() {
        use crate::test_support::price_estimator;

        let estimator = price_estimator("gpt-5.6-sol", "gpt-transcribe");
        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "What time is it?");
        let text = "Quelle heure est-il ?";
        let reported = request
            .parse_response(&chat_completion_body(text, Some(REASONING_USAGE)))
            .unwrap();
        let estimated = request
            .parse_response(&chat_completion_body(text, None))
            .unwrap();

        let reported_cost = estimator.estimate_translation_cost(reported.tokens);
        assert_eq!(
            reported_cost,
            estimator.estimate_translation_cost(TokenCounts {
                input: 58,
                output: 331
            })
        );
        // The text alone does not show the reasoning tokens.
        assert!(reported_cost > estimator.estimate_translation_cost(estimated.tokens));
    }

    #[test]
    fn test_refusal_tokens_are_estimated_from_the_refusal_without_usage() {
        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "Hello");
        let refusal = "I'm sorry, but I can't help with that.";
        let body = chat_completion_body_with(
            "null",
            &serde_json::to_string(refusal).unwrap(),
            "stop",
            None,
        );

        let translation = request.parse_response(&body).unwrap();

        assert_eq!(
            translation.text,
            Err(NoTranslation::Refused(refusal.to_string()))
        );
        // Four bytes a token: 38 bytes of refusal give 9 tokens.
        assert_eq!(translation.tokens.output, 9);
    }

    #[test]
    fn test_blank_refusal_is_not_a_refusal() {
        let request = ChatGptRequest::translation("gpt-5.6-sol", "French", "Hello");
        let body = chat_completion_body_with(r#""Bonjour""#, r#""""#, "stop", None);

        let translation = request.parse_response(&body).unwrap();

        assert_eq!(translation.text, Ok("Bonjour".to_string()));
    }
}
