//! The HTTP client of the OpenAI API.

use reqwest::multipart::Form;
use reqwest::{Client, RequestBuilder, StatusCode};
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

/// The base URL of the OpenAI API. The tests give `OpenAi` the URL of a
/// local server instead.
pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Time allowed to open the TCP and TLS connection to the API.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Time allowed for one whole request, from connect until the response body
/// is read. The transcription upload is the longest request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Longest error message from the API that the activity log shows, in
/// characters. A longer message is cut and ends in "...".
const MAX_MESSAGE_CHARS: usize = 120;

/// The OpenAI API at a base URL. Every request sends the API key as a
/// bearer token.
#[derive(Clone)]
pub struct OpenAi {
    client: Client,
    /// Without a slash at the end
    base_url: String,
}

impl OpenAi {
    /// The API at `base_url`, such as `OPENAI_BASE_URL`.
    pub fn new(base_url: &str) -> reqwest::Result<Self> {
        Self::with_timeouts(base_url, CONNECT_TIMEOUT, REQUEST_TIMEOUT)
    }

    pub(crate) fn with_timeouts(
        base_url: &str,
        connect: Duration,
        request: Duration,
    ) -> reqwest::Result<Self> {
        let client = Client::builder()
            .connect_timeout(connect)
            .timeout(request)
            .build()?;
        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Send `body` as JSON to `path`, such as `chat/completions`, and
    /// return the response body. The API key comes from the settings, which
    /// can change between requests.
    pub async fn post_json(
        &self,
        path: &str,
        api_key: &str,
        body: &impl Serialize,
    ) -> Result<String, ApiError> {
        self.send(self.post(path).json(body), api_key).await
    }

    /// Send `form` as `multipart/form-data` to `path`, such as
    /// `audio/transcriptions`, and return the response body.
    pub async fn post_multipart(
        &self,
        path: &str,
        api_key: &str,
        form: Form,
    ) -> Result<String, ApiError> {
        self.send(self.post(path).multipart(form), api_key).await
    }

    fn post(&self, path: &str) -> RequestBuilder {
        self.client.post(format!("{}/{}", self.base_url, path))
    }

    /// Send `request` with the API key and read the response body. This is
    /// the only place that sets the authorization header.
    async fn send(&self, request: RequestBuilder, api_key: &str) -> Result<String, ApiError> {
        let response = request.bearer_auth(api_key).send().await?;
        let status = response.status();
        let body = response.text().await?;
        if status.is_success() {
            Ok(body)
        } else {
            Err(ApiError::Http { status, body })
        }
    }
}

/// A request to the OpenAI API that failed.
#[derive(Debug)]
pub enum ApiError {
    /// No response came, or its body could not be read: no connection, a
    /// timeout, or a connection that closed.
    Transport(reqwest::Error),
    /// The API answered with a status other than 2xx. `body` usually holds
    /// an OpenAI error object.
    Http { status: StatusCode, body: String },
}

impl ApiError {
    /// The whole error for stderr: the status and the body as the API sent
    /// them, or the transport error with its causes.
    pub fn details(&self) -> String {
        match self {
            ApiError::Transport(e) => format!("{:?}", e),
            ApiError::Http { status, body } => format!("HTTP {}: {}", status, body),
        }
    }
}

impl From<reqwest::Error> for ApiError {
    fn from(e: reqwest::Error) -> Self {
        ApiError::Transport(e)
    }
}

/// The message for the activity log.
impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Transport(e) => write!(f, "Request to the OpenAI API failed: {}", e),
            ApiError::Http { status, body } => match error_message(body) {
                Some(message) => f.write_str(&message),
                None if body.trim().is_empty() => {
                    write!(f, "API request failed with HTTP {}", status)
                }
                None => write!(
                    f,
                    "API request failed with HTTP {}: {}",
                    status,
                    shorten(body.trim())
                ),
            },
        }
    }
}

impl std::error::Error for ApiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ApiError::Transport(e) => Some(e),
            ApiError::Http { .. } => None,
        }
    }
}

/// The message of an OpenAI error body, `{"error": {"code": ..., "message":
/// ...}}`: a fixed text for a common code, or else the message in the body.
/// `None` if the body holds neither.
fn error_message(body: &str) -> Option<String> {
    let parsed: Value = serde_json::from_str(body).ok()?;
    let error = parsed.get("error")?;
    let fixed = match error.get("code").and_then(Value::as_str) {
        Some("invalid_api_key") => Some("Invalid API key. Check your OpenAI API key in settings."),
        Some("insufficient_quota") => Some("OpenAI API quota exceeded. Check your billing."),
        Some("rate_limit_exceeded") => Some("Rate limit exceeded. Please wait and try again."),
        Some("model_not_found") => Some("Model not found. Check your model settings."),
        _ => None,
    };
    match fixed {
        Some(message) => Some(message.to_string()),
        None => error.get("message").and_then(Value::as_str).map(shorten),
    }
}

/// `text`, cut to `MAX_MESSAGE_CHARS` characters if it is longer. It counts
/// characters, not bytes, so the cut cannot fall inside a multibyte
/// character.
fn shorten(text: &str) -> String {
    if text.chars().count() > MAX_MESSAGE_CHARS {
        let kept: String = text.chars().take(MAX_MESSAGE_CHARS - 3).collect();
        format!("{}...", kept)
    } else {
        text.to_string()
    }
}

/// A local HTTP server for tests. It sends each request that it receives
/// to the test, and answers it with the response for its path.
#[cfg(test)]
pub(crate) mod test_server {
    use reqwest::StatusCode;
    use std::time::Duration;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc;
    use tokio::task::JoinHandle;

    /// The path of every route starts with this, like the paths of the API.
    const PREFIX: &str = "/v1/";

    /// A request as the server received it.
    #[derive(Debug)]
    pub(crate) struct Request {
        pub method: String,
        pub path: String,
        /// Names in lower case
        pub headers: Vec<(String, String)>,
        pub body: Vec<u8>,
    }

    impl Request {
        /// The value of the header `name` (lower case).
        pub fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.as_str())
        }

        pub fn body_text(&self) -> String {
            String::from_utf8_lossy(&self.body).into_owned()
        }
    }

    /// The answer to a request.
    #[derive(Clone)]
    pub(crate) struct Response {
        pub status: StatusCode,
        pub body: String,
    }

    impl Response {
        pub fn ok(body: impl Into<String>) -> Self {
            Self {
                status: StatusCode::OK,
                body: body.into(),
            }
        }

        pub fn error(status: StatusCode, body: impl Into<String>) -> Self {
            Self {
                status,
                body: body.into(),
            }
        }
    }

    pub(crate) struct TestServer {
        /// The base URL to give `OpenAi`
        pub base_url: String,
        requests: mpsc::UnboundedReceiver<Request>,
        task: JoinHandle<()>,
    }

    impl TestServer {
        /// Serve `routes`. A request to the base URL plus a route path gets
        /// the response of that route, and any other request gets 404.
        pub async fn start(routes: Vec<(&'static str, Response)>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base_url = format!("http://{}{}", listener.local_addr().unwrap(), PREFIX);
            let (tx, requests) = mpsc::unbounded_channel();
            let task = tokio::spawn(async move {
                loop {
                    let (connection, _) = listener.accept().await.unwrap();
                    let mut reader = BufReader::new(connection);
                    let request = read_request(&mut reader).await;
                    let response = request
                        .path
                        .strip_prefix(PREFIX)
                        .and_then(|route| routes.iter().find(|(path, _)| *path == route))
                        .map_or_else(
                            || Response::error(StatusCode::NOT_FOUND, "no route"),
                            |(_, response)| response.clone(),
                        );
                    // Before the answer, so the test has the request when
                    // the client has the response
                    if tx.send(request).is_err() {
                        return;
                    }
                    respond(reader.into_inner(), response).await;
                }
            });
            Self {
                base_url,
                requests,
                task,
            }
        }

        /// The next request that the server received. Waits up to 5 s.
        pub async fn request(&mut self) -> Request {
            tokio::time::timeout(Duration::from_secs(5), self.requests.recv())
                .await
                .expect("the server received no request")
                .unwrap()
        }

        /// The requests received and not yet returned by `request`.
        pub fn received(&mut self) -> Vec<Request> {
            std::iter::from_fn(|| self.requests.try_recv().ok()).collect()
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    /// Read one request from `reader`.
    async fn read_request(reader: &mut BufReader<TcpStream>) -> Request {
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let mut request_line = line.split_whitespace();
        let method = request_line.next().unwrap().to_string();
        let path = request_line.next().unwrap().to_string();
        let mut headers = Vec::new();
        loop {
            line.clear();
            reader.read_line(&mut line).await.unwrap();
            let header = line.trim_end();
            if header.is_empty() {
                break;
            }
            let (name, value) = header.split_once(':').unwrap();
            headers.push((name.trim().to_lowercase(), value.trim().to_string()));
        }
        let mut request = Request {
            method,
            path,
            headers,
            body: Vec::new(),
        };
        assert_eq!(
            request.header("transfer-encoding"),
            None,
            "the server reads only a body with a content-length"
        );
        let length = request
            .header("content-length")
            .map_or(0, |length| length.parse().unwrap());
        request.body = vec![0; length];
        reader.read_exact(&mut request.body).await.unwrap();
        request
    }

    /// Send `response` and close the connection, so the client opens a new
    /// one for the next request.
    async fn respond(mut connection: TcpStream, response: Response) {
        let head = format!(
            "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.status,
            response.body.len()
        );
        connection.write_all(head.as_bytes()).await.unwrap();
        connection
            .write_all(response.body.as_bytes())
            .await
            .unwrap();
        connection.shutdown().await.unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::test_server::{Response, TestServer};
    use super::*;

    /// The activity log text of an HTTP error with `status` and `body`.
    fn shown(status: StatusCode, body: &str) -> String {
        ApiError::Http {
            status,
            body: body.to_string(),
        }
        .to_string()
    }

    /// An OpenAI error body with `code` and `message`.
    fn error_body(code: &str, message: &str) -> String {
        serde_json::json!({ "error": { "message": message, "type": "x", "code": code } })
            .to_string()
    }

    #[test]
    fn test_a_common_error_code_shows_a_fixed_message() {
        for (status, code, expected) in [
            (
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "Invalid API key. Check your OpenAI API key in settings.",
            ),
            (
                StatusCode::TOO_MANY_REQUESTS,
                "insufficient_quota",
                "OpenAI API quota exceeded. Check your billing.",
            ),
            (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                "Rate limit exceeded. Please wait and try again.",
            ),
            (
                StatusCode::NOT_FOUND,
                "model_not_found",
                "Model not found. Check your model settings.",
            ),
        ] {
            assert_eq!(
                shown(status, &error_body(code, "raw text of the API")),
                expected,
                "{}",
                code
            );
        }
    }

    #[test]
    fn test_an_unknown_error_code_shows_the_message_of_the_body() {
        assert_eq!(
            shown(
                StatusCode::BAD_REQUEST,
                &error_body("invalid_value", "Invalid file format.")
            ),
            "Invalid file format."
        );
        // The API sends null for an error without a code
        let no_code = r#"{"error": {"message": "Server overloaded.", "code": null}}"#;
        assert_eq!(
            shown(StatusCode::SERVICE_UNAVAILABLE, no_code),
            "Server overloaded."
        );
    }

    #[test]
    fn test_a_body_without_an_error_message_shows_the_status_and_the_body() {
        assert_eq!(
            shown(StatusCode::BAD_GATEWAY, "<html>Bad gateway</html>\n"),
            "API request failed with HTTP 502 Bad Gateway: <html>Bad gateway</html>"
        );
        assert_eq!(
            shown(StatusCode::BAD_REQUEST, r#"{"error": {"code": "x"}}"#),
            r#"API request failed with HTTP 400 Bad Request: {"error": {"code": "x"}}"#
        );
        assert_eq!(
            shown(StatusCode::INTERNAL_SERVER_ERROR, ""),
            "API request failed with HTTP 500 Internal Server Error"
        );
    }

    #[test]
    fn test_a_long_body_without_an_error_message_is_cut() {
        let body = "語".repeat(200);
        assert_eq!(
            shown(StatusCode::BAD_GATEWAY, &body),
            format!(
                "API request failed with HTTP 502 Bad Gateway: {}...",
                "語".repeat(117)
            )
        );
    }

    #[test]
    fn test_details_keep_the_whole_body() {
        let body = error_body("invalid_api_key", "Incorrect API key provided: sk-abc.");
        let error = ApiError::Http {
            status: StatusCode::UNAUTHORIZED,
            body: body.clone(),
        };
        assert_eq!(error.details(), format!("HTTP 401 Unauthorized: {}", body));
    }

    /// The API at a local port where nothing listens.
    async fn closed_port_api() -> OpenAi {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        drop(listener);
        OpenAi::new(&base_url).unwrap()
    }

    #[tokio::test]
    async fn test_a_transport_error_names_the_api_and_the_cause() {
        let error = closed_port_api()
            .await
            .post_json("chat/completions", "sk-test", &serde_json::json!({}))
            .await
            .unwrap_err();

        assert!(matches!(error, ApiError::Transport(_)), "{:?}", error);
        let shown = error.to_string();
        assert!(
            shown.starts_with("Request to the OpenAI API failed: error sending request"),
            "{}",
            shown
        );
        assert!(shown.contains("/v1/chat/completions"), "{}", shown);
        // stderr gets the cause too
        assert!(
            error.details().contains("ConnectError"),
            "{}",
            error.details()
        );
    }

    #[tokio::test]
    async fn test_post_json_sends_the_body_and_the_key_to_the_path() {
        let mut server =
            TestServer::start(vec![("chat/completions", Response::ok(r#"{"ok": 1}"#))]).await;
        let api = OpenAi::new(&server.base_url).unwrap();

        let body = api
            .post_json(
                "chat/completions",
                "sk-test",
                &serde_json::json!({ "a": 1 }),
            )
            .await
            .unwrap();

        assert_eq!(body, r#"{"ok": 1}"#);
        let request = server.request().await;
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
        assert_eq!(request.header("content-type"), Some("application/json"));
        assert_eq!(request.body_text(), r#"{"a":1}"#);
    }

    #[tokio::test]
    async fn test_post_multipart_sends_the_form_and_the_key_to_the_path() {
        let mut server =
            TestServer::start(vec![("audio/transcriptions", Response::ok("{}"))]).await;
        let api = OpenAi::new(&server.base_url).unwrap();

        let form = Form::new().text("model", "whisper-1");
        api.post_multipart("audio/transcriptions", "sk-test", form)
            .await
            .unwrap();

        let request = server.request().await;
        assert_eq!(request.path, "/v1/audio/transcriptions");
        assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
        assert!(
            request
                .header("content-type")
                .is_some_and(|value| value.starts_with("multipart/form-data; boundary=")),
            "{:?}",
            request
        );
        assert!(
            request
                .body_text()
                .contains("name=\"model\"\r\n\r\nwhisper-1\r\n"),
            "{}",
            request.body_text()
        );
    }

    #[tokio::test]
    async fn test_a_status_other_than_2xx_is_an_http_error_with_the_body() {
        let body = error_body("invalid_api_key", "Incorrect API key provided.");
        let server = TestServer::start(vec![(
            "chat/completions",
            Response::error(StatusCode::UNAUTHORIZED, body.clone()),
        )])
        .await;
        let api = OpenAi::new(&server.base_url).unwrap();

        let error = api
            .post_json("chat/completions", "sk-wrong", &serde_json::json!({}))
            .await
            .unwrap_err();

        match error {
            ApiError::Http {
                status,
                body: received,
            } => {
                assert_eq!(status, StatusCode::UNAUTHORIZED);
                assert_eq!(received, body);
            }
            other => panic!("expected an HTTP error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_a_slash_at_the_end_of_the_base_url_is_not_doubled() {
        let mut server = TestServer::start(vec![("chat/completions", Response::ok("{}"))]).await;
        // The base URL of the server ends in a slash
        assert!(server.base_url.ends_with('/'));
        let api = OpenAi::new(&server.base_url).unwrap();

        api.post_json("chat/completions", "sk-test", &serde_json::json!({}))
            .await
            .unwrap();

        assert_eq!(server.request().await.path, "/v1/chat/completions");
    }
}
