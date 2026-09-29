use reqwest::Client;
use std::time::Duration;

/// Time allowed to open the TCP and TLS connection to the API.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Time allowed for one whole request, from connect until the response body
/// is read. The transcription upload is the longest request.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Build the HTTP client shared by all OpenAI API calls.
pub fn build_api_client() -> reqwest::Result<Client> {
    client_with_timeouts(CONNECT_TIMEOUT, REQUEST_TIMEOUT)
}

pub(crate) fn client_with_timeouts(
    connect: Duration,
    request: Duration,
) -> reqwest::Result<Client> {
    Client::builder()
        .connect_timeout(connect)
        .timeout(request)
        .build()
}
