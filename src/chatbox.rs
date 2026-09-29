use crate::config::Config;
use rosc::{encoder::encode, OscMessage, OscPacket, OscType};
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::time::{sleep, Instant};

/// VRChat chatbox has a 144 character limit per message
const VRCHAT_CHATBOX_CHAR_LIMIT: usize = 144;

/// The VRChat chatbox. A message longer than `VRCHAT_CHATBOX_CHAR_LIMIT`
/// goes out in chunks, and each chunk stays on screen for `display_time`
/// before the next one replaces it. After the last chunk of a message the
/// processing loop goes on at once, so the next utterance is transcribed
/// and translated while the message shows. The first chunk of the next
/// message waits for what is left of `display_time`.
pub struct Chatbox {
    socket: Arc<UdpSocket>,
    /// When the last chunk went out
    last_sent: Option<Instant>,
}

impl Chatbox {
    pub fn new(socket: Arc<UdpSocket>) -> Self {
        Self {
            socket,
            last_sent: None,
        }
    }

    /// Send from `socket` from now on.
    pub fn set_socket(&mut self, socket: Arc<UdpSocket>) {
        self.socket = socket;
    }

    /// Wait until the last chunk that went out was on screen for the
    /// `display_time` of `config`. `send` does this wait before each chunk.
    /// A caller can do it first, to check after the wait that the message
    /// is still to go out.
    pub async fn wait_for_display(&self, config: &Config) {
        if let Some(last_sent) = self.last_sent {
            let display_time = Duration::from_millis(config.osc.display_time);
            sleep(display_time.saturating_sub(last_sent.elapsed())).await;
        }
    }

    /// Send `message` in chunks of up to `VRCHAT_CHATBOX_CHAR_LIMIT`
    /// characters, at most `max_message_chunks` of them. Before each chunk,
    /// wait until the chunk before it, of this message or the one before,
    /// was on screen for `display_time`. The wait uses the `display_time`
    /// of `config`, so a new value applies from the next chunk.
    pub async fn send(&mut self, message: &str, config: &Config) -> Result<(), Box<dyn Error>> {
        let osc_address = format!("{}:{}", config.osc.address, config.osc.output_port);

        let chunks: Vec<String> = message
            .chars()
            .collect::<Vec<char>>()
            .chunks(VRCHAT_CHATBOX_CHAR_LIMIT)
            .map(|chunk| chunk.iter().collect::<String>())
            .collect();

        for (i, chunk) in chunks
            .iter()
            .enumerate()
            .take(config.osc.max_message_chunks)
        {
            self.wait_for_display(config).await;

            let osc_message = OscMessage {
                addr: "/chatbox/input".to_string(),
                args: vec![
                    OscType::String(chunk.to_string()),
                    OscType::Bool(true),
                    OscType::Bool(i == 0),
                ],
            };

            let buf = encode(&OscPacket::Message(osc_message))?;
            self.socket.send_to(&buf, osc_address.as_str()).await?;
            self.last_sent = Some(Instant::now());
        }

        Ok(())
    }
}
