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

#[cfg(test)]
mod tests {
    use super::*;

    // ===========================================================================
    // Test: Chatbox display pause
    // ===========================================================================

    /// A `Chatbox` that sends to a local UDP socket, the config to send
    /// with (`display_time` in ms, up to 10 chunks), and that socket.
    async fn local_chatbox(display_time: u64) -> (Chatbox, Config, std::net::UdpSocket) {
        let receiver = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = receiver.local_addr().unwrap().port();
        config.osc.display_time = display_time;
        config.osc.max_message_chunks = 10;
        (Chatbox::new(socket), config, receiver)
    }

    /// The texts of the chatbox messages that reached `receiver` since the
    /// last call. `receiver` is a std socket read with a timeout in real
    /// time, so reading it does not move the paused tokio clock.
    fn received_texts(receiver: &std::net::UdpSocket) -> Vec<String> {
        let mut texts = Vec::new();
        let mut buf = [0u8; 1024];
        while let Ok(len) = receiver.recv(&mut buf) {
            match rosc::decoder::decode_udp(&buf[..len]).unwrap().1 {
                rosc::OscPacket::Message(message) => match message.args.first() {
                    Some(rosc::OscType::String(text)) => texts.push(text.clone()),
                    _ => panic!("unexpected chatbox message {:?}", message),
                },
                bundle => panic!("unexpected OSC bundle {:?}", bundle),
            }
        }
        texts
    }

    #[tokio::test(start_paused = true)]
    async fn test_chatbox_pauses_between_chunks_but_not_after_the_last() {
        use tokio::time::sleep_until;

        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        let message = format!("{}{}{}", "a".repeat(144), "b".repeat(144), "c");

        let start = Instant::now();
        let (sent_after, arrivals) = tokio::join!(
            async {
                chatbox.send(&message, &config).await.unwrap();
                start.elapsed()
            },
            async {
                // Between the expected send times: 0 s, 10 s and 20 s
                let mut arrivals = Vec::new();
                for seconds in [5, 15, 25] {
                    sleep_until(start + Duration::from_secs(seconds)).await;
                    arrivals.push(received_texts(&receiver));
                }
                arrivals
            }
        );

        assert_eq!(
            arrivals,
            [["a".repeat(144)], ["b".repeat(144)], ["c".to_string()]]
        );
        assert_eq!(sent_after, Duration::from_secs(20));
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_waits_the_rest_of_the_display_time() {
        use tokio::time::sleep;

        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        chatbox.send("first", &config).await.unwrap();
        sleep(Duration::from_secs(4)).await;

        let start = Instant::now();
        chatbox.send("second", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::from_secs(6));
        assert_eq!(received_texts(&receiver), ["first", "second"]);
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_waits_from_the_last_chunk() {
        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        // Two chunks: the second goes out 10 s after the first.
        chatbox.send(&"a".repeat(145), &config).await.unwrap();

        let start = Instant::now();
        chatbox.send("next", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::from_secs(10));
        assert_eq!(received_texts(&receiver).last().unwrap(), "next");
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_after_the_display_time_does_not_wait() {
        use tokio::time::sleep;

        let (mut chatbox, config, receiver) = local_chatbox(10_000).await;
        chatbox.send("first", &config).await.unwrap();
        sleep(Duration::from_secs(15)).await;

        let start = Instant::now();
        chatbox.send("second", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::ZERO);
        assert_eq!(received_texts(&receiver), ["first", "second"]);
    }

    #[tokio::test(start_paused = true)]
    async fn test_next_message_waits_the_current_display_time() {
        let (mut chatbox, mut config, _receiver) = local_chatbox(10_000).await;
        chatbox.send("first", &config).await.unwrap();
        config.osc.display_time = 2_000;

        let start = Instant::now();
        chatbox.send("second", &config).await.unwrap();

        assert_eq!(start.elapsed(), Duration::from_secs(2));
    }

    // ===========================================================================
    // Test: Shutdown interrupts in flight work
    // ===========================================================================

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_interrupts_chatbox_display_pause() {
        use crate::shutdown::Shutdown;
        use tokio::time::sleep;

        let (mut chatbox, config, _receiver) = local_chatbox(30_000).await;
        // Three chunks, so the chatbox pauses 60 s in total.
        let message = "a".repeat(300);

        let shutdown = Shutdown::new();
        let requester = shutdown.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(1)).await;
            requester.request();
        });

        let start = Instant::now();
        let result = shutdown.run_until(chatbox.send(&message, &config)).await;

        assert!(result.is_none(), "shutdown did not stop the chatbox send");
        assert_eq!(start.elapsed(), Duration::from_secs(1));
    }

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_interrupts_wait_before_next_message() {
        use crate::shutdown::Shutdown;
        use tokio::time::sleep;

        let (mut chatbox, config, receiver) = local_chatbox(30_000).await;
        chatbox.send("first", &config).await.unwrap();
        assert_eq!(received_texts(&receiver), ["first"]);

        let shutdown = Shutdown::new();
        let requester = shutdown.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(1)).await;
            requester.request();
        });

        let start = Instant::now();
        let result = shutdown.run_until(chatbox.send("second", &config)).await;

        assert!(result.is_none(), "shutdown did not stop the wait");
        assert_eq!(start.elapsed(), Duration::from_secs(1));
        assert_eq!(received_texts(&receiver), Vec::<String>::new());
    }
}
