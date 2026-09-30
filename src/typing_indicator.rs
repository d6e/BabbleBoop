use crate::app_state::{FailureLog, Logger};
use crate::config::Config;
use rosc::{encoder::encode, OscMessage, OscPacket, OscType};
use std::sync::Arc;
use tokio::net::UdpSocket;

/// The typing indicator of the VRChat chatbox. Like `Chatbox`, it takes
/// the settings with each call and holds no copy of them. An on and an
/// off go to the address in the settings of the caller, except the off
/// of `stop_typing_elsewhere`, which goes to the address of the last on.
///
/// It remembers where BabbleBoop turned the indicator on, and sends an off
/// only there. While translation is off, another app can use the chatbox,
/// and an off that BabbleBoop did not cause can clear the indicator of
/// that app. So a second off for one recording, an off for a recording
/// that did not turn the indicator on, and an off at a destination where
/// it is not on send nothing.
pub struct TypingIndicator {
    socket: Arc<UdpSocket>,
    logger: Logger,
    /// A recording turns the indicator on once and off at most once, a
    /// long recording turns it on again after each part, and switching
    /// translation off and the end of the processing loop each send an
    /// off. While VRChat cannot be reached at the configured address,
    /// every send fails the same way.
    send_failure: FailureLog,
    /// The destination (`address:port`) of the last on, until an off goes
    /// there. It is kept also when the send of the on failed, so the off
    /// still goes out.
    on_at: Option<String>,
}

/// Where `config` sends the chatbox messages and the typing indicator.
fn destination(config: &Config) -> String {
    format!("{}:{}", config.osc.address, config.osc.output_port)
}

impl TypingIndicator {
    pub fn new(socket: Arc<UdpSocket>, logger: Logger) -> Self {
        TypingIndicator {
            socket,
            logger,
            send_failure: FailureLog::default(),
            on_at: None,
        }
    }

    /// Send from `socket` from now on.
    pub fn set_socket(&mut self, socket: Arc<UdpSocket>) {
        self.socket = socket;
    }

    /// Send the typing state to the VRChat chatbox at `destination`. A
    /// failed send goes to the activity log, once until a send works again
    /// or fails with a different message.
    async fn send(&mut self, is_typing: bool, destination: &str) {
        let typing_message = OscMessage {
            addr: "/chatbox/typing".to_string(),
            args: vec![OscType::Bool(is_typing)],
        };
        if let Ok(buf) = encode(&OscPacket::Message(typing_message)) {
            match self.socket.send_to(&buf, destination).await {
                Ok(_) => {
                    if self.send_failure.succeeded() {
                        self.logger
                            .info(format!("Typing indicator sent to {} again", destination));
                    }
                }
                Err(e) => self.send_failure.failed(
                    &self.logger,
                    format!("Cannot send the typing indicator to {}: {}", destination, e),
                ),
            }
        }
    }

    /// Turn the indicator on at the address in `config`.
    pub async fn start_typing(&mut self, config: &Config) {
        let destination = destination(config);
        self.send(true, &destination).await;
        self.on_at = Some(destination);
    }

    /// Turn the indicator off at the address in `config`, if BabbleBoop
    /// turned it on there. Otherwise send nothing.
    pub async fn stop_typing(&mut self, config: &Config) {
        let destination = destination(config);
        if self.on_at.as_ref() == Some(&destination) {
            self.on_at = None;
            self.send(false, &destination).await;
        }
    }

    /// Before the settings change to `config`: if BabbleBoop turned the
    /// indicator on at a different address, turn it off there, and
    /// return true, so the caller can turn it on at the new address. A
    /// new name for the same place, such as localhost for 127.0.0.1,
    /// counts as a different address: to find that it is the same place
    /// takes a DNS lookup.
    pub async fn stop_typing_elsewhere(&mut self, config: &Config) -> bool {
        match self.on_at.take() {
            Some(old) if old != destination(config) => {
                self.send(false, &old).await;
                true
            }
            on_at => {
                self.on_at = on_at;
                false
            }
        }
    }

    /// Turn the indicator off at the address in `config`, also if
    /// BabbleBoop did not turn it on there. Only for the two times when
    /// the user expects the indicator to be off: translation switched off,
    /// and the end of the processing loop. The remembered state is what
    /// BabbleBoop sent, not what VRChat shows. They differ after an off
    /// whose send failed or whose UDP packet was lost, and after a run of
    /// BabbleBoop that ended before its off. So these two send the off in
    /// any case. The cost is one packet that can clear the indicator of
    /// another app.
    pub async fn force_stop_typing(&mut self, config: &Config) {
        self.on_at = None;
        self.send(false, &destination(config)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{LogEntry, LogLevel};
    use crate::test_support::{recv_osc, Osc};
    use std::time::Duration;
    use tokio::sync::mpsc;

    /// Level and text of the entries in the activity log since the last call.
    fn new_entries(log_rx: &mut mpsc::Receiver<LogEntry>) -> Vec<(LogLevel, String)> {
        std::iter::from_fn(|| log_rx.try_recv().ok())
            .map(|entry| (entry.level, entry.message))
            .collect()
    }

    /// Settings that send to `vrchat`.
    fn sending_to(vrchat: &UdpSocket) -> Config {
        let mut config = Config::default();
        config.osc.address = "127.0.0.1".to_string();
        config.osc.output_port = vrchat.local_addr().unwrap().port();
        config
    }

    /// An indicator that sends from a local socket, and two VRChat
    /// sockets to send to.
    async fn indicator_and_two_vrchats() -> (TypingIndicator, UdpSocket, UdpSocket) {
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let (log_tx, _log_rx) = mpsc::channel(10);
        let indicator = TypingIndicator::new(socket, Logger::new(log_tx, Default::default()));
        let first = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let second = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        (indicator, first, second)
    }

    /// The next message on `vrchat`.
    async fn received(vrchat: &UdpSocket) -> Osc {
        recv_osc(vrchat, Duration::from_secs(5)).await.0
    }

    /// Whether a message waits on `vrchat`. The sends are to local
    /// sockets, so a message sent before a later received one is there.
    fn has_message(vrchat: &UdpSocket) -> bool {
        let mut buf = [0u8; 1024];
        vrchat.try_recv(&mut buf).is_ok()
    }

    #[tokio::test]
    async fn test_an_off_without_an_on_sends_nothing() {
        let (mut indicator, vrchat, marker) = indicator_and_two_vrchats().await;

        indicator.stop_typing(&sending_to(&vrchat)).await;
        // Sent after the off, if it was sent
        indicator.force_stop_typing(&sending_to(&marker)).await;

        assert_eq!(received(&marker).await, Osc::Typing(false));
        assert!(!has_message(&vrchat), "an off went out without an on");
    }

    #[tokio::test]
    async fn test_a_second_off_sends_nothing() {
        let (mut indicator, vrchat, _) = indicator_and_two_vrchats().await;
        let config = sending_to(&vrchat);

        indicator.start_typing(&config).await;
        indicator.stop_typing(&config).await;
        indicator.stop_typing(&config).await;
        indicator.start_typing(&config).await;

        for expected in [Osc::Typing(true), Osc::Typing(false), Osc::Typing(true)] {
            assert_eq!(received(&vrchat).await, expected);
        }
    }

    #[tokio::test]
    async fn test_an_off_goes_only_to_the_destination_that_is_on() {
        let (mut indicator, first, second) = indicator_and_two_vrchats().await;

        indicator.start_typing(&sending_to(&first)).await;
        indicator.stop_typing(&sending_to(&second)).await;
        indicator.stop_typing(&sending_to(&first)).await;
        // Sent after the off to `second`, if it was sent
        indicator.force_stop_typing(&sending_to(&second)).await;

        assert_eq!(received(&first).await, Osc::Typing(true));
        assert_eq!(received(&first).await, Osc::Typing(false));
        assert_eq!(received(&second).await, Osc::Typing(false));
        assert!(!has_message(&second), "an off went where typing was not on");
    }

    #[tokio::test]
    async fn test_typing_on_elsewhere_goes_off_there_once() {
        let (mut indicator, first, second) = indicator_and_two_vrchats().await;

        assert!(!indicator.stop_typing_elsewhere(&sending_to(&second)).await);
        indicator.start_typing(&sending_to(&first)).await;
        assert!(!indicator.stop_typing_elsewhere(&sending_to(&first)).await);
        assert!(indicator.stop_typing_elsewhere(&sending_to(&second)).await);
        assert!(!indicator.stop_typing_elsewhere(&sending_to(&second)).await);
        // Typing is off at `first`, so this sends nothing
        indicator.stop_typing(&sending_to(&first)).await;
        indicator.start_typing(&sending_to(&first)).await;

        for expected in [Osc::Typing(true), Osc::Typing(false), Osc::Typing(true)] {
            assert_eq!(received(&first).await, expected);
        }
        assert!(!has_message(&second));
    }

    #[tokio::test]
    async fn test_the_forced_off_is_sent_whatever_the_state() {
        let (mut indicator, vrchat, _) = indicator_and_two_vrchats().await;
        let config = sending_to(&vrchat);

        indicator.force_stop_typing(&config).await;
        indicator.start_typing(&config).await;
        indicator.force_stop_typing(&config).await;
        indicator.force_stop_typing(&config).await;
        // The forced off turned it off
        indicator.stop_typing(&config).await;
        indicator.start_typing(&config).await;

        for expected in [
            Osc::Typing(false),
            Osc::Typing(true),
            Osc::Typing(false),
            Osc::Typing(false),
            Osc::Typing(true),
        ] {
            assert_eq!(received(&vrchat).await, expected);
        }
    }

    #[tokio::test]
    async fn test_an_on_whose_send_failed_still_counts_as_on() {
        let (mut indicator, vrchat, _) = indicator_and_two_vrchats().await;
        let mut config = Config::default();
        // An IPv4 socket cannot send to an IPv6 address
        config.osc.address = "[::1]".to_string();

        indicator.start_typing(&config).await;

        assert!(indicator.stop_typing_elsewhere(&sending_to(&vrchat)).await);
    }

    #[tokio::test]
    async fn test_a_failed_send_is_logged_once_until_it_changes() {
        let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let receiver_port = receiver.local_addr().unwrap().port();
        let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
        let vrchat_at = |address: &str, port: u16| {
            let mut config = Config::default();
            config.osc.address = address.to_string();
            config.osc.output_port = port;
            config
        };
        let (log_tx, mut log_rx) = mpsc::channel(10);
        let mut indicator = TypingIndicator::new(socket, Logger::new(log_tx, Default::default()));
        let one_error_naming = |entries: Vec<(LogLevel, String)>, address: &str| {
            assert_eq!(entries.len(), 1, "{:?}", entries);
            assert_eq!(entries[0].0, LogLevel::Error, "{:?}", entries);
            assert!(entries[0].1.contains(address), "{:?}", entries);
        };

        // An IPv4 socket cannot send to an IPv6 address: every send fails
        let config = vrchat_at("[::1]", 9000);
        indicator.start_typing(&config).await;
        one_error_naming(new_entries(&mut log_rx), "[::1]:9000");
        indicator.stop_typing(&config).await;
        indicator.start_typing(&config).await;
        assert_eq!(new_entries(&mut log_rx), []);

        // A different address fails with a different message
        let config = vrchat_at("[::1]", 9001);
        indicator.force_stop_typing(&config).await;
        one_error_naming(new_entries(&mut log_rx), "[::1]:9001");

        // The send works again
        let config = vrchat_at("127.0.0.1", receiver_port);
        indicator.start_typing(&config).await;
        let entries = new_entries(&mut log_rx);
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert_eq!(entries[0].0, LogLevel::Info, "{:?}", entries);
        let mut buf = [0u8; 256];
        let (len, _) = receiver.recv_from(&mut buf).await.unwrap();
        assert!(len > 0);
        indicator.stop_typing(&config).await;
        assert_eq!(new_entries(&mut log_rx), []);

        // The same failure as before is logged again after a send that worked
        let config = vrchat_at("[::1]", 9001);
        indicator.start_typing(&config).await;
        one_error_naming(new_entries(&mut log_rx), "[::1]:9001");
    }
}
