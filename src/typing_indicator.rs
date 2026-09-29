use crate::app_state::{FailureLog, Logger};
use crate::config::Config;
use rosc::{encoder::encode, OscMessage, OscPacket, OscType};
use std::sync::{Arc, Mutex, PoisonError};
use tokio::net::UdpSocket;

/// The typing indicator of the VRChat chatbox. Like `Chatbox`, it takes
/// the settings with each call, so it always sends to the address in the
/// settings of the caller, and holds no copy of them.
pub struct TypingIndicator {
    socket: Arc<UdpSocket>,
    logger: Logger,
    /// A recording turns the indicator on once and off up to twice, and a
    /// long recording turns it on again after each part. While VRChat
    /// cannot be reached at the configured address, every send fails the
    /// same way.
    send_failure: Mutex<FailureLog>,
}

impl TypingIndicator {
    pub fn new(socket: Arc<UdpSocket>, logger: Logger) -> Self {
        TypingIndicator {
            socket,
            logger,
            send_failure: Mutex::default(),
        }
    }

    /// Send the typing state to the VRChat chatbox at the address in
    /// `config`. A failed send goes to the activity log, once until a send
    /// works again or fails with a different message.
    async fn set_typing(&self, is_typing: bool, config: &Config) {
        let typing_message = OscMessage {
            addr: "/chatbox/typing".to_string(),
            args: vec![OscType::Bool(is_typing)],
        };
        if let Ok(buf) = encode(&OscPacket::Message(typing_message)) {
            let osc_address = format!("{}:{}", config.osc.address, config.osc.output_port);
            let result = self.socket.send_to(&buf, osc_address.as_str()).await;
            // The lock only guards the last failure, which stays valid
            let mut send_failure = self
                .send_failure
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            match result {
                Ok(_) => {
                    if send_failure.succeeded() {
                        self.logger
                            .info(format!("Typing indicator sent to {} again", osc_address));
                    }
                }
                Err(e) => send_failure.failed(
                    &self.logger,
                    format!("Cannot send the typing indicator to {}: {}", osc_address, e),
                ),
            }
        }
    }

    pub async fn start_typing(&self, config: &Config) {
        self.set_typing(true, config).await;
    }

    pub async fn stop_typing(&self, config: &Config) {
        self.set_typing(false, config).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{LogEntry, LogLevel};
    use tokio::sync::mpsc;

    /// Level and text of the entries in the activity log since the last call.
    fn new_entries(log_rx: &mut mpsc::Receiver<LogEntry>) -> Vec<(LogLevel, String)> {
        std::iter::from_fn(|| log_rx.try_recv().ok())
            .map(|entry| (entry.level, entry.message))
            .collect()
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
        let indicator = TypingIndicator::new(socket, Logger::new(log_tx, Default::default()));
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
        indicator.stop_typing(&config).await;
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
