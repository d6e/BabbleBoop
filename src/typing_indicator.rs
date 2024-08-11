use crate::config::Config;
use rosc::{encoder::encode, OscMessage, OscPacket, OscType};
use std::sync::Arc;
use tokio::net::UdpSocket;

#[derive(Clone)]
pub struct TypingIndicator {
    socket: Arc<UdpSocket>,
    config: Arc<Config>,
}

impl TypingIndicator {
    pub fn new(socket: Arc<UdpSocket>, config: Arc<Config>) -> Self {
        TypingIndicator { socket, config }
    }

    async fn set_typing(&self, is_typing: bool) {
        let typing_message = OscMessage {
            addr: "/chatbox/typing".to_string(),
            args: vec![OscType::Bool(is_typing)],
        };
        if let Ok(buf) = encode(&OscPacket::Message(typing_message)) {
            let osc_address = format!(
                "{}:{}",
                self.config.osc.address, self.config.osc.output_port
            );
            if let Err(e) = self.socket.send_to(&buf, osc_address.as_str()).await {
                eprintln!("Error sending typing indicator: {}", e);
            }
        }
    }

    pub async fn start_typing(&self) {
        self.set_typing(true).await;
    }

    pub async fn stop_typing(&self) {
        self.set_typing(false).await;
    }
}
