use crate::config::Config;
use rosc::{encoder::encode, OscMessage, OscPacket, OscType};
use std::error::Error;
use tokio::net::UdpSocket;
use tokio::time::sleep;

/// VRChat chatbox has a 144 character limit per message
const VRCHAT_CHATBOX_CHAR_LIMIT: usize = 144;

pub async fn send_to_chatbox(
    message: &str,
    config: &Config,
    socket: &UdpSocket,
) -> Result<(), Box<dyn Error>> {
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
        let osc_message = OscMessage {
            addr: "/chatbox/input".to_string(),
            args: vec![
                OscType::String(chunk.to_string()),
                OscType::Bool(true),
                OscType::Bool(i == 0),
            ],
        };

        let buf = encode(&OscPacket::Message(osc_message))?;
        socket.send_to(&buf, osc_address.as_str()).await?;

        sleep(tokio::time::Duration::from_millis(config.osc.display_time)).await;
    }

    Ok(())
}
