use async_std::net::UdpSocket;
use rosc::{decoder, encoder::encode, OscMessage, OscPacket, OscType};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

#[derive(Deserialize)]
struct Config {
    osc: OscConfig,
    openai: OpenAiConfig,
}

#[derive(Deserialize)]
struct OscConfig {
    address: String,
    port: u16,
    response_address: String,
}

#[derive(Deserialize)]
struct OpenAiConfig {
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct ChatGptRequest {
    model: String,
    messages: Vec<ChatGptMessage>,
}

#[derive(Serialize, Deserialize)]
struct ChatGptMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatGptResponse {
    choices: Vec<ChatGptChoice>,
}

#[derive(Deserialize)]
struct ChatGptChoice {
    message: ChatGptMessage,
}

async fn ask_chatgpt(prompt: &str, config: &OpenAiConfig) -> Result<String, Box<dyn Error>> {
    let client = reqwest::Client::new();

    let request_body = ChatGptRequest {
        model: config.model.clone(),
        messages: vec![ChatGptMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let res = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(&config.api_key)
        .json(&request_body)
        .send()
        .await?;

    let res_body: ChatGptResponse = res.json().await?;
    Ok(res_body.choices[0].message.content.clone())
}

async fn osc_message_handler(
    packet: OscPacket,
    config: &Config,
    socket: &UdpSocket,
) -> Result<(), Box<dyn Error>> {
    if let OscPacket::Message(msg) = packet {
        if msg.addr == "/chatgpt/translate" {
            if let Some(OscType::String(input)) = msg.args.get(0) {
                println!("Received OSC message: {}", input);

                let response = ask_chatgpt(input, &config.openai).await?;
                println!("ChatGPT response: {}", response);

                let osc_response = OscMessage {
                    addr: config.osc.response_address.clone(),
                    args: vec![OscType::String(response)],
                };

                let buf = encode(&OscPacket::Message(osc_response))?;
                let osc_address = format!("{}:{}", config.osc.address, config.osc.port);
                socket.send_to(&buf, osc_address).await?;
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load configuration from config.toml
    let config_data = fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_data)?;

    // Bind the socket to the configured address and port
    let socket_address = format!("{}:{}", config.osc.address, config.osc.port);
    let socket = UdpSocket::bind(socket_address).await?;
    let mut buf = [0u8; 1024];

    println!("Listening for OSC messages on {}:{}", config.osc.address, config.osc.port);

    loop {
        let (size, _addr) = socket.recv_from(&mut buf).await?;
        if let Ok((_, packet)) = decoder::decode_udp(&buf[..size]) {
            osc_message_handler(packet, &config, &socket).await?;
        }
    }
}
