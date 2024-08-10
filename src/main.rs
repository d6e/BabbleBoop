use async_std::net::UdpSocket;
use rosc::{encoder::encode, OscMessage, OscPacket, OscType};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;

#[derive(Deserialize)]
struct Config {
    osc: OscConfig,
    openai: OpenAiConfig,
    translation: TranslationConfig,
}

#[derive(Deserialize)]
struct OscConfig {
    address: String,
    input_port: u16,
    output_port: u16,
    input_address: String,
    output_address: String,
    max_message_chunks: usize,
    display_time: u64,
}

#[derive(Deserialize)]
struct OpenAiConfig {
    api_key: String,
    model: String,
}

#[derive(Deserialize)]
struct TranslationConfig {
    target_language: String,
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

async fn send_to_chatbox(message: &str, config: &Config, socket: &UdpSocket) -> Result<(), Box<dyn Error>> {
    // Set typing indicator to true
    let typing_on = OscMessage {
        addr: "/chatbox/typing".to_string(),
        args: vec![OscType::Bool(true)],
    };
    let buf = encode(&OscPacket::Message(typing_on))?;
    let osc_address = format!("{}:{}", config.osc.address, config.osc.output_port);
    socket.send_to(&buf, osc_address.as_str()).await?;

    // Split message into chunks of 144 characters or less, respecting Unicode character boundaries
    let chunks: Vec<String> = message
        .chars()
        .collect::<Vec<char>>()
        .chunks(144)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect();

    // Send each chunk as a separate message
    for (i, chunk) in chunks.iter().enumerate().take(config.osc.max_message_chunks) {
        let osc_message = OscMessage {
            addr: "/chatbox/input".to_string(),
            args: vec![
                OscType::String(chunk.to_string()),
                OscType::Bool(true), // Send immediately
                OscType::Bool(i == 0),  // Trigger notification only for the first chunk
            ],
        };

        let buf = encode(&OscPacket::Message(osc_message))?;
        socket.send_to(&buf, osc_address.as_str()).await?;

        // Add a small delay between messages to ensure proper order
        tokio::time::sleep(tokio::time::Duration::from_millis(config.osc.display_time)).await;
    }

    // Set typing indicator to false
    let typing_off = OscMessage {
        addr: "/chatbox/typing".to_string(),
        args: vec![OscType::Bool(false)],
    };
    let buf = encode(&OscPacket::Message(typing_off))?;
    socket.send_to(&buf, osc_address.as_str()).await?;

    Ok(())
}

async fn osc_message_handler(
    packet: OscPacket,
    config: &Config,
    socket: &UdpSocket,
) -> Result<(), Box<dyn Error>> {
    if let OscPacket::Message(msg) = packet {
        if msg.addr == config.osc.input_address {
            if let Some(OscType::String(input)) = msg.args.get(0) {
                println!("Received OSC message: {}", input);

                let translation_prompt = format!(
                    "Translate the following text to {}: \"{}\"",
                    config.translation.target_language, input
                );

                let response = ask_chatgpt(&translation_prompt, &config.openai).await?;
                println!("ChatGPT response: {}", response);

                send_to_chatbox(&response, &config, socket).await?;
            }
        } else {
            println!("Received OSC message: {}", msg.addr);
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load configuration from config.toml
    let config_data = fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_data)?;

    // Bind the socket to the configured address and input port
    let socket_address = format!("{}:{}", config.osc.address, config.osc.input_port);
    let socket = UdpSocket::bind(socket_address).await?;
    let mut buf = [0u8; 1024];

    println!("Listening for OSC messages on {}:{}", config.osc.address, config.osc.input_port);
    println!("Sending responses to port {}", config.osc.output_port);
    println!("Translating to: {}", config.translation.target_language);
   

    loop {
        let (size, _addr) = socket.recv_from(&mut buf).await?;
        if let Ok((_, packet)) = rosc::decoder::decode_udp(&buf[..size]) {
            osc_message_handler(packet, &config, &socket).await?;
        }
    }
}