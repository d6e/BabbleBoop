use async_std::net::UdpSocket;
use bytes::BytesMut;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavWriter;
use rosc::{encoder::encode, OscMessage, OscPacket, OscType};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;


#[derive(Deserialize)]
struct Config {
    osc: OscConfig,
    openai: OpenAiConfig,
    translation: TranslationConfig,
    audio: AudioConfig,
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

#[derive(Deserialize)]
struct AudioConfig {
    recording_duration: f32,
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

struct RateLimiter {
    last_request: Instant,
    request_count: usize,
}

impl RateLimiter {
    fn new() -> Self {
        RateLimiter {
            last_request: Instant::now(),
            request_count: 0,
        }
    }

    async fn wait(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_request);

        if elapsed < Duration::from_secs(60) {
            if self.request_count >= 3 {
                let wait_time = Duration::from_secs(60) - elapsed;
                sleep(wait_time).await;
                self.request_count = 0;
                self.last_request = Instant::now();
            }
        } else {
            self.request_count = 0;
            self.last_request = now;
        }

        self.request_count += 1;
    }
}

async fn transcribe_audio(audio_data: Vec<u8>, config: &OpenAiConfig, rate_limiter: &mut RateLimiter) -> Result<String, Box<dyn Error>> {
    println!("Starting audio transcription. Audio data size: {} bytes", audio_data.len());
    
    if audio_data.is_empty() {
        return Err("Audio data is empty".into());
    }

    rate_limiter.wait().await;

    let client = reqwest::Client::new();
    let part = reqwest::multipart::Part::bytes(audio_data)
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", "whisper-1");

    println!("Sending request to OpenAI Whisper API");
    let res = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {}", &config.api_key))
        .multipart(form)
        .send()
        .await?;

    if !res.status().is_success() {
        let error_text = res.text().await?;
        return Err(format!("API request failed: {}", error_text).into());
    }

    #[derive(Deserialize)]
    struct TranscriptionResponse {
        text: String,
    }

    let transcription: TranscriptionResponse = res.json().await?;
    println!("Transcription received: {}", transcription.text);

    if transcription.text.is_empty() {
        return Err("Received empty transcription from API".into());
    }

    Ok(transcription.text)
}

async fn process_audio(
    audio_data: Vec<u8>,
    config: &Config,
    socket: &UdpSocket,
    rate_limiter: &mut RateLimiter,
) -> Result<(), Box<dyn Error>> {
    let transcription = transcribe_audio(audio_data, &config.openai, rate_limiter).await?;
    println!("Transcription: {}", transcription);

    let translation_prompt = format!(
        "Translate the following text to {}: \"{}\"",
        config.translation.target_language, transcription
    );

    let response = ask_chatgpt(&translation_prompt, &config.openai).await?;
    println!("Translation: {}", response);

    send_to_chatbox(&response, &config, socket).await?;

    Ok(())
}

fn record_audio(config: &Config) -> Result<Vec<u8>, Box<dyn Error>> {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    let device_config = device.default_input_config()?;

    let sample_rate = device_config.sample_rate().0 as f32;
    let channels = device_config.channels() as usize;
    let sample_format = device_config.sample_format();

    let audio_data = Arc::new(Mutex::new(Vec::new()));
    let audio_data_clone = Arc::clone(&audio_data);

    let err_fn = |err| eprintln!("An error occurred on the audio stream: {}", err);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &device_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut buffer = audio_data_clone.lock().unwrap();
                buffer.extend_from_slice(data);
            },
            err_fn,
            None, // Add this line for the buffer size option
        )?,
        _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;
    std::thread::sleep(std::time::Duration::from_secs_f32(
        config.audio.recording_duration,
    ));
    drop(stream);

    let audio_data = audio_data.lock().unwrap();
    let mut wav_buffer = Vec::new();
    {
        let mut writer = WavWriter::new(
            Cursor::new(&mut wav_buffer),
            hound::WavSpec {
                channels: channels as u16,
                sample_rate: sample_rate as u32,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;

        for &sample in audio_data.iter() {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }

    Ok(wav_buffer)
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config_data = fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_data)?;

    let socket_address = format!("{}:{}", config.osc.address, config.osc.input_port);
    let socket = UdpSocket::bind(socket_address).await?;

    println!("Listening for audio input...");
    println!("Translating to: {}", config.translation.target_language);

    let mut rate_limiter = RateLimiter::new();

    loop {
        let audio_data = record_audio(&config)?;
        process_audio(audio_data, &config, &socket, &mut rate_limiter).await?;
    }
}