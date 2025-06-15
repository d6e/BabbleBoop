use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::watch;
use tokio::time::{sleep, timeout};
use rosc::{OscPacket, OscMessage, OscType, encoder::encode, decoder};

/// Helper to create a test OSC message
fn create_test_message(addr: &str, args: Vec<OscType>) -> Vec<u8> {
    let msg = OscMessage { addr: addr.to_string(), args };
    let packet = OscPacket::Message(msg);
    encode(&packet).unwrap()
}

/// Helper to receive and decode OSC messages
async fn receive_osc_message(socket: &UdpSocket, timeout_duration: Duration) -> Result<(OscPacket, std::net::SocketAddr), String> {
    let mut buf = [0u8; 65536];
    match timeout(timeout_duration, socket.recv_from(&mut buf)).await {
        Ok(Ok((size, addr))) => {
            match decoder::decode_udp(&buf[..size]) {
                Ok((_, packet)) => Ok((packet, addr)),
                Err(e) => Err(format!("Failed to decode OSC packet: {}", e)),
            }
        }
        Ok(Err(e)) => Err(format!("Socket receive error: {}", e)),
        Err(_) => Err("Timeout waiting for message".to_string()),
    }
}

#[tokio::test]
async fn test_large_packet_handling() {
    // Test handling of packets near the maximum buffer size
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19010,
            output_port: 19054,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19029,
        },
        openai: babble_boop::config::OpenAiConfig {
            api_key: "test".to_string(),
            model: "gpt-4o-mini".to_string(),
        },
        translation: babble_boop::config::TranslationConfig {
            target_language: "Japanese".to_string(),
            include_original_message: false,
        },
        audio: babble_boop::config::AudioConfig {
            silence_threshold: 100,
            noise_gate_threshold: 0.3,
            noise_gate_hold_time: 0.20,
            min_transcription_duration: 1.0,
        },
        rate_limit: babble_boop::config::RateLimitConfig {
            requests_per_minute: 50,
        },
        debug: false,
    });
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19010").await.unwrap());
    let listener_socket = UdpSocket::bind("127.0.0.1:19054").await.unwrap();
    let sender_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    let passthrough_handle = tokio::spawn(async move {
        let _ = passthrough.start_passthrough(shutdown_rx).await;
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Create a large string (but still valid for OSC)
    let large_string = "X".repeat(60000); // Large but under 65536 with OSC overhead
    let large_msg = create_test_message("/test/large", vec![OscType::String(large_string.clone())]);
    
    // Send the large message
    sender_socket.send_to(&large_msg, "127.0.0.1:19029").await.unwrap();
    
    // Should receive the large message
    match receive_osc_message(&listener_socket, Duration::from_secs(2)).await {
        Ok((packet, _)) => {
            if let OscPacket::Message(msg) = packet {
                assert_eq!(msg.addr, "/test/large");
                assert!(!msg.args.is_empty());
                if let OscType::String(s) = &msg.args[0] {
                    assert_eq!(s.len(), 60000, "Large string should be preserved");
                    assert_eq!(s, &large_string);
                }
            }
        }
        Err(e) => panic!("Failed to receive large message: {}", e),
    }
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}

#[tokio::test]
async fn test_invalid_packet_ordering() {
    // Test that ensures we properly handle invalid packets without affecting valid ones
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19011,
            output_port: 19055,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19030,
        },
        openai: babble_boop::config::OpenAiConfig {
            api_key: "test".to_string(),
            model: "gpt-4o-mini".to_string(),
        },
        translation: babble_boop::config::TranslationConfig {
            target_language: "Japanese".to_string(),
            include_original_message: false,
        },
        audio: babble_boop::config::AudioConfig {
            silence_threshold: 100,
            noise_gate_threshold: 0.3,
            noise_gate_hold_time: 0.20,
            min_transcription_duration: 1.0,
        },
        rate_limit: babble_boop::config::RateLimitConfig {
            requests_per_minute: 50,
        },
        debug: true,
    });
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19011").await.unwrap());
    let listener_socket = UdpSocket::bind("127.0.0.1:19055").await.unwrap();
    let sender_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    let passthrough_handle = tokio::spawn(async move {
        let _ = passthrough.start_passthrough(shutdown_rx).await;
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Send interleaved valid and invalid packets
    let valid_msg1 = create_test_message("/test/msg1", vec![OscType::Int(1)]);
    sender_socket.send_to(&valid_msg1, "127.0.0.1:19030").await.unwrap();
    sleep(Duration::from_millis(10)).await;
    
    sender_socket.send_to(b"INVALID PACKET", "127.0.0.1:19030").await.unwrap();
    sleep(Duration::from_millis(10)).await;
    
    let valid_msg2 = create_test_message("/test/msg2", vec![OscType::Int(2)]);
    sender_socket.send_to(&valid_msg2, "127.0.0.1:19030").await.unwrap();
    sleep(Duration::from_millis(10)).await;
    
    sender_socket.send_to(&[0xFF, 0xFF, 0xFF], "127.0.0.1:19030").await.unwrap();
    sleep(Duration::from_millis(10)).await;
    
    let valid_msg3 = create_test_message("/test/msg3", vec![OscType::Int(3)]);
    sender_socket.send_to(&valid_msg3, "127.0.0.1:19030").await.unwrap();
    
    // Should receive exactly 3 valid messages in order
    let mut received_values = vec![];
    for _ in 0..3 {
        match receive_osc_message(&listener_socket, Duration::from_secs(1)).await {
            Ok((packet, _)) => {
                if let OscPacket::Message(msg) = packet {
                    assert!(msg.addr.starts_with("/test/msg"));
                    assert!(!msg.args.is_empty());
                    if let OscType::Int(val) = msg.args[0] {
                        received_values.push(val);
                    }
                }
            }
            Err(e) => panic!("Failed to receive valid message: {}", e),
        }
    }
    
    assert_eq!(received_values, vec![1, 2, 3], "Should receive all valid messages in order");
    
    // Ensure no more messages (invalid ones were dropped)
    match receive_osc_message(&listener_socket, Duration::from_millis(200)).await {
        Err(e) if e.contains("Timeout") => {
            // Good, no more messages
        }
        Ok(_) => panic!("Received unexpected message"),
        Err(e) => panic!("Unexpected error: {}", e),
    }
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}

#[tokio::test]  
async fn test_empty_packet_handling() {
    // Test that empty packets are properly ignored
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19012,
            output_port: 19056,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19031,
        },
        openai: babble_boop::config::OpenAiConfig {
            api_key: "test".to_string(),
            model: "gpt-4o-mini".to_string(),
        },
        translation: babble_boop::config::TranslationConfig {
            target_language: "Japanese".to_string(),
            include_original_message: false,
        },
        audio: babble_boop::config::AudioConfig {
            silence_threshold: 100,
            noise_gate_threshold: 0.3,
            noise_gate_hold_time: 0.20,
            min_transcription_duration: 1.0,
        },
        rate_limit: babble_boop::config::RateLimitConfig {
            requests_per_minute: 50,
        },
        debug: true,
    });
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19012").await.unwrap());
    let listener_socket = UdpSocket::bind("127.0.0.1:19056").await.unwrap();
    let sender_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    let passthrough_handle = tokio::spawn(async move {
        let _ = passthrough.start_passthrough(shutdown_rx).await;
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Send empty packet (should be ignored due to size > 0 check)
    sender_socket.send_to(b"", "127.0.0.1:19031").await.unwrap();
    sleep(Duration::from_millis(50)).await;
    
    // Send valid message
    let valid_msg = create_test_message("/test/after_empty", vec![OscType::String("test".to_string())]);
    sender_socket.send_to(&valid_msg, "127.0.0.1:19031").await.unwrap();
    
    // Should only receive the valid message
    match receive_osc_message(&listener_socket, Duration::from_secs(1)).await {
        Ok((packet, _)) => {
            if let OscPacket::Message(msg) = packet {
                assert_eq!(msg.addr, "/test/after_empty");
            }
        }
        Err(e) => panic!("Failed to receive valid message: {}", e),
    }
    
    // Ensure no empty packet was forwarded
    match receive_osc_message(&listener_socket, Duration::from_millis(200)).await {
        Err(e) if e.contains("Timeout") => {
            // Good, empty packet was not forwarded
        }
        Ok(_) => panic!("Received unexpected packet"),
        Err(e) => panic!("Unexpected error: {}", e),
    }
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}