use std::sync::Arc;
use std::time::Duration;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, watch};
use tokio::time::{sleep, timeout};
use rosc::{OscPacket, OscMessage, OscType, OscBundle, OscTime, encoder::encode, decoder};

/// Test configuration for OSC passthrough tests
struct TestConfig {
    passthrough_port: u16,
    output_port: u16,
}

impl Default for TestConfig {
    fn default() -> Self {
        TestConfig {
            passthrough_port: 19015, // Use different ports to avoid conflicts
            output_port: 19000,
        }
    }
}

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
async fn test_basic_osc_forwarding() {
    let config = TestConfig::default();
    
    // Create temporary config for the test
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19001,
            output_port: config.output_port,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: config.passthrough_port,
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
    
    // Create main socket
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19001").await.unwrap());
    
    // Create listener socket on output port
    let listener_socket = UdpSocket::bind(format!("127.0.0.1:{}", config.output_port)).await.unwrap();
    
    // Create sender socket
    let sender_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    
    // Start passthrough
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    let passthrough_handle = tokio::spawn(async move {
        let _ = passthrough.start_passthrough(shutdown_rx).await;
    });
    
    // Give passthrough time to start
    sleep(Duration::from_millis(100)).await;
    
    // Test 1: Send a simple message
    let test_msg = create_test_message("/test/simple", vec![OscType::String("Hello".to_string())]);
    sender_socket.send_to(&test_msg, format!("127.0.0.1:{}", config.passthrough_port)).await.unwrap();
    
    // Receive and verify
    match receive_osc_message(&listener_socket, Duration::from_secs(1)).await {
        Ok((packet, _)) => {
            if let OscPacket::Message(msg) = packet {
                assert_eq!(msg.addr, "/test/simple");
                assert_eq!(msg.args.len(), 1);
                if let OscType::String(s) = &msg.args[0] {
                    assert_eq!(s, "Hello");
                } else {
                    panic!("Expected string argument");
                }
            } else {
                panic!("Expected OSC message, got bundle");
            }
        }
        Err(e) => panic!("Failed to receive forwarded message: {}", e),
    }
    
    // Cleanup
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}

#[tokio::test]
async fn test_facetracking_parameters() {
    let config = TestConfig::default();
    
    // Similar setup as above
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19002,
            output_port: 19010,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19016,
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
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19002").await.unwrap());
    let listener_socket = UdpSocket::bind(format!("127.0.0.1:{}", 19010)).await.unwrap();
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
    
    // Send facetracking parameters
    let params = vec![
        ("/avatar/parameters/EyeLeftX", 0.5f32),
        ("/avatar/parameters/EyeRightX", 0.5f32),
        ("/avatar/parameters/MouthOpen", 0.3f32),
    ];
    
    for (addr, value) in &params {
        let msg = create_test_message(addr, vec![OscType::Float(*value)]);
        sender_socket.send_to(&msg, format!("127.0.0.1:{}", 19016)).await.unwrap();
    }
    
    // Verify all messages were forwarded
    let mut received = 0;
    for _ in 0..params.len() {
        match receive_osc_message(&listener_socket, Duration::from_secs(1)).await {
            Ok((packet, _)) => {
                if let OscPacket::Message(msg) = packet {
                    // Find matching parameter
                    for (addr, expected_value) in &params {
                        if msg.addr == *addr {
                            assert!(!msg.args.is_empty(), "Message should have arguments");
                            if let OscType::Float(value) = msg.args[0] {
                                assert_eq!(value, *expected_value);
                                received += 1;
                                break;
                            }
                        }
                    }
                }
            }
            Err(e) => panic!("Failed to receive facetracking parameter: {}", e),
        }
    }
    
    assert_eq!(received, params.len(), "Not all facetracking parameters were forwarded");
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}

#[tokio::test]
async fn test_invalid_packet_dropping() {
    let config = TestConfig::default();
    
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19003,
            output_port: 19020,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19017,
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
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19003").await.unwrap());
    let listener_socket = UdpSocket::bind(format!("127.0.0.1:{}", 19020)).await.unwrap();
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
    
    // Send invalid packets with small delays to ensure ordering
    let invalid_packets = vec![
        b"Not an OSC packet".to_vec(),
        b"Random garbage \x00\x01\x02".to_vec(),
        b"".to_vec(), // Empty packet
        vec![0xFF; 100], // Random bytes
    ];
    
    for packet in &invalid_packets {
        sender_socket.send_to(packet, format!("127.0.0.1:{}", 19017)).await.unwrap();
        sleep(Duration::from_millis(10)).await; // Small delay to ensure processing
    }
    
    // Wait a bit before sending valid packet to ensure all invalid ones are processed
    sleep(Duration::from_millis(50)).await;
    
    // Send one valid packet
    let valid_msg = create_test_message("/test/valid", vec![OscType::Int(42)]);
    sender_socket.send_to(&valid_msg, format!("127.0.0.1:{}", 19017)).await.unwrap();
    
    // Should only receive the valid packet
    match receive_osc_message(&listener_socket, Duration::from_secs(1)).await {
        Ok((packet, _)) => {
            if let OscPacket::Message(msg) = packet {
                assert_eq!(msg.addr, "/test/valid");
                assert!(!msg.args.is_empty(), "Message should have arguments");
                if let OscType::Int(value) = msg.args[0] {
                    assert_eq!(value, 42);
                }
            }
        }
        Err(e) => panic!("Failed to receive valid message: {}", e),
    }
    
    // Verify no invalid packets were forwarded (should timeout)
    match receive_osc_message(&listener_socket, Duration::from_millis(500)).await {
        Err(e) if e.contains("Timeout") => {
            // Good, no invalid packets were forwarded
        }
        Ok(_) => panic!("Received unexpected packet - invalid packet was forwarded"),
        Err(e) => panic!("Unexpected error: {}", e),
    }
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}

#[tokio::test]
async fn test_concurrent_operations() {
    let config = TestConfig::default();
    
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19004,
            output_port: 19030,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19018,
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
        debug: false, // Disable debug for performance test
    });
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19004").await.unwrap());
    let listener_socket = UdpSocket::bind(format!("127.0.0.1:{}", 19030)).await.unwrap();
    
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    let passthrough_handle = tokio::spawn(async move {
        let _ = passthrough.start_passthrough(shutdown_rx).await;
    });
    
    sleep(Duration::from_millis(100)).await;
    
    // Spawn multiple senders
    let (tx, mut rx) = mpsc::channel(100);
    let mut sender_handles = vec![];
    
    // Sender 1: Rapid facetracking data
    let tx1 = tx.clone();
    let handle1 = tokio::spawn(async move {
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        for i in 0..50 {
            let value = (i as f32) / 100.0;
            let msg = create_test_message("/avatar/parameters/TestParam", vec![OscType::Float(value)]);
            socket.send_to(&msg, format!("127.0.0.1:{}", 19018)).await.unwrap();
            let _ = tx1.send(1).await;
            sleep(Duration::from_millis(10)).await;
        }
    });
    sender_handles.push(handle1);
    
    // Sender 2: Chatbox messages
    let tx2 = tx.clone();
    let handle2 = tokio::spawn(async move {
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        for i in 0..10 {
            let msg = create_test_message("/chatbox/test", vec![OscType::String(format!("Message {}", i))]);
            socket.send_to(&msg, format!("127.0.0.1:{}", 19018)).await.unwrap();
            let _ = tx2.send(1).await;
            sleep(Duration::from_millis(50)).await;
        }
    });
    sender_handles.push(handle2);
    
    drop(tx); // Close original sender
    
    // Count received messages
    let receiver_handle = tokio::spawn(async move {
        let mut count = 0;
        loop {
            match receive_osc_message(&listener_socket, Duration::from_millis(100)).await {
                Ok(_) => count += 1,
                Err(e) if e.contains("Timeout") => break,
                Err(e) => eprintln!("Receive error: {}", e),
            }
        }
        count
    });
    
    // Wait for all senders to complete
    for handle in sender_handles {
        handle.await.unwrap();
    }
    
    // Count sent messages
    let mut sent_count = 0;
    while let Ok(_) = rx.try_recv() {
        sent_count += 1;
    }
    
    // Wait a bit more for messages to be forwarded
    sleep(Duration::from_millis(200)).await;
    
    // Get received count
    let received_count = receiver_handle.await.unwrap();
    
    // Verify most messages were forwarded (allow for some UDP loss)
    assert!(received_count >= (sent_count * 95) / 100, 
            "Too many messages lost: sent {}, received {}", sent_count, received_count);
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}

#[tokio::test]
async fn test_graceful_shutdown() {
    let config = TestConfig::default();
    
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19005,
            output_port: 19040,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19019,
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
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19005").await.unwrap());
    
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    let passthrough_handle = tokio::spawn(async move {
        let _ = passthrough.start_passthrough(shutdown_rx).await;
    });
    
    // Give it time to start
    sleep(Duration::from_millis(100)).await;
    
    // Send shutdown signal
    let _ = shutdown_tx.send(true);
    
    // Verify it shuts down quickly
    match timeout(Duration::from_secs(2), passthrough_handle).await {
        Ok(Ok(_)) => {
            // Good, shut down successfully
        }
        Ok(Err(e)) => panic!("Passthrough task panicked: {:?}", e),
        Err(_) => panic!("Passthrough did not shut down within timeout"),
    }
}

#[tokio::test]
async fn test_passthrough_disabled() {
    // Test that passthrough doesn't start when disabled
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19006,
            output_port: 19050,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: false, // DISABLED
            passthrough_port: 19025,
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
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19006").await.unwrap());
    
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    // This should return immediately without binding any socket
    let result = passthrough.start_passthrough(shutdown_rx).await;
    assert!(result.is_ok(), "Passthrough should return Ok when disabled");
    
    // Verify port 19025 is not bound by trying to bind to it
    match UdpSocket::bind("127.0.0.1:19025").await {
        Ok(_) => {
            // Good, we could bind, meaning passthrough didn't bind
        }
        Err(_) => {
            panic!("Port 19025 is in use - passthrough shouldn't bind when disabled");
        }
    }
    
    let _ = shutdown_tx.send(true);
}

#[tokio::test]
async fn test_shared_socket_source_port() {
    // Test that forwarded messages come from the main socket's port
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19007, // Main socket binds here
            output_port: 19051,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19026,
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
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19007").await.unwrap());
    let listener_socket = UdpSocket::bind("127.0.0.1:19051").await.unwrap();
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
    
    // Send a message
    let test_msg = create_test_message("/test/source", vec![OscType::String("check source".to_string())]);
    sender_socket.send_to(&test_msg, "127.0.0.1:19026").await.unwrap();
    
    // Receive and verify source port
    match receive_osc_message(&listener_socket, Duration::from_secs(1)).await {
        Ok((packet, src_addr)) => {
            // Messages should come from port 19007 (main socket port)
            assert_eq!(src_addr.port(), 19007, "Message should come from main socket port");
            
            if let OscPacket::Message(msg) = packet {
                assert_eq!(msg.addr, "/test/source");
            }
        }
        Err(e) => panic!("Failed to receive message: {}", e),
    }
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}

#[tokio::test]
async fn test_socket_bind_failure() {
    // Test error handling when passthrough port is already in use
    
    // First, bind the port that passthrough will try to use
    let _blocking_socket = UdpSocket::bind("127.0.0.1:19027").await.unwrap();
    
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19008,
            output_port: 19052,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19027, // Already in use!
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
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19008").await.unwrap());
    
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let passthrough = babble_boop::osc_passthrough::OscPassthrough::new(
        Arc::clone(&main_socket),
        Arc::clone(&test_config),
    );
    
    // This should fail because port is already in use
    let result = passthrough.start_passthrough(shutdown_rx).await;
    assert!(result.is_err(), "Passthrough should fail when port is in use");
}

#[tokio::test]
async fn test_osc_bundle_forwarding() {
    // Test that OSC bundles are properly forwarded
    let test_config = Arc::new(babble_boop::config::Config {
        osc: babble_boop::config::OscConfig {
            address: "127.0.0.1".to_string(),
            input_port: 19009,
            output_port: 19053,
            max_message_chunks: 9,
            display_time: 3000,
            passthrough_enabled: true,
            passthrough_port: 19028,
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
    
    let main_socket = Arc::new(UdpSocket::bind("127.0.0.1:19009").await.unwrap());
    let listener_socket = UdpSocket::bind("127.0.0.1:19053").await.unwrap();
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
    
    // Create and send an OSC bundle
    let msg1 = OscPacket::Message(OscMessage {
        addr: "/bundle/msg1".to_string(),
        args: vec![OscType::Int(1)],
    });
    let msg2 = OscPacket::Message(OscMessage {
        addr: "/bundle/msg2".to_string(),
        args: vec![OscType::Float(2.5)],
    });
    
    let bundle = OscBundle {
        timetag: OscTime { seconds: 0, fractional: 1 },
        content: vec![msg1, msg2],
    };
    
    let bundle_packet = OscPacket::Bundle(bundle);
    let bundle_bytes = encode(&bundle_packet).unwrap();
    
    sender_socket.send_to(&bundle_bytes, "127.0.0.1:19028").await.unwrap();
    
    // Receive and verify bundle
    match receive_osc_message(&listener_socket, Duration::from_secs(1)).await {
        Ok((packet, _)) => {
            if let OscPacket::Bundle(received_bundle) = packet {
                assert_eq!(received_bundle.content.len(), 2, "Bundle should contain 2 messages");
                
                // Verify first message
                if let OscPacket::Message(msg) = &received_bundle.content[0] {
                    assert_eq!(msg.addr, "/bundle/msg1");
                    assert!(!msg.args.is_empty(), "Bundle message 1 should have arguments");
                    if let OscType::Int(val) = msg.args[0] {
                        assert_eq!(val, 1);
                    }
                }
                
                // Verify second message
                if let OscPacket::Message(msg) = &received_bundle.content[1] {
                    assert_eq!(msg.addr, "/bundle/msg2");
                    assert!(!msg.args.is_empty(), "Bundle message 2 should have arguments");
                    if let OscType::Float(val) = msg.args[0] {
                        assert_eq!(val, 2.5);
                    }
                }
            } else {
                panic!("Expected OSC bundle, got message");
            }
        }
        Err(e) => panic!("Failed to receive bundle: {}", e),
    }
    
    let _ = shutdown_tx.send(true);
    let _ = timeout(Duration::from_secs(1), passthrough_handle).await;
}