use crate::config::Config;
use rosc::{OscPacket, decoder};
use std::{error::Error, sync::Arc};
use tokio::net::UdpSocket;
use tokio::sync::watch;

pub struct OscPassthrough {
    socket: Arc<UdpSocket>,
    config: Arc<Config>,
}

impl OscPassthrough {
    pub fn new(socket: Arc<UdpSocket>, config: Arc<Config>) -> Self {
        OscPassthrough { socket, config }
    }

    pub async fn start_passthrough(&self, mut shutdown_rx: watch::Receiver<bool>) -> Result<(), Box<dyn Error>> {
        if !self.config.osc.passthrough_enabled {
            return Ok(());
        }

        // Create a new socket for the passthrough port
        let passthrough_in_address = format!("{}:{}", self.config.osc.address, self.config.osc.passthrough_port);
        let passthrough_socket = match UdpSocket::bind(&passthrough_in_address).await {
            Ok(socket) => socket,
            Err(e) => {
                eprintln!("Failed to bind OSC passthrough socket to {}: {}", passthrough_in_address, e);
                eprintln!("Please ensure port {} is not in use by another application", self.config.osc.passthrough_port);
                return Err(Box::new(e));
            }
        };
        
        // Output address uses the regular output_port
        let output_address = format!("{}:{}", self.config.osc.address, self.config.osc.output_port);

        println!("OSC passthrough enabled: Forwarding messages from port {} to port {}", 
            self.config.osc.passthrough_port, self.config.osc.output_port);

        let mut buf = [0u8; 65536]; // OSC packet max size
        let mut invalid_packet_count = 0u64;
        
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        println!("OSC passthrough shutting down gracefully");
                        if invalid_packet_count > 0 {
                            println!("Total invalid packets dropped: {}", invalid_packet_count);
                        }
                        break;
                    }
                }
                result = passthrough_socket.recv_from(&mut buf) => {
                    match result {
                        Ok((size, src_addr)) => {
                            // Received a packet, validate it's OSC before forwarding
                            if size > 0 {
                                // Validate OSC packet
                                match decoder::decode_udp(&buf[..size]) {
                                    Ok((_, packet)) => {
                                        // Valid OSC packet
                                        if self.config.debug {
                                            match &packet {
                                                OscPacket::Message(msg) => {
                                                    println!("OSC passthrough: Forwarding message '{}' from {}", msg.addr, src_addr);
                                                },
                                                OscPacket::Bundle(bundle) => {
                                                    println!("OSC passthrough: Forwarding bundle with {} messages from {}", bundle.content.len(), src_addr);
                                                }
                                            }
                                        }

                                        // Forward the raw packet using the shared socket
                                        // This ensures OSC messages share the same socket as chatbox messages
                                        if let Err(e) = self.socket.send_to(&buf[..size], &output_address).await {
                                            eprintln!("Error forwarding OSC packet: {}", e);
                                        }
                                    }
                                    Err(e) => {
                                        invalid_packet_count += 1;
                                        if self.config.debug {
                                            eprintln!("Received invalid OSC packet from {}: {}", src_addr, e);
                                            eprintln!("Packet size: {} bytes, total invalid packets: {}", size, invalid_packet_count);
                                        }
                                        // Don't forward invalid packets
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Error receiving OSC packet: {}", e);
                            // If it's a critical error (socket closed), break the loop
                            if e.kind() == std::io::ErrorKind::ConnectionAborted || 
                               e.kind() == std::io::ErrorKind::UnexpectedEof {
                                eprintln!("Critical error in OSC passthrough, shutting down");
                                break;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}