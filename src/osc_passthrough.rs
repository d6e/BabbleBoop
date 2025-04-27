use crate::config::Config;
use rosc::{OscPacket, decoder};
use std::{error::Error, sync::Arc};
use tokio::net::UdpSocket;

pub struct OscPassthrough {
    socket: Arc<UdpSocket>,
    config: Arc<Config>,
}

impl OscPassthrough {
    pub fn new(socket: Arc<UdpSocket>, config: Arc<Config>) -> Self {
        OscPassthrough { socket, config }
    }

    pub async fn start_passthrough(&self) -> Result<(), Box<dyn Error>> {
        if !self.config.osc.passthrough_enabled {
            return Ok(());
        }

        // Create a new socket for the passthrough port
        let passthrough_in_address = format!("{}:{}", self.config.osc.address, self.config.osc.passthrough_port);
        let passthrough_socket = UdpSocket::bind(&passthrough_in_address).await?;
        
        // Output address uses the regular output_port
        let output_address = format!("{}:{}", self.config.osc.address, self.config.osc.output_port);

        println!("OSC passthrough enabled: Forwarding messages from port {} to port {}", 
            self.config.osc.passthrough_port, self.config.osc.output_port);

        let mut buf = [0u8; 65536]; // OSC packet max size
        
        loop {
            match passthrough_socket.recv_from(&mut buf).await {
                Ok((size, _src_addr)) => {
                    // Received an OSC packet, forward to the output port
                    if size > 0 {
                        // Debug log if debug mode enabled
                        if self.config.debug {
                            if let Ok((_, packet)) = decoder::decode_udp(&buf[..size]) {
                                match packet {
                                    OscPacket::Message(msg) => {
                                        println!("OSC passthrough: Forwarding message: {}", msg.addr);
                                    },
                                    OscPacket::Bundle(bundle) => {
                                        println!("OSC passthrough: Forwarding bundle with {} messages", bundle.content.len());
                                    }
                                }
                            }
                        }

                        // Forward the raw packet using the shared socket
                        // This ensures OSC messages share the same socket as chatbox messages
                        if let Err(e) = self.socket.send_to(&buf[..size], &output_address).await {
                            eprintln!("Error forwarding OSC packet: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error receiving OSC packet: {}", e);
                }
            }
        }
    }
}