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

        let mut buf = [0u8; 65536]; // OSC packet max size
        let passthrough_address = format!("{}:{}", self.config.osc.address, self.config.osc.passthrough_port);

        println!("OSC passthrough enabled: Forwarding messages from port {} to port {}", 
            self.config.osc.input_port, self.config.osc.passthrough_port);

        loop {
            match self.socket.recv_from(&mut buf).await {
                Ok((size, _src_addr)) => {
                    // Received an OSC packet, forward to the passthrough port
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

                        // Forward the raw packet, no need to decode/encode
                        if let Err(e) = self.socket.send_to(&buf[..size], &passthrough_address).await {
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