use crate::config::Config;
use crate::types::AudioEvent;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavWriter;
use std::error::Error;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

struct NoiseGate {
    threshold: f32,
    hold_time: f32,
    last_active: std::time::Instant,
    is_active: bool,
}

impl NoiseGate {
    fn new(threshold: f32, hold_time: f32) -> Self {
        NoiseGate {
            threshold,
            hold_time,
            last_active: std::time::Instant::now(),
            is_active: false,
        }
    }

    fn process(&mut self, samples: &[f32]) -> bool {
        let max_amplitude = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        if max_amplitude > self.threshold {
            self.last_active = std::time::Instant::now();
            self.is_active = true;
        } else if self.is_active && self.last_active.elapsed().as_secs_f32() > self.hold_time {
            self.is_active = false;
        }

        self.is_active
    }
}

pub fn start_audio_recording(
    config: &Config,
    tx: mpsc::Sender<AudioEvent>,
) -> Result<(), Box<dyn Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    let device_config = device.default_input_config()?;

    let sample_rate = device_config.sample_rate().0 as f32;
    let channels = device_config.channels() as usize;
    let sample_format = device_config.sample_format();

    let err_fn = |err| eprintln!("An error occurred on the audio stream: {}", err);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let audio_data = Arc::new(Mutex::new(Vec::new()));
            let audio_data_clone = Arc::clone(&audio_data);

            let tx_clone = tx.clone();

            let mut noise_gate = NoiseGate::new(
                config.audio.noise_gate_threshold,
                config.audio.noise_gate_hold_time,
            );

            let mut is_recording = false;
            let mut silent_frames = 0;
            let silence_threshold = config.audio.silence_threshold;

            device.build_input_stream(
                &device_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if noise_gate.process(data) {
                        let mut buffer = audio_data_clone.lock().unwrap();

                        if !is_recording {
                            is_recording = true;
                            println!("Sound detected. Starting recording...");
                            let _ = tx_clone.try_send(AudioEvent::StartRecording);
                        }

                        buffer.extend_from_slice(data);
                        silent_frames = 0;
                    } else if is_recording {
                        silent_frames += 1;

                        if silent_frames >= silence_threshold {
                            is_recording = false;
                            silent_frames = 0;

                            let mut buffer = audio_data_clone.lock().unwrap();
                            if !buffer.is_empty() {
                                println!(
                                    "Silence detected. Stopping recording and processing audio..."
                                );
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
                                    )
                                    .unwrap();

                                    for &sample in buffer.iter() {
                                        writer.write_sample(sample).unwrap();
                                    }
                                    writer.finalize().unwrap();
                                }

                                let _ = tx_clone.try_send(AudioEvent::AudioData(wav_buffer));
                                buffer.clear();
                            }

                            let _ = tx_clone.try_send(AudioEvent::StopRecording);
                        } else {
                            // Keep recording during short pauses
                            let mut buffer = audio_data_clone.lock().unwrap();
                            buffer.extend_from_slice(data);
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        _ => return Err("Unsupported sample format".into()),
    };

    stream.play()?;

    // Keep the stream alive
    std::mem::forget(stream);

    Ok(())
}
