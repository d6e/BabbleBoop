use crate::app_state::{AppCommand, AppState};
use crate::config::{Config, CONFIG_PATH};
use eframe::egui;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub struct BabbleBoopApp {
    app_state: Arc<AppState>,
    config_draft: Config,
    status_message: Option<(String, std::time::Instant)>,
}

impl BabbleBoopApp {
    pub fn new(app_state: Arc<AppState>) -> Self {
        let config_draft = app_state.config.read().expect("Config lock poisoned").clone();
        Self {
            app_state,
            config_draft,
            status_message: None,
        }
    }

    fn show_status(&mut self, ctx: &egui::Context) {
        if let Some((msg, time)) = &self.status_message {
            if time.elapsed().as_secs() < 3 {
                egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
                    ui.label(msg.as_str());
                });
            } else {
                self.status_message = None;
            }
        }
    }

    fn set_status(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), std::time::Instant::now()));
    }

    fn send_command(&self, cmd: AppCommand) -> Result<(), String> {
        self.app_state
            .command_tx
            .blocking_send(cmd)
            .map_err(|e| format!("Failed to send command: {}", e))
    }
}

impl eframe::App for BabbleBoopApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Signal shutdown to all threads
        self.app_state.request_shutdown();
        // Send Quit command to processing loop
        if let Err(e) = self.send_command(AppCommand::Quit) {
            eprintln!("Warning: {}", e);
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.show_status(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("BabbleBoop");
            ui.add_space(10.0);

            // Enable/Disable toggle
            let enabled = self.app_state.enabled.load(Ordering::Relaxed);
            let toggle_text = if enabled { "Enabled" } else { "Disabled" };
            let toggle_color = if enabled {
                egui::Color32::from_rgb(100, 200, 100)
            } else {
                egui::Color32::from_rgb(200, 100, 100)
            };

            let mut toggle_error: Option<String> = None;
            ui.horizontal(|ui| {
                ui.label("Translation:");
                if ui
                    .add(egui::Button::new(toggle_text).fill(toggle_color))
                    .clicked()
                {
                    let new_state = !enabled;
                    self.app_state.enabled.store(new_state, Ordering::Relaxed);
                    if let Err(e) = self.send_command(AppCommand::SetEnabled(new_state)) {
                        toggle_error = Some(e);
                    }
                }
            });
            if let Some(e) = toggle_error {
                self.set_status(e);
            }

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                // OSC Settings
                ui.collapsing("OSC Settings", |ui| {
                    egui::Grid::new("osc_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Address:");
                            ui.text_edit_singleline(&mut self.config_draft.osc.address);
                            ui.end_row();

                            ui.label("Input Port:");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.input_port));
                            ui.end_row();

                            ui.label("Output Port:");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.output_port));
                            ui.end_row();

                            ui.label("Display Time (ms):");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.display_time));
                            ui.end_row();

                            ui.label("Max Message Chunks:");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.max_message_chunks));
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // OpenAI Settings
                ui.collapsing("OpenAI Settings", |ui| {
                    egui::Grid::new("openai_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("API Key:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.config_draft.openai.api_key)
                                    .password(true),
                            );
                            ui.end_row();

                            ui.label("Model:");
                            ui.text_edit_singleline(&mut self.config_draft.openai.model);
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Translation Settings
                ui.collapsing("Translation Settings", |ui| {
                    egui::Grid::new("translation_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Target Language:");
                            ui.text_edit_singleline(&mut self.config_draft.translation.target_language);
                            ui.end_row();

                            ui.label("Include Original:");
                            ui.checkbox(
                                &mut self.config_draft.translation.include_original_message,
                                "",
                            );
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Audio Settings
                ui.collapsing("Audio Settings (requires restart)", |ui| {
                    egui::Grid::new("audio_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Silence Threshold:");
                            ui.add(egui::DragValue::new(&mut self.config_draft.audio.silence_threshold));
                            ui.end_row();

                            ui.label("Noise Gate Threshold:");
                            ui.add(
                                egui::DragValue::new(&mut self.config_draft.audio.noise_gate_threshold)
                                    .speed(0.01)
                                    .range(0.0..=1.0),
                            );
                            ui.end_row();

                            ui.label("Noise Gate Hold Time (s):");
                            ui.add(
                                egui::DragValue::new(&mut self.config_draft.audio.noise_gate_hold_time)
                                    .speed(0.01)
                                    .range(0.0..=2.0),
                            );
                            ui.end_row();

                            ui.label("Min Transcription Duration (s):");
                            ui.add(
                                egui::DragValue::new(&mut self.config_draft.audio.min_transcription_duration)
                                    .speed(0.1)
                                    .range(0.0..=10.0),
                            );
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Rate Limit Settings
                ui.collapsing("Rate Limit", |ui| {
                    egui::Grid::new("rate_limit_grid")
                        .num_columns(2)
                        .spacing([10.0, 4.0])
                        .show(ui, |ui| {
                            ui.label("Requests per Minute:");
                            ui.add(egui::DragValue::new(
                                &mut self.config_draft.rate_limit.requests_per_minute,
                            ));
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Keep Audio Files
                ui.horizontal(|ui| {
                    ui.label("Keep Audio Files:");
                    ui.checkbox(&mut self.config_draft.keep_audio_files, "");
                });

                ui.add_space(20.0);

                // Save button
                if ui.button("Save Settings").clicked() {
                    match self.config_draft.save(CONFIG_PATH) {
                        Ok(()) => {
                            // Update the shared config
                            if let Ok(mut config) = self.app_state.config.write() {
                                *config = self.config_draft.clone();
                            }
                            match self.send_command(AppCommand::UpdateConfig(self.config_draft.clone())) {
                                Ok(()) => self.set_status("Settings saved successfully"),
                                Err(e) => self.set_status(format!("Settings saved to file, but {}", e)),
                            }
                        }
                        Err(e) => {
                            self.set_status(format!("Failed to save: {}", e));
                        }
                    }
                }
            });
        });

        // Request repaint for status message timeout
        if self.status_message.is_some() {
            ctx.request_repaint();
        }
    }
}

pub fn run_gui(app_state: Arc<AppState>) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 500.0])
            .with_min_inner_size([350.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "BabbleBoop",
        options,
        Box::new(|_cc| Ok(Box::new(BabbleBoopApp::new(app_state)))),
    )
}
