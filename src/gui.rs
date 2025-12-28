use crate::app_state::{AppCommand, AppState, LogEntry};
use crate::config::{Config, CONFIG_PATH};
use eframe::egui;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Draw an audio level meter with threshold indicator
fn draw_audio_level_meter(ui: &mut egui::Ui, current_level: f32, threshold: f32) {
    let meter_size = egui::vec2(ui.available_width().min(200.0), 16.0);
    let (rect, _response) = ui.allocate_exact_size(meter_size, egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, egui::Color32::from_gray(40));

        // Level bar with color gradient based on level
        let level_width = rect.width() * current_level.min(1.0);
        if level_width > 0.0 {
            let level_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(level_width, rect.height()));
            let color = if current_level > 0.8 {
                egui::Color32::from_rgb(220, 60, 60) // Red for high levels
            } else if current_level > 0.5 {
                egui::Color32::from_rgb(220, 180, 60) // Yellow for medium
            } else {
                egui::Color32::from_rgb(60, 180, 60) // Green for low
            };
            painter.rect_filled(level_rect, 2.0, color);
        }

        // Threshold indicator line
        let threshold_x = rect.left() + rect.width() * threshold;
        painter.vline(
            threshold_x,
            rect.y_range(),
            egui::Stroke::new(2.0, egui::Color32::WHITE),
        );
    }
}

const MAX_LOG_ENTRIES: usize = 50;

#[derive(Clone, Copy, PartialEq)]
enum StatusType {
    Success,
    Error,
    Info,
}

const OPENAI_MODELS: &[&str] = &[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
];

const TRANSCRIPTION_MODELS: &[&str] = &["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"];

/// Custom toggle switch widget
fn toggle_switch(on: &mut bool) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| {
        let desired_size = egui::vec2(36.0, 20.0);
        let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        if response.clicked() {
            *on = !*on;
            response.mark_changed();
        }

        if ui.is_rect_visible(rect) {
            let how_on = ui.ctx().animate_bool_responsive(response.id, *on);
            let visuals = ui.style().interact_selectable(&response, *on);

            let rect = rect.expand(visuals.expansion);
            let radius = 0.5 * rect.height();

            // Track background
            let bg_color = egui::Color32::from_rgb(
                (60.0 + how_on * 40.0) as u8,
                (60.0 + how_on * 100.0) as u8,
                (60.0 + how_on * 40.0) as u8,
            );
            ui.painter().rect(rect, radius, bg_color, visuals.bg_stroke);

            // Knob
            let knob_radius = radius - 2.0;
            let knob_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
            let knob_center = egui::pos2(knob_x, rect.center().y);
            ui.painter().circle(
                knob_center,
                knob_radius,
                egui::Color32::WHITE,
                egui::Stroke::NONE,
            );
        }

        response
    }
}

pub struct BabbleBoopApp {
    app_state: Arc<AppState>,
    config_draft: Config,
    saved_config: Config,
    status_message: Option<(String, StatusType, std::time::Instant)>,
    log_rx: mpsc::Receiver<LogEntry>,
    log_entries: Vec<LogEntry>,
    start_time: std::time::Instant,
}

impl BabbleBoopApp {
    pub fn new(app_state: Arc<AppState>, log_rx: mpsc::Receiver<LogEntry>) -> Self {
        let config_draft = app_state
            .config
            .read()
            .expect("Config lock poisoned")
            .clone();
        let saved_config = config_draft.clone();
        Self {
            app_state,
            config_draft,
            saved_config,
            status_message: None,
            log_rx,
            log_entries: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }

    fn poll_log_entries(&mut self) {
        while let Ok(entry) = self.log_rx.try_recv() {
            self.log_entries.push(entry);
            if self.log_entries.len() > MAX_LOG_ENTRIES {
                self.log_entries.remove(0);
            }
        }
    }

    fn format_timestamp(&self, entry: &LogEntry) -> String {
        let elapsed = entry.timestamp.duration_since(self.start_time);
        let total_secs = elapsed.as_secs();
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        format!("{:02}:{:02}:{:02}", hours, mins, secs)
    }

    fn has_unsaved_changes(&self) -> bool {
        self.config_draft != self.saved_config
    }

    fn reload_config(&mut self) {
        self.config_draft = self.saved_config.clone();
        self.set_status_info("Changes discarded");
    }

    fn validate_config(&self) -> Result<(), String> {
        // Validate ports
        if self.config_draft.osc.input_port == 0 {
            return Err("Input port cannot be 0".to_string());
        }
        if self.config_draft.osc.output_port == 0 {
            return Err("Output port cannot be 0".to_string());
        }
        if self.config_draft.osc.input_port == self.config_draft.osc.output_port {
            return Err("Input and output ports cannot be the same".to_string());
        }

        // Validate API key
        if self.config_draft.openai.api_key.trim().is_empty() {
            return Err("OpenAI API key is required".to_string());
        }

        // Validate model
        if self.config_draft.openai.model.trim().is_empty() {
            return Err("OpenAI model is required".to_string());
        }

        // Validate target language
        if self
            .config_draft
            .translation
            .target_language
            .trim()
            .is_empty()
        {
            return Err("Target language is required".to_string());
        }

        Ok(())
    }

    fn show_status(&mut self, ctx: &egui::Context) {
        if let Some((msg, status_type, time)) = &self.status_message {
            if time.elapsed().as_secs() < 3 {
                egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
                    let color = match status_type {
                        StatusType::Success => egui::Color32::from_rgb(100, 200, 100),
                        StatusType::Error => egui::Color32::from_rgb(220, 80, 80),
                        StatusType::Info => egui::Color32::from_rgb(150, 150, 220),
                    };
                    ui.colored_label(color, msg.as_str());
                });
            } else {
                self.status_message = None;
            }
        }
    }

    fn set_status_success(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), StatusType::Success, std::time::Instant::now()));
    }

    fn set_status_error(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), StatusType::Error, std::time::Instant::now()));
    }

    fn set_status_info(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), StatusType::Info, std::time::Instant::now()));
    }

    fn send_command(&self, cmd: AppCommand) -> Result<(), String> {
        self.app_state
            .command_tx
            .blocking_send(cmd)
            .map_err(|e| format!("Failed to send command: {}", e))
    }

    fn show_button_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("button_panel").show(ctx, |ui| {
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                // Unsaved changes indicator
                if self.has_unsaved_changes() {
                    ui.label(
                        egui::RichText::new("● Unsaved changes")
                            .color(egui::Color32::from_rgb(255, 180, 0)),
                    );
                    ui.separator();
                }

                // Reset button (only show if there are changes)
                if self.has_unsaved_changes()
                    && ui
                        .button("Reset")
                        .on_hover_text("Discard changes and reload saved settings")
                        .clicked()
                {
                    self.reload_config();
                }

                // Save button
                let save_button = ui.add_enabled(
                    self.has_unsaved_changes(),
                    egui::Button::new("Save Settings"),
                );

                if save_button.clicked() {
                    if let Err(e) = self.validate_config() {
                        self.set_status_error(e);
                    } else {
                        match self.config_draft.save(CONFIG_PATH) {
                            Ok(()) => {
                                // Update the shared config
                                if let Ok(mut config) = self.app_state.config.write() {
                                    *config = self.config_draft.clone();
                                }
                                self.saved_config = self.config_draft.clone();
                                match self.send_command(AppCommand::UpdateConfig(
                                    self.config_draft.clone(),
                                )) {
                                    Ok(()) => {
                                        self.set_status_success("Settings saved successfully")
                                    }
                                    Err(e) => self.set_status_error(format!(
                                        "Settings saved to file, but {}",
                                        e
                                    )),
                                }
                            }
                            Err(e) => {
                                self.set_status_error(format!("Failed to save: {}", e));
                            }
                        }
                    }
                }
            });

            ui.add_space(4.0);
        });
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
        // Poll for new log entries
        self.poll_log_entries();

        // Render bottom panels FIRST so CentralPanel knows remaining space
        self.show_status(ctx);
        self.show_button_panel(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("BabbleBoop");
            ui.add_space(10.0);

            // Enable/Disable toggle
            let mut enabled = self.app_state.enabled.load(Ordering::Relaxed);

            let mut toggle_error: Option<String> = None;
            ui.horizontal(|ui| {
                ui.label("Translation:");
                ui.add_space(8.0);

                // Toggle switch
                let response = ui.add(toggle_switch(&mut enabled));
                if response.changed() {
                    self.app_state.enabled.store(enabled, Ordering::Relaxed);
                    if let Err(e) = self.send_command(AppCommand::SetEnabled(enabled)) {
                        toggle_error = Some(e);
                    }
                }

                // Status text
                let (status_text, status_color) = if enabled {
                    ("Enabled", egui::Color32::from_rgb(80, 160, 80))
                } else {
                    ("Disabled", egui::Color32::from_rgb(140, 140, 140))
                };
                ui.label(egui::RichText::new(status_text).color(status_color));
            });
            if let Some(e) = toggle_error {
                self.set_status_error(e);
            }

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(10.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                let grid_spacing = [10.0, 6.0];
                let label_width = 160.0;

                // OSC Settings
                egui::CollapsingHeader::new("OSC Settings")
                    .default_open(true)
                    .show(ui, |ui| {
                    egui::Grid::new("osc_grid")
                        .num_columns(2)
                        .spacing(grid_spacing)
                        .min_col_width(label_width)
                        .show(ui, |ui| {
                            ui.label("Address:")
                                .on_hover_text("OSC server address (usually 127.0.0.1 for local)");
                            ui.text_edit_singleline(&mut self.config_draft.osc.address);
                            ui.end_row();

                            ui.label("Input Port:")
                                .on_hover_text("Port to receive OSC messages from VRChat");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.input_port).range(1..=65535));
                            ui.end_row();

                            ui.label("Output Port:")
                                .on_hover_text("Port to send OSC messages to VRChat");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.output_port).range(1..=65535));
                            ui.end_row();

                            ui.label("Display Time (ms):")
                                .on_hover_text("How long messages stay visible in VRChat");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.display_time).range(1000..=30000));
                            ui.end_row();

                            ui.label("Max Message Chunks:")
                                .on_hover_text("Maximum number of message parts for long text");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.max_message_chunks).range(1..=10));
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // OpenAI Settings
                egui::CollapsingHeader::new("OpenAI Settings")
                    .default_open(true)
                    .show(ui, |ui| {
                    egui::Grid::new("openai_grid")
                        .num_columns(2)
                        .spacing(grid_spacing)
                        .min_col_width(label_width)
                        .show(ui, |ui| {
                            ui.label("API Key:")
                                .on_hover_text("Your OpenAI API key for transcription and translation");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.config_draft.openai.api_key)
                                    .password(true),
                            );
                            ui.end_row();

                            ui.label("Model:")
                                .on_hover_text("OpenAI model for translation (gpt-4o recommended)");
                            egui::ComboBox::from_id_salt("model_combo")
                                .selected_text(&self.config_draft.openai.model)
                                .show_ui(ui, |ui| {
                                    for model in OPENAI_MODELS {
                                        ui.selectable_value(
                                            &mut self.config_draft.openai.model,
                                            model.to_string(),
                                            *model,
                                        );
                                    }
                                });
                            ui.end_row();

                            ui.label("Transcription:")
                                .on_hover_text("Model for speech-to-text (gpt-4o-mini-transcribe is cheapest)");
                            egui::ComboBox::from_id_salt("transcription_model_combo")
                                .selected_text(&self.config_draft.openai.transcription_model)
                                .show_ui(ui, |ui| {
                                    for model in TRANSCRIPTION_MODELS {
                                        ui.selectable_value(
                                            &mut self.config_draft.openai.transcription_model,
                                            model.to_string(),
                                            *model,
                                        );
                                    }
                                });
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Translation Settings
                egui::CollapsingHeader::new("Translation Settings")
                    .default_open(true)
                    .show(ui, |ui| {
                    egui::Grid::new("translation_grid")
                        .num_columns(2)
                        .spacing(grid_spacing)
                        .min_col_width(label_width)
                        .show(ui, |ui| {
                            ui.label("Target Language:")
                                .on_hover_text("Language to translate your speech into (e.g., Japanese, Spanish)");
                            ui.text_edit_singleline(&mut self.config_draft.translation.target_language);
                            ui.end_row();

                            ui.label("Include Original:")
                                .on_hover_text("Show original text alongside the translation");
                            ui.checkbox(
                                &mut self.config_draft.translation.include_original_message,
                                "",
                            );
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Audio Settings
                let audio_header = egui::CollapsingHeader::new("Audio Settings")
                    .show(ui, |ui| {
                    // Audio level meter at the top
                    ui.label("Input Level:");
                    let level_bits = self.app_state.current_audio_level.load(Ordering::Relaxed);
                    let current_level = f32::from_bits(level_bits);
                    draw_audio_level_meter(ui, current_level, self.config_draft.audio.noise_gate_threshold);
                    ui.add_space(4.0);

                    // Test Microphone button
                    let is_testing = self.app_state.test_mode_active.load(Ordering::Relaxed);
                    ui.horizontal(|ui| {
                        if is_testing {
                            if ui.button("Stop Recording").on_hover_text("Stop recording and play back").clicked() {
                                if let Err(e) = self.send_command(AppCommand::StopTestRecording) {
                                    self.set_status_error(format!("Failed to stop test: {}", e));
                                }
                            }
                        } else if ui.button("Test Microphone").on_hover_text("Record audio and play it back (click again to stop)").clicked() {
                            if let Err(e) = self.send_command(AppCommand::StartTestRecording) {
                                self.set_status_error(format!("Failed to start test: {}", e));
                            }
                        }
                    });
                    ui.add_space(8.0);

                    egui::Grid::new("audio_grid")
                        .num_columns(2)
                        .spacing(grid_spacing)
                        .min_col_width(label_width)
                        .show(ui, |ui| {
                            ui.label("Silence Threshold:")
                                .on_hover_text("Number of consecutive silent samples before stopping recording");
                            ui.add(egui::DragValue::new(&mut self.config_draft.audio.silence_threshold).range(1..=200));
                            ui.end_row();

                            ui.label("Noise Gate Threshold:")
                                .on_hover_text("Audio level below which input is considered silence (0.0-1.0). White line on meter shows threshold.");
                            ui.add(
                                egui::DragValue::new(&mut self.config_draft.audio.noise_gate_threshold)
                                    .speed(0.01)
                                    .range(0.0..=1.0),
                            );
                            ui.end_row();

                            ui.label("Noise Gate Hold (s):")
                                .on_hover_text("Time to keep gate open after audio drops below threshold");
                            ui.add(
                                egui::DragValue::new(&mut self.config_draft.audio.noise_gate_hold_time)
                                    .speed(0.01)
                                    .range(0.0..=2.0),
                            );
                            ui.end_row();

                            ui.label("Min Duration (s):")
                                .on_hover_text("Minimum recording length before transcription (filters out noise)");
                            ui.add(
                                egui::DragValue::new(&mut self.config_draft.audio.min_transcription_duration)
                                    .speed(0.1)
                                    .range(0.0..=10.0),
                            );
                            ui.end_row();
                        });
                });
                // Request repaint when audio settings is open to update the level meter
                if audio_header.body_returned.is_some() {
                    ctx.request_repaint();
                }

                ui.add_space(5.0);

                // Rate Limit Settings
                egui::CollapsingHeader::new("Rate Limit")
                    .show(ui, |ui| {
                    egui::Grid::new("rate_limit_grid")
                        .num_columns(2)
                        .spacing(grid_spacing)
                        .min_col_width(label_width)
                        .show(ui, |ui| {
                            ui.label("Requests per Minute:")
                                .on_hover_text("Maximum API requests per minute to avoid rate limiting");
                            ui.add(egui::DragValue::new(
                                &mut self.config_draft.rate_limit.requests_per_minute,
                            ).range(1..=120));
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Debug/Development Settings
                egui::CollapsingHeader::new("Debug")
                    .show(ui, |ui| {
                    egui::Grid::new("debug_grid")
                        .num_columns(2)
                        .spacing(grid_spacing)
                        .min_col_width(label_width)
                        .show(ui, |ui| {
                            ui.label("Keep Audio Files:")
                                .on_hover_text("Save recorded audio files for debugging");
                            ui.checkbox(&mut self.config_draft.keep_audio_files, "");
                            ui.end_row();

                            ui.add_enabled_ui(self.config_draft.keep_audio_files, |ui| {
                                ui.label("Max Audio Files:")
                                    .on_hover_text("Maximum number of audio files to keep");
                            });
                            ui.add_enabled_ui(self.config_draft.keep_audio_files, |ui| {
                                ui.add(egui::DragValue::new(&mut self.config_draft.max_audio_files).range(1..=100));
                            });
                            ui.end_row();
                        });
                });

                ui.add_space(5.0);

                // Activity Log
                egui::CollapsingHeader::new("Activity Log")
                    .default_open(true)
                    .show(ui, |ui| {
                        if self.log_entries.is_empty() {
                            ui.label(egui::RichText::new("No activity yet...").italics().color(egui::Color32::GRAY));
                        } else {
                            egui::ScrollArea::vertical()
                                .id_salt("activity_log_scroll")
                                .max_height(150.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    for entry in &self.log_entries {
                                        let timestamp = self.format_timestamp(entry);
                                        ui.horizontal_wrapped(|ui| {
                                            ui.label(
                                                egui::RichText::new(format!("[{}]", timestamp))
                                                    .color(egui::Color32::from_rgb(120, 120, 120))
                                                    .small()
                                            );
                                            ui.label(
                                                egui::RichText::new(&entry.original)
                                                    .color(egui::Color32::from_rgb(180, 180, 220))
                                            );
                                            ui.label(
                                                egui::RichText::new("->")
                                                    .color(egui::Color32::from_rgb(100, 100, 100))
                                            );
                                            ui.label(
                                                egui::RichText::new(&entry.translated)
                                                    .color(egui::Color32::from_rgb(120, 200, 120))
                                            );
                                        });
                                    }
                                });
                        }
                    });
            });
        });

        // Request repaint for status message timeout and log polling
        if self.status_message.is_some() || !self.log_entries.is_empty() {
            ctx.request_repaint();
        }
    }
}

pub fn run_gui(
    app_state: Arc<AppState>,
    log_rx: mpsc::Receiver<LogEntry>,
    first_run: bool,
) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 600.0])
            .with_min_inner_size([350.0, 500.0]),
        ..Default::default()
    };

    eframe::run_native(
        "BabbleBoop",
        options,
        Box::new(move |_cc| {
            let mut app = BabbleBoopApp::new(app_state, log_rx);
            if first_run {
                app.set_status_info("Welcome! Please set your OpenAI API key to get started.");
            }
            Ok(Box::new(app))
        }),
    )
}

/// Shows a simple error dialog using egui. Used for startup errors.
pub fn run_error_dialog(title: &str, message: &str) -> Result<(), eframe::Error> {
    let title = title.to_string();
    let message = message.to_string();
    let window_title = title.clone();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 200.0])
            .with_min_inner_size([300.0, 150.0]),
        ..Default::default()
    };

    eframe::run_native(
        &window_title,
        options,
        Box::new(move |_cc| Ok(Box::new(ErrorDialog::new(title, message)))),
    )
}

struct ErrorDialog {
    title: String,
    message: String,
}

impl ErrorDialog {
    fn new(title: String, message: String) -> Self {
        Self { title, message }
    }
}

impl eframe::App for ErrorDialog {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.heading(
                    egui::RichText::new(&self.title).color(egui::Color32::from_rgb(220, 80, 80)),
                );
                ui.add_space(15.0);
                ui.label(&self.message);
                ui.add_space(20.0);
                if ui.button("OK").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
        });
    }
}
