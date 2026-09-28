use crate::app_state::{AppCommand, AppState, LogEntry, LogLevel};
use crate::config::{Config, ThemeMode, CONFIG_PATH};
use crate::models;
use crate::processing_loop::TEST_RECORDING_LIMIT;
use crate::theme::{self, AppColors};
use eframe::egui;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// Draw an audio level meter with draggable threshold indicator
fn draw_audio_level_meter(
    ui: &mut egui::Ui,
    current_level: f32,
    threshold: &mut f32,
    colors: &AppColors,
) {
    let meter_size = egui::vec2(ui.available_width().min(200.0), 16.0);
    let (rect, response) = ui.allocate_exact_size(meter_size, egui::Sense::click_and_drag());

    // Handle dragging to adjust threshold
    if response.dragged() {
        if let Some(pos) = response.interact_pointer_pos() {
            let new_threshold = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            *threshold = new_threshold;
        }
    }

    // Change cursor to indicate draggability
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
    }

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, colors.meter_background);

        // Level bar with color gradient based on level
        let level_width = rect.width() * current_level.min(1.0);
        if level_width > 0.0 {
            let level_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(level_width, rect.height()));
            let color = if current_level > 0.8 {
                colors.meter_high
            } else if current_level > 0.5 {
                colors.meter_medium
            } else {
                colors.meter_low
            };
            painter.rect_filled(level_rect, 2.0, color);
        }

        // Threshold indicator line (highlight when hovered/dragged)
        let threshold_x = rect.left() + rect.width() * *threshold;
        let line_color = if response.hovered() || response.dragged() {
            colors.meter_threshold_active
        } else {
            colors.meter_threshold
        };
        painter.vline(
            threshold_x,
            rect.y_range(),
            egui::Stroke::new(2.0f32, line_color),
        );
    }
}

/// Draw noise gate state indicator with hold time countdown
fn draw_noise_gate_state(
    ui: &mut egui::Ui,
    is_active: bool,
    hold_remaining: f32,
    hold_time: f32,
    colors: &AppColors,
) {
    let meter_size = egui::vec2(ui.available_width().min(200.0), 10.0);
    let (rect, _response) = ui.allocate_exact_size(meter_size, egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, colors.meter_background);

        if is_active {
            let color = if hold_remaining > 0.0 {
                // In hold state: amber
                colors.meter_medium
            } else {
                // Active audio: green
                colors.meter_low
            };

            // Fill amount based on hold state
            let fill_ratio = if hold_remaining > 0.0 && hold_time > 0.0 {
                hold_remaining / hold_time
            } else {
                1.0
            };

            let fill_width = rect.width() * fill_ratio;
            if fill_width > 0.0 {
                let fill_rect =
                    egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, rect.height()));
                painter.rect_filled(fill_rect, 2.0, color);
            }
        }
    }
}

/// Draw a progress bar showing silent frames toward silence threshold
fn draw_silence_counter(ui: &mut egui::Ui, silent_frames: u32, threshold: u32, colors: &AppColors) {
    let meter_size = egui::vec2(ui.available_width().min(200.0), 10.0);
    let (rect, _response) = ui.allocate_exact_size(meter_size, egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, colors.meter_background);

        // Progress bar
        let progress = if threshold > 0 {
            (silent_frames as f32 / threshold as f32).min(1.0)
        } else {
            0.0
        };

        let fill_width = rect.width() * progress;
        if fill_width > 0.0 {
            let fill_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, rect.height()));
            // Yellow to red gradient as silence progresses
            let color = colors
                .meter_medium
                .lerp_to_gamma(colors.meter_high, progress);
            painter.rect_filled(fill_rect, 2.0, color);
        }
    }
}

/// Draw recording duration progress toward minimum transcription duration
fn draw_recording_duration(
    ui: &mut egui::Ui,
    duration: f32,
    min_duration: f32,
    colors: &AppColors,
) {
    let meter_size = egui::vec2(ui.available_width().min(200.0), 10.0);
    let (rect, _response) = ui.allocate_exact_size(meter_size, egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, colors.meter_background);

        // Progress bar
        let progress = if min_duration > 0.0 {
            (duration / min_duration).min(1.0)
        } else {
            1.0
        };

        let fill_width = rect.width() * progress;
        if fill_width > 0.0 {
            let fill_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(fill_width, rect.height()));
            // Red to green as duration increases
            let color = if progress >= 1.0 {
                colors.meter_low
            } else {
                colors.meter_high.lerp_to_gamma(colors.meter_low, progress)
            };
            painter.rect_filled(fill_rect, 2.0, color);
        }
    }
}

const MAX_LOG_ENTRIES: usize = 50;

/// How long a status message stays in the status bar.
const STATUS_MESSAGE_TIME: Duration = Duration::from_secs(3);

/// Time between frames while the audio meters are visible, about 30 per
/// second.
const METER_REFRESH_INTERVAL: Duration = Duration::from_millis(33);

/// Spacing and label column width of the settings grids.
const GRID_SPACING: [f32; 2] = [10.0, 6.0];
const LABEL_WIDTH: f32 = 160.0;

#[derive(Clone, Copy, PartialEq)]
enum StatusType {
    Success,
    Error,
    Info,
}

/// Text field for a model name, with a list of suggested models next to it.
/// The config takes any model name, for example a new or fine-tuned model.
fn model_name_edit(ui: &mut egui::Ui, id_salt: &str, model: &mut String, presets: &[&str]) {
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(model).desired_width(150.0));
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text("")
            .width(0.0)
            .show_ui(ui, |ui| {
                for preset in presets {
                    ui.selectable_value(model, preset.to_string(), *preset);
                }
            });
    });
}

/// Custom toggle switch widget
fn toggle_switch<'a>(on: &'a mut bool, colors: &'a AppColors) -> impl egui::Widget + 'a {
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
            let bg_color = colors.toggle_off.lerp_to_gamma(colors.toggle_on, how_on);
            ui.painter().rect(rect, radius, bg_color, visuals.bg_stroke);

            // Knob
            let knob_radius = radius - 2.0;
            let knob_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
            let knob_center = egui::pos2(knob_x, rect.center().y);
            ui.painter().circle(
                knob_center,
                knob_radius,
                colors.toggle_knob,
                egui::Stroke::NONE,
            );
        }

        response
    }
}

pub struct BabbleBoopApp {
    app_state: Arc<AppState>,
    pub(crate) config_draft: Config,
    saved_config: Config,
    status_message: Option<(String, StatusType, std::time::Instant)>,
    log_rx: mpsc::Receiver<LogEntry>,
    log_entries: Vec<LogEntry>,
    start_time: std::time::Instant,
    colors: AppColors,
}

impl BabbleBoopApp {
    pub fn new(app_state: Arc<AppState>, log_rx: mpsc::Receiver<LogEntry>) -> Self {
        let config_draft = app_state
            .config
            .read()
            .expect("Config lock poisoned")
            .clone();
        let saved_config = config_draft.clone();
        let colors = theme::get_colors(config_draft.theme);
        Self {
            app_state,
            config_draft,
            saved_config,
            status_message: None,
            log_rx,
            log_entries: Vec::new(),
            start_time: std::time::Instant::now(),
            colors,
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

    fn reload_config(&mut self, ctx: &egui::Context) {
        self.config_draft = self.saved_config.clone();
        self.colors = theme::get_colors(self.config_draft.theme);
        ctx.set_visuals(theme::get_visuals(self.config_draft.theme));
        self.set_status_info("Changes discarded");
    }

    /// The draft to save, or why it cannot be saved. Removes spaces and
    /// line breaks around the text fields first. The API does not know a
    /// model name with a space at the end. A pasted API key often has a
    /// line break at the end, and a request cannot put a line break in its
    /// Authorization header (http 1.4.0, src/header/value.rs:557).
    pub(crate) fn config_to_save(&mut self) -> Result<Config, String> {
        let draft = &mut self.config_draft;
        for text in [
            &mut draft.openai.api_key,
            &mut draft.openai.model,
            &mut draft.openai.transcription_model,
            &mut draft.translation.target_language,
        ] {
            *text = text.trim().to_string();
        }
        self.validate_config()?;
        Ok(self.config_draft.clone())
    }

    /// Write `new_config` to the config file and send it to the processing
    /// loop.
    fn save_config(&mut self, new_config: Config) {
        if let Err(e) = new_config.save(CONFIG_PATH) {
            self.set_status_error(format!("Failed to save: {}", e));
            return;
        }
        // Update the shared config
        if let Ok(mut config) = self.app_state.config.write() {
            *config = new_config.clone();
        }
        self.saved_config = new_config.clone();
        match self.send_command(AppCommand::UpdateConfig(new_config)) {
            Ok(()) => self.set_status_success("Settings saved successfully"),
            Err(e) => self.set_status_error(format!("Settings saved to file, but {}", e)),
        }
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
        if self
            .config_draft
            .openai
            .transcription_model
            .trim()
            .is_empty()
        {
            return Err("Transcription model is required".to_string());
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
            if time.elapsed() < STATUS_MESSAGE_TIME {
                let colors = self.colors;
                egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
                    let color = match status_type {
                        StatusType::Success => colors.success,
                        StatusType::Error => colors.error,
                        StatusType::Info => colors.info,
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

    pub(crate) fn set_status_info(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), StatusType::Info, std::time::Instant::now()));
    }

    fn send_command(&self, cmd: AppCommand) -> Result<(), String> {
        self.app_state
            .command_tx
            .blocking_send(cmd)
            .map_err(|e| format!("Failed to send command: {}", e))
    }

    fn show_button_panel(&mut self, ctx: &egui::Context) {
        let colors = self.colors;
        let mut should_reload = false;
        egui::TopBottomPanel::bottom("button_panel").show(ctx, |ui| {
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                // Unsaved changes indicator
                if self.has_unsaved_changes() {
                    ui.label(
                        egui::RichText::new("● Unsaved changes").color(colors.unsaved_indicator),
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
                    should_reload = true;
                }

                // Save button
                let save_button = ui.add_enabled(
                    self.has_unsaved_changes(),
                    egui::Button::new("Save Settings"),
                );

                if save_button.clicked() {
                    match self.config_to_save() {
                        Ok(new_config) => self.save_config(new_config),
                        Err(e) => self.set_status_error(e),
                    }
                }
            });

            ui.add_space(4.0);
        });
        if should_reload {
            self.reload_config(ctx);
        }
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
        self.ui(ctx);
    }
}

impl BabbleBoopApp {
    /// Show one frame. Separate from `update` so tests can run it without
    /// an `eframe::Frame`.
    pub(crate) fn ui(&mut self, ctx: &egui::Context) {
        // Poll for new log entries
        self.poll_log_entries();

        // Render bottom panels FIRST so CentralPanel knows remaining space
        self.show_status(ctx);
        self.show_button_panel(ctx);

        let colors = self.colors;
        let mut theme_changed = false;
        let mut new_theme = self.config_draft.theme;

        egui::CentralPanel::default().show(ctx, |ui| {
            // Header with theme toggle
            ui.horizontal(|ui| {
                ui.heading("BabbleBoop");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let is_dark = self.config_draft.theme == ThemeMode::Dark;
                    let label = if is_dark { "Light" } else { "Dark" };
                    if ui.button(label).on_hover_text("Switch theme").clicked() {
                        new_theme = if is_dark { ThemeMode::Light } else { ThemeMode::Dark };
                        theme_changed = true;
                    }
                });
            });
            ui.add_space(10.0);

            // Enable/Disable toggle
            let mut enabled = self.app_state.enabled.load(Ordering::Relaxed);
            let total_cost = self.app_state.get_total_cost();

            let mut toggle_error: Option<String> = None;
            ui.horizontal(|ui| {
                ui.label("Translation:");
                ui.add_space(8.0);

                // Toggle switch
                let response = ui.add(toggle_switch(&mut enabled, &colors));
                if response.changed() {
                    self.app_state.enabled.store(enabled, Ordering::Relaxed);
                    if let Err(e) = self.send_command(AppCommand::SetEnabled(enabled)) {
                        toggle_error = Some(e);
                    }
                }

                // Status text
                if self.app_state.is_processing_stopped() {
                    ui.label(egui::RichText::new("Processing stopped").color(colors.error))
                        .on_hover_text("Translation does not work until you restart BabbleBoop. The activity log shows the cause.");
                } else {
                    let (status_text, status_color) = if enabled {
                        ("Enabled", colors.enabled_text)
                    } else {
                        ("Disabled", colors.disabled_text)
                    };
                    ui.label(egui::RichText::new(status_text).color(status_color));
                }

                // Cost display on the right
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!("${:.4}", total_cost))
                            .color(colors.cost_text)
                            .small()
                    ).on_hover_text("Estimated API cost of all sessions. The total is saved in total_cost.txt.");
                });
            });
            if let Some(e) = toggle_error {
                self.set_status_error(e);
            }

            ui.add_space(10.0);

            // Activity Log (fixed height with scroll)
            ui.label(egui::RichText::new("Activity Log").strong());
            egui::Frame::none()
                .fill(colors.panel_background)
                .inner_margin(6.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    egui::ScrollArea::vertical()
                        .id_salt("activity_log_scroll")
                        .max_height(120.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            if self.log_entries.is_empty() {
                                ui.label(egui::RichText::new("No activity yet...").italics().color(colors.text_muted));
                            } else {
                                for entry in &self.log_entries {
                                    let timestamp = self.format_timestamp(entry);
                                    let color = match entry.level {
                                        LogLevel::Info => colors.log_info,
                                        LogLevel::Success => colors.log_success,
                                        LogLevel::Error => colors.log_error,
                                    };
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!("[{}]", timestamp))
                                                .color(colors.log_timestamp)
                                                .small()
                                        );
                                        ui.label(
                                            egui::RichText::new(&entry.message)
                                                .color(color)
                                        );
                                    });
                                }
                            }
                        });
                });

            ui.separator();
            ui.add_space(10.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                let grid_spacing = GRID_SPACING;
                let label_width = LABEL_WIDTH;

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

                            ui.label("Model:").on_hover_text(
                                "OpenAI model for translation. Type a model name, or select one from the list.",
                            );
                            model_name_edit(
                                ui,
                                "model_presets",
                                &mut self.config_draft.openai.model,
                                &models::suggested_chat_models(),
                            );
                            ui.end_row();

                            ui.label("Transcription:").on_hover_text(
                                "OpenAI model for speech to text. Type a model name, or select one from the list.",
                            );
                            model_name_edit(
                                ui,
                                "transcription_model_presets",
                                &mut self.config_draft.openai.transcription_model,
                                &models::suggested_transcription_models(),
                            );
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
                egui::CollapsingHeader::new("Audio Settings")
                    .show(ui, |ui| self.audio_settings_ui(ui));

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

            });
        });

        // Apply theme change if requested
        if theme_changed {
            self.config_draft.theme = new_theme;
            self.colors = theme::get_colors(new_theme);
            ctx.set_visuals(theme::get_visuals(new_theme));
        }

        // Other frames come from input and from GuiWaker, which wakes the
        // GUI for new log entries and cost. The status message needs one
        // more frame to hide it when it expires.
        if let Some((_, _, shown_at)) = &self.status_message {
            ctx.request_repaint_after(STATUS_MESSAGE_TIME.saturating_sub(shown_at.elapsed()));
        }
    }

    /// Body of the Audio Settings section: live meters, the test button and
    /// the audio settings.
    pub(crate) fn audio_settings_ui(&mut self, ui: &mut egui::Ui) {
        // The meters show state of the audio thread, which does not wake the
        // GUI. Not while minimized: nothing is visible, and eframe still runs
        // a frame for each request.
        let minimized = ui.ctx().input(|i| i.viewport().minimized == Some(true));
        if !minimized {
            ui.ctx().request_repaint_after(METER_REFRESH_INTERVAL);
        }

        let colors = self.colors;
        // Audio level meter at the top
        ui.label("Input Level:");
        let level_bits = self.app_state.current_audio_level.load(Ordering::Relaxed);
        let current_level = f32::from_bits(level_bits);
        draw_audio_level_meter(
            ui,
            current_level,
            &mut self.config_draft.audio.noise_gate_threshold,
            &colors,
        );
        ui.add_space(4.0);

        // Read audio state from atomics
        let is_recording = self.app_state.is_recording.load(Ordering::Relaxed);
        let silent_frames = self.app_state.silent_frames.load(Ordering::Relaxed);
        let noise_gate_active = self.app_state.noise_gate_active.load(Ordering::Relaxed);
        let hold_remaining = f32::from_bits(
            self.app_state
                .noise_gate_hold_remaining
                .load(Ordering::Relaxed),
        );
        let recording_duration =
            f32::from_bits(self.app_state.recording_duration.load(Ordering::Relaxed));

        // Noise gate state visualization
        ui.horizontal(|ui| {
            ui.label("Gate:");
            let gate_text = if noise_gate_active {
                if hold_remaining > 0.0 {
                    format!("Hold ({:.2}s)", hold_remaining)
                } else {
                    "Open".to_string()
                }
            } else {
                "Closed".to_string()
            };
            ui.label(egui::RichText::new(gate_text).small());
        });
        draw_noise_gate_state(
            ui,
            noise_gate_active,
            hold_remaining,
            self.config_draft.audio.noise_gate_hold_time,
            &colors,
        );
        ui.add_space(4.0);

        // Silence counter (only visible when recording)
        if is_recording {
            ui.horizontal(|ui| {
                ui.label("Silence:");
                ui.label(
                    egui::RichText::new(format!(
                        "{}/{}",
                        silent_frames, self.config_draft.audio.silence_threshold
                    ))
                    .small(),
                );
            });
            draw_silence_counter(
                ui,
                silent_frames,
                self.config_draft.audio.silence_threshold,
                &colors,
            );
            ui.add_space(4.0);

            // Recording duration
            ui.horizontal(|ui| {
                ui.label("Duration:");
                let min_dur = self.config_draft.audio.min_transcription_duration;
                let status = if recording_duration >= min_dur {
                    format!("{:.1}s (ready)", recording_duration)
                } else {
                    format!("{:.1}s / {:.1}s", recording_duration, min_dur)
                };
                ui.label(egui::RichText::new(status).small());
            });
            draw_recording_duration(
                ui,
                recording_duration,
                self.config_draft.audio.min_transcription_duration,
                &colors,
            );
            ui.add_space(4.0);
        }

        // Test Microphone button
        let is_testing = self.app_state.test_mode_active.load(Ordering::Relaxed);
        ui.horizontal(|ui| {
            if is_testing {
                if ui
                    .button("Stop Recording")
                    .on_hover_text("Stop recording and play back")
                    .clicked()
                {
                    if let Err(e) = self.send_command(AppCommand::StopTestRecording) {
                        self.set_status_error(format!("Failed to stop test: {}", e));
                    }
                }
            } else if ui
                .button("Test Microphone")
                .on_hover_text(format!(
                    "Record up to {} s of audio and play it back (click again to stop)",
                    TEST_RECORDING_LIMIT.as_secs()
                ))
                .clicked()
            {
                if let Err(e) = self.send_command(AppCommand::StartTestRecording) {
                    self.set_status_error(format!("Failed to start test: {}", e));
                }
            }
        });
        ui.add_space(8.0);

        egui::Grid::new("audio_grid")
            .num_columns(2)
            .spacing(GRID_SPACING)
            .min_col_width(LABEL_WIDTH)
            .show(ui, |ui| {
                ui.label("Silence Threshold:")
                    .on_hover_text(
                        "Number of silent audio buffers in a row, after the noise gate closes, \
                        before the recording stops and is sent. The audio device sets the buffer \
                        length. On most Windows devices one buffer is about 10 ms, so 100 is about \
                        1 second.",
                    );
                ui.add(egui::DragValue::new(&mut self.config_draft.audio.silence_threshold).range(1..=200));
                ui.end_row();

                ui.label("Noise Gate Threshold:")
                    .on_hover_text("Audio level below which input is considered silence (0.0-1.0). Drag the line on the meter or use this field.");
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
        Box::new(move |cc| {
            app_state.gui_waker.attach(cc.egui_ctx.clone());
            let mut app = BabbleBoopApp::new(app_state, log_rx);

            // Apply saved theme on startup
            let theme_mode = app.config_draft.theme;
            cc.egui_ctx.set_visuals(theme::get_visuals(theme_mode));

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
