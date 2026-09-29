use crate::app_state::{AppCommand, AppState, LogEntry, LogLevel};
use crate::config::{
    Config, ConfigWarning, ThemeMode, DISPLAY_TIME_MS_RANGE, MAX_AUDIO_FILES_RANGE,
    MAX_MESSAGE_CHUNKS_RANGE, MIN_TRANSCRIPTION_DURATION_RANGE, NOISE_GATE_HOLD_TIME_RANGE,
    NOISE_GATE_THRESHOLD_RANGE, PORT_RANGE, REQUESTS_PER_MINUTE_RANGE, SILENCE_DURATION_RANGE,
};
use crate::models;
use crate::processing_loop::TEST_RECORDING_LIMIT;
use crate::recorder::{RecorderStatus, MAX_RECORDING};
use crate::theme::{self, AppColors};
use eframe::egui;
use std::path::PathBuf;
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

/// Draw a progress bar showing the seconds of quiet input toward the
/// silence duration that ends the recording
fn draw_silence_counter(
    ui: &mut egui::Ui,
    quiet_time: f32,
    silence_duration: f32,
    colors: &AppColors,
) {
    let meter_size = egui::vec2(ui.available_width().min(200.0), 10.0);
    let (rect, _response) = ui.allocate_exact_size(meter_size, egui::Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 2.0, colors.meter_background);

        // Progress bar
        let progress = if silence_duration > 0.0 {
            (quiet_time / silence_duration).min(1.0)
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
    /// The file that the settings came from. Save writes it.
    config_file: PathBuf,
    config_draft: Config,
    saved_config: Config,
    /// The config file has values that the loader replaced, so the file
    /// differs from `saved_config` until the next successful save.
    file_differs: bool,
    /// The processing loop did not get `saved_config`, because the command
    /// could not be sent, so Save sends it again.
    saved_config_not_applied: bool,
    status_message: Option<(String, StatusType, std::time::Instant)>,
    log_rx: mpsc::Receiver<LogEntry>,
    log_entries: Vec<LogEntry>,
    start_time: std::time::Instant,
    colors: AppColors,
}

impl BabbleBoopApp {
    /// The window for the settings in `config`, which came from
    /// `config_file`.
    pub fn new(
        app_state: Arc<AppState>,
        config: Config,
        log_rx: mpsc::Receiver<LogEntry>,
        config_file: PathBuf,
    ) -> Self {
        let config_draft = config.clone();
        let saved_config = config;
        let colors = theme::get_colors(config_draft.theme);
        Self {
            app_state,
            config_file,
            config_draft,
            saved_config,
            file_differs: false,
            saved_config_not_applied: false,
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

    /// Show the settings as unsaved until the next successful save if the
    /// loader replaced values of the config file, so that Save can write
    /// the values in use to the file.
    fn note_replaced_values(&mut self, warnings: &[ConfigWarning]) {
        self.file_differs = !warnings.is_empty();
    }

    fn draft_changed(&self) -> bool {
        self.config_draft != self.saved_config
    }

    fn has_unsaved_changes(&self) -> bool {
        self.file_differs || self.draft_changed()
    }

    /// Save has something to do: write changes to the file, or send saved
    /// settings that the processing loop did not get.
    fn can_save(&self) -> bool {
        self.has_unsaved_changes() || self.saved_config_not_applied
    }

    fn reload_config(&mut self, ctx: &egui::Context) {
        self.config_draft = self.saved_config.clone();
        self.colors = theme::get_colors(self.config_draft.theme);
        ctx.set_visuals(theme::get_visuals(self.config_draft.theme));
        self.set_status_info("Changes discarded");
    }

    /// The draft to save, or why it cannot be saved. Removes spaces and
    /// line breaks around every text field first. The API does not know a
    /// model name with a space at the end. A pasted API key often has a
    /// line break at the end, and a request cannot put a line break in its
    /// Authorization header (http 1.4.0, src/header/value.rs:557).
    fn config_to_save(&mut self) -> Result<Config, String> {
        let draft = &mut self.config_draft;
        for text in [
            &mut draft.osc.address,
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
    /// loop, which applies it.
    fn save_config(&mut self, new_config: Config) {
        if let Err(e) = new_config.save(&self.config_file) {
            self.set_status_error(format!("Failed to save: {}", e));
            return;
        }
        self.saved_config = new_config.clone();
        self.file_differs = false;
        match self.send_command(AppCommand::UpdateConfig(new_config)) {
            Ok(()) => {
                self.saved_config_not_applied = false;
                self.set_status_success("Settings saved successfully");
            }
            Err(e) => {
                self.saved_config_not_applied = true;
                self.set_status_error(format!("Settings saved to file, but not applied. {}", e));
            }
        }
    }

    fn validate_config(&self) -> Result<(), String> {
        if self.config_draft.osc.address.trim().is_empty() {
            return Err("OSC address is required".to_string());
        }

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

    fn set_status_info(&mut self, msg: impl Into<String>) {
        self.status_message = Some((msg.into(), StatusType::Info, std::time::Instant::now()));
    }

    /// Send `cmd` to the processing loop. Does not wait: the GUI thread
    /// must not block while the loop is busy, so a full or closed channel
    /// is an error.
    fn send_command(&self, cmd: AppCommand) -> Result<(), String> {
        self.app_state
            .command_tx
            .try_send(cmd)
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
                } else if self.saved_config_not_applied {
                    ui.label(
                        egui::RichText::new("● Settings not applied")
                            .color(colors.unsaved_indicator),
                    )
                    .on_hover_text(
                        "The settings are saved to the file, but are not in use. \
                        Click Save Settings to apply them.",
                    );
                    ui.separator();
                }

                // Reset button (only show if the draft has changes)
                if self.draft_changed()
                    && ui
                        .button("Reset")
                        .on_hover_text("Discard changes and reload saved settings")
                        .clicked()
                {
                    should_reload = true;
                }

                // Save button
                let save_button =
                    ui.add_enabled(self.can_save(), egui::Button::new("Save Settings"));

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
        // Tell the processing loop and the audio input thread to stop,
        // through `Shutdown`. This does not close the command channel. The
        // channel closes when processing stops, which can come first, for
        // example after a startup error. `main.rs` requests shutdown again
        // after `run_gui` returns.
        self.app_state.request_shutdown();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui(ctx);
    }
}

impl BabbleBoopApp {
    /// Show one frame. Separate from `update` so tests can run it without
    /// an `eframe::Frame`.
    fn ui(&mut self, ctx: &egui::Context) {
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
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.input_port).range(PORT_RANGE));
                            ui.end_row();

                            ui.label("Output Port:")
                                .on_hover_text("Port to send OSC messages to VRChat");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.output_port).range(PORT_RANGE));
                            ui.end_row();

                            ui.label("Display Time (ms):")
                                .on_hover_text("How long messages stay visible in VRChat");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.display_time).range(DISPLAY_TIME_MS_RANGE));
                            ui.end_row();

                            ui.label("Max Message Chunks:")
                                .on_hover_text("Maximum number of message parts for long text");
                            ui.add(egui::DragValue::new(&mut self.config_draft.osc.max_message_chunks).range(MAX_MESSAGE_CHUNKS_RANGE));
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
                            ).range(REQUESTS_PER_MINUTE_RANGE));
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
                                ui.add(egui::DragValue::new(&mut self.config_draft.max_audio_files).range(MAX_AUDIO_FILES_RANGE));
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
    fn audio_settings_ui(&mut self, ui: &mut egui::Ui) {
        // The meters show state of the audio thread, which does not wake the
        // GUI. Not while minimized: nothing is visible, and eframe still runs
        // a frame for each request (eframe 0.29.1
        // src/native/glow_integration.rs:484-730 does not skip a minimized
        // window, it only sleeps 10 ms after the frame).
        //
        // Only Windows and X11 report the minimized state, so on macOS and
        // Wayland the refresh continues while minimized. egui-winit 0.29.1
        // reads the state after window creation only when not on macOS
        // (src/lib.rs:987-993). winit 0.30.12 `is_minimized` always returns
        // `None` on Wayland
        // (src/platform_impl/linux/wayland/window/mod.rs:355-357), and
        // egui-winit reads `None` as not minimized. egui 0.29.1
        // `ViewportInfo` has no occlusion field (src/data/input.rs:224-236).
        // Its `focused` field is not usable here: the user can look at the
        // meters while another application has focus.
        let minimized = ui.ctx().input(|i| i.viewport().minimized == Some(true));
        if !minimized {
            ui.ctx().request_repaint_after(METER_REFRESH_INTERVAL);
        }

        let colors = self.colors;
        // Audio level meter at the top
        ui.label("Input Level:");
        let current_level = self.app_state.audio.level.load();
        draw_audio_level_meter(
            ui,
            current_level,
            &mut self.config_draft.audio.noise_gate_threshold,
            &colors,
        );
        ui.add_space(4.0);

        // One snapshot, so all lines below show the same buffer
        let RecorderStatus {
            is_recording,
            quiet_time,
            gate_open: noise_gate_active,
            hold_remaining,
            recording_duration,
            split: recording_split,
        } = self.app_state.audio.status();

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
            let silence_duration = self.config_draft.audio.silence_duration;
            ui.horizontal(|ui| {
                ui.label("Silence:").on_hover_text(
                    "Seconds of silence since the noise gate closed, and the silence duration \
                    that ends the recording",
                );
                ui.label(
                    egui::RichText::new(format!("{:.1}s / {:.1}s", quiet_time, silence_duration))
                        .small(),
                );
            });
            draw_silence_counter(ui, quiet_time, silence_duration, &colors);
            ui.add_space(4.0);

            // Recording duration. The minimum transcription duration does
            // not apply to the last part of a recording that reached the
            // length limit.
            let min_duration_shown = if recording_split {
                0.0
            } else {
                self.config_draft.audio.min_transcription_duration
            };
            ui.horizontal(|ui| {
                ui.label("Duration:");
                let min_dur = min_duration_shown;
                let status = if recording_duration >= min_dur {
                    format!("{:.1}s (ready)", recording_duration)
                } else {
                    format!("{:.1}s / {:.1}s", recording_duration, min_dur)
                };
                ui.label(egui::RichText::new(status).small());
            });
            draw_recording_duration(ui, recording_duration, min_duration_shown, &colors);
            ui.add_space(4.0);
        }

        // Test Microphone button
        let is_testing = self.app_state.audio.test_mode.load(Ordering::Relaxed);
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
                ui.label("Silence Duration (s):").on_hover_text(
                    "Seconds of silence, after the noise gate closes, before the recording \
                    stops and is sent",
                );
                ui.add(
                    egui::DragValue::new(&mut self.config_draft.audio.silence_duration)
                        .speed(0.05)
                        .range(SILENCE_DURATION_RANGE),
                );
                ui.end_row();

                ui.label("Noise Gate Threshold:")
                    .on_hover_text("Audio level below which input is considered silence (0.0-1.0). Drag the line on the meter or use this field.");
                ui.add(
                    egui::DragValue::new(&mut self.config_draft.audio.noise_gate_threshold)
                        .speed(0.01)
                        .range(NOISE_GATE_THRESHOLD_RANGE),
                );
                ui.end_row();

                ui.label("Noise Gate Hold (s):")
                    .on_hover_text("Time to keep gate open after audio drops below threshold");
                ui.add(
                    egui::DragValue::new(&mut self.config_draft.audio.noise_gate_hold_time)
                        .speed(0.01)
                        .range(NOISE_GATE_HOLD_TIME_RANGE),
                );
                ui.end_row();

                ui.label("Min Duration (s):").on_hover_text(format!(
                    "Recordings shorter than this are not transcribed (filters out noise). \
                    A recording that reaches {} s is split into parts. The minimum does not \
                    apply to these parts.",
                    MAX_RECORDING.as_secs()
                ));
                ui.add(
                    egui::DragValue::new(&mut self.config_draft.audio.min_transcription_duration)
                        .speed(0.1)
                        .range(MIN_TRANSCRIPTION_DURATION_RANGE),
                );
                ui.end_row();
            });
    }
}

pub fn run_gui(
    app_state: Arc<AppState>,
    config: Config,
    log_rx: mpsc::Receiver<LogEntry>,
    first_run: bool,
    config_warnings: Vec<ConfigWarning>,
    config_file: PathBuf,
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
            let mut app = BabbleBoopApp::new(app_state, config, log_rx, config_file);
            app.note_replaced_values(&config_warnings);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{AppCommand, AppState, LogEntry, LogLevel};
    use crate::config::Config;
    use crate::recorder::RecorderStatus;
    use eframe::egui;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// A config file in a folder that does not exist, so a click on Save
    /// fails instead of writing a file.
    fn unused_config_file() -> PathBuf {
        std::env::temp_dir()
            .join(format!("babble_boop_no_config_dir_{}", std::process::id()))
            .join("config.toml")
    }

    fn test_app_with_state() -> (BabbleBoopApp, Arc<AppState>) {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        (
            BabbleBoopApp::new(
                Arc::clone(&app_state),
                Config::default(),
                log_rx,
                unused_config_file(),
            ),
            app_state,
        )
    }

    fn test_app() -> (BabbleBoopApp, tokio::sync::mpsc::Sender<LogEntry>) {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx.clone()));
        let app = BabbleBoopApp::new(app_state, Config::default(), log_rx, unused_config_file());
        (app, log_tx)
    }

    fn raw_input() -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 600.0),
            )),
            ..Default::default()
        }
    }

    /// Run a few frames, so that the first frame layout passes are over,
    /// and return how long the GUI asks to wait before the next frame.
    /// `Duration::MAX` means it waits for input or a wake up.
    fn repaint_delay_after_frames(
        ctx: &egui::Context,
        mut run_ui: impl FnMut(&egui::Context),
    ) -> Duration {
        let mut output = ctx.run(raw_input(), &mut run_ui);
        for _ in 0..4 {
            output = ctx.run(raw_input(), &mut run_ui);
        }
        output.viewport_output[&egui::ViewportId::ROOT].repaint_delay
    }

    #[test]
    fn test_idle_gui_with_log_entries_does_not_repaint() {
        let (mut app, log_tx) = test_app();
        log_tx
            .try_send(LogEntry {
                timestamp: Instant::now(),
                message: "Starting audio recording...".to_string(),
                level: LogLevel::Info,
            })
            .unwrap();

        let delay = repaint_delay_after_frames(&egui::Context::default(), |ctx| app.ui(ctx));

        assert_eq!(delay, Duration::MAX);
    }

    #[test]
    fn test_status_message_repaints_when_it_expires() {
        let (mut app, _log_tx) = test_app();
        app.set_status_info("Settings saved successfully");

        let delay = repaint_delay_after_frames(&egui::Context::default(), |ctx| app.ui(ctx));

        assert!(delay > Duration::from_secs(2), "delay {:?}", delay);
        assert!(delay <= Duration::from_secs(3), "delay {:?}", delay);
    }

    #[test]
    fn test_audio_settings_keep_meters_live() {
        let (mut app, _log_tx) = test_app();

        let delay = repaint_delay_after_frames(&egui::Context::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        });

        assert!(delay <= Duration::from_millis(50), "delay {:?}", delay);
    }

    #[test]
    fn test_audio_settings_do_not_repaint_while_minimized() {
        let (mut app, _log_tx) = test_app();
        let ctx = egui::Context::default();
        let minimized = || {
            let mut input = raw_input();
            input
                .viewports
                .entry(egui::ViewportId::ROOT)
                .or_default()
                .minimized = Some(true);
            input
        };
        let mut run_ui = |ctx: &egui::Context| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        };

        let mut output = ctx.run(minimized(), &mut run_ui);
        for _ in 0..4 {
            output = ctx.run(minimized(), &mut run_ui);
        }

        assert_eq!(
            output.viewport_output[&egui::ViewportId::ROOT].repaint_delay,
            Duration::MAX
        );
    }

    /// Text of all labels painted in `output`. Labels outside the visible
    /// part of a scroll area are not painted.
    fn painted_text(output: &egui::FullOutput) -> Vec<String> {
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) => Some(text.galley.text().to_string()),
                _ => None,
            })
            .collect()
    }

    /// Run frames the way eframe does: one, then one more for each frame
    /// that asks for a repaint at once. Returns the last frame's output.
    fn run_until_idle(ctx: &egui::Context, app: &mut BabbleBoopApp) -> egui::FullOutput {
        for _ in 0..10 {
            let output = ctx.run(raw_input(), |ctx| app.ui(ctx));
            if output.viewport_output[&egui::ViewportId::ROOT].repaint_delay != Duration::ZERO {
                return output;
            }
        }
        panic!("the GUI still repaints continuously after 10 frames");
    }

    #[test]
    fn test_new_log_entry_is_visible_when_gui_goes_idle() {
        let (mut app, app_state) = test_app_with_state();
        let ctx = egui::Context::default();
        app_state.gui_waker.attach(ctx.clone());
        // More entries than the log shows, so the log scrolls.
        for i in 0..20 {
            app_state.logger.info(format!("entry {}", i));
        }
        run_until_idle(&ctx, &mut app);

        // The GUI is idle. The processing thread logs one more line.
        app_state.logger.info("Translation: hallo");
        assert!(ctx.has_requested_repaint());
        let output = run_until_idle(&ctx, &mut app);

        assert!(
            painted_text(&output).contains(&"Translation: hallo".to_string()),
            "{:?}",
            painted_text(&output)
        );
    }

    // ===========================================================================
    // Test: Meters and the toggle follow the theme
    // ===========================================================================

    fn themed_app(theme: crate::config::ThemeMode) -> (BabbleBoopApp, Arc<AppState>) {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let config = Config {
            theme,
            ..Config::default()
        };
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        (
            BabbleBoopApp::new(Arc::clone(&app_state), config, log_rx, unused_config_file()),
            app_state,
        )
    }

    /// Filled rectangles of exactly `width` x `height` points in `output`.
    fn painted_rect_fills(
        output: &egui::FullOutput,
        width: f32,
        height: f32,
    ) -> Vec<egui::Color32> {
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect)
                    if rect.rect.width() == width && rect.rect.height() == height =>
                {
                    Some(rect.fill)
                }
                _ => None,
            })
            .collect()
    }

    fn is_light(color: egui::Color32) -> bool {
        (u32::from(color.r()) + u32::from(color.g()) + u32::from(color.b())) / 3 > 128
    }

    #[test]
    fn test_meter_tracks_follow_the_theme() {
        use crate::config::ThemeMode;

        for theme in [ThemeMode::Dark, ThemeMode::Light] {
            let (mut app, app_state) = themed_app(theme);
            // Recording shows the silence and duration meters. Level,
            // silence and duration are zero, so only the tracks are painted.
            app_state.audio.publish(RecorderStatus {
                is_recording: true,
                ..RecorderStatus::default()
            });
            let output = egui::Context::default().run(raw_input(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
            });

            let window_is_light = is_light(crate::theme::get_visuals(theme).panel_fill);
            let level_track = painted_rect_fills(&output, 200.0, 16.0);
            let small_tracks = painted_rect_fills(&output, 200.0, 10.0);
            assert_eq!(level_track.len(), 1, "{:?}", theme);
            // Gate, silence and duration.
            assert_eq!(small_tracks.len(), 3, "{:?}", theme);
            for track in level_track.iter().chain(&small_tracks) {
                assert_eq!(
                    is_light(*track),
                    window_is_light,
                    "{:?} theme meter track {:?}",
                    theme,
                    track
                );
            }
        }
    }

    // ===========================================================================
    // Test: The duration line shows when a recording can be sent
    // ===========================================================================

    /// One frame of the audio settings while a recording that is 0.5 s long
    /// is in progress, with a minimum transcription duration of 1 s.
    fn recording_frame(recording_split: bool) -> egui::FullOutput {
        let (mut app, app_state) = test_app_with_state();
        app.config_draft.audio.min_transcription_duration = 1.0;
        app_state.audio.publish(RecorderStatus {
            is_recording: true,
            recording_duration: 0.5,
            split: recording_split,
            ..RecorderStatus::default()
        });
        egui::Context::default().run(raw_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        })
    }

    /// The painted "Duration:" status in the frame from `recording_frame`.
    fn duration_status(recording_split: bool) -> Vec<String> {
        painted_text(&recording_frame(recording_split))
            .into_iter()
            .filter(|text| text.starts_with("0.5s"))
            .collect()
    }

    /// Widths of the filled parts of the 10 point high meters in the frame
    /// from `recording_frame`. The noise gate is closed and there is no
    /// quiet time, so only the duration meter has a filled part.
    fn duration_bar_fill_widths(recording_split: bool) -> Vec<f32> {
        small_meter_fill_widths(&recording_frame(recording_split))
    }

    /// Widths of the filled parts of the 10 point high meters in `output`.
    fn small_meter_fill_widths(output: &egui::FullOutput) -> Vec<f32> {
        let track = crate::theme::get_colors(Config::default().theme).meter_background;
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect) if rect.rect.height() == 10.0 && rect.fill != track => {
                    Some(rect.rect.width())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_last_part_of_a_split_recording_shows_ready_below_the_minimum() {
        // The minimum applies only to a whole recording, not to the last
        // part of a recording that reached the length limit.
        assert_eq!(duration_status(true), ["0.5s (ready)"]);
    }

    #[test]
    fn test_whole_recording_below_the_minimum_shows_not_ready() {
        assert_eq!(duration_status(false), ["0.5s / 1.0s"]);
    }

    #[test]
    fn test_last_part_of_a_split_recording_fills_the_duration_bar() {
        // The 200 point bar is full, because the minimum does not apply.
        assert_eq!(duration_bar_fill_widths(true), [200.0]);
    }

    #[test]
    fn test_whole_recording_below_the_minimum_half_fills_the_duration_bar() {
        // 0.5 s of the 1 s minimum fills half of the 200 point bar.
        assert_eq!(duration_bar_fill_widths(false), [100.0]);
    }

    // ===========================================================================
    // Test: The silence line shows seconds
    // ===========================================================================

    /// One frame of the audio settings while a recording is in progress,
    /// 0.5 s after the noise gate closed, with a silence duration of 2 s.
    fn silence_frame() -> egui::FullOutput {
        let (mut app, app_state) = test_app_with_state();
        app.config_draft.audio.silence_duration = 2.0;
        app_state.audio.publish(RecorderStatus {
            is_recording: true,
            quiet_time: 0.5,
            ..RecorderStatus::default()
        });
        egui::Context::default().run(raw_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        })
    }

    #[test]
    fn test_silence_line_shows_the_seconds_of_silence_and_the_duration() {
        let text = painted_text(&silence_frame());
        assert!(text.contains(&"0.5s / 2.0s".to_string()), "{:?}", text);
    }

    #[test]
    fn test_silence_bar_fills_by_seconds() {
        // 0.5 s of the 2 s duration fills a quarter of the 200 point bar.
        // The gate is closed and the duration is 0, so only the silence
        // meter has a filled part.
        assert_eq!(small_meter_fill_widths(&silence_frame()), [50.0]);
    }

    // ===========================================================================
    // Test: The audio settings show the published recorder status
    // ===========================================================================

    /// The texts from the gate state to the Test Microphone button in one
    /// frame of the audio settings, with a silence duration of 3 s and a
    /// minimum transcription duration of 2 s.
    fn status_texts(app: &mut BabbleBoopApp) -> Vec<String> {
        app.config_draft.audio.silence_duration = 3.0;
        app.config_draft.audio.min_transcription_duration = 2.0;
        let output = egui::Context::default().run(raw_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.audio_settings_ui(ui));
        });
        painted_text(&output)
            .into_iter()
            .skip_while(|text| text != "Gate:")
            .skip(1)
            .take_while(|text| text != "Test Microphone")
            .collect()
    }

    #[test]
    fn test_the_audio_settings_show_the_published_status() {
        let (mut app, app_state) = test_app_with_state();
        app_state.audio.publish(RecorderStatus {
            is_recording: true,
            quiet_time: 0.7,
            gate_open: true,
            hold_remaining: 0.25,
            recording_duration: 1.5,
            split: false,
        });
        assert_eq!(
            status_texts(&mut app),
            [
                "Hold (0.25s)",
                "Silence:",
                "0.7s / 3.0s",
                "Duration:",
                "1.5s / 2.0s"
            ]
        );

        app_state.audio.publish(RecorderStatus {
            is_recording: true,
            quiet_time: 0.2,
            gate_open: false,
            hold_remaining: 0.0,
            recording_duration: 1.0,
            split: true,
        });
        assert_eq!(
            status_texts(&mut app),
            [
                "Closed",
                "Silence:",
                "0.2s / 3.0s",
                "Duration:",
                "1.0s (ready)"
            ]
        );

        app_state.audio.publish(RecorderStatus {
            gate_open: true,
            ..RecorderStatus::default()
        });
        assert_eq!(status_texts(&mut app), ["Open"]);
    }

    #[test]
    fn test_toggle_off_track_follows_the_theme() {
        use crate::config::ThemeMode;
        use std::sync::atomic::Ordering;

        for theme in [ThemeMode::Dark, ThemeMode::Light] {
            let (mut app, app_state) = themed_app(theme);
            app_state.enabled.store(false, Ordering::Relaxed);
            let output = egui::Context::default().run(raw_input(), |ctx| app.ui(ctx));

            let window_is_light = is_light(crate::theme::get_visuals(theme).panel_fill);
            let toggle = painted_rect_fills(&output, 36.0, 20.0);
            assert_eq!(toggle.len(), 1, "{:?}", theme);
            assert_eq!(is_light(toggle[0]), window_is_light, "{:?} theme", theme);
        }
    }

    // ===========================================================================
    // Test: The translation model takes a custom name
    // ===========================================================================

    /// Centre of the first painted text that reads `text`.
    fn text_center(output: &egui::FullOutput, text: &str) -> egui::Pos2 {
        output
            .shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                egui::Shape::Text(shape) if shape.galley.text() == text => {
                    Some(egui::Rect::from_min_size(shape.pos, shape.galley.size()).center())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("no painted text {:?}", text))
    }

    fn with_events(events: Vec<egui::Event>) -> egui::RawInput {
        egui::RawInput {
            events,
            ..raw_input()
        }
    }

    /// Run one frame with `events` for what it does to `app`.
    fn run_with_events(ctx: &egui::Context, app: &mut BabbleBoopApp, events: Vec<egui::Event>) {
        #[expect(
            clippy::let_underscore_must_use,
            reason = "the tests that call this do not check the painted output"
        )]
        let _ = ctx.run(with_events(events), |ctx| app.ui(ctx));
    }

    #[test]
    fn test_custom_translation_model_can_be_typed() {
        let (mut app, _log_tx) = test_app();
        let ctx = egui::Context::default();
        let mut output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        for _ in 0..2 {
            output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        }
        let model = text_center(&output, &Config::default().openai.model);

        click(&ctx, &mut app, model);
        let select_all = egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        };
        let typed = egui::Event::Text("my-finetuned-model".to_string());
        run_with_events(&ctx, &mut app, vec![select_all, typed]);

        assert_eq!(app.config_draft.openai.model, "my-finetuned-model");
    }

    #[test]
    fn test_custom_transcription_model_can_be_typed() {
        let (mut app, _log_tx) = test_app();
        let ctx = egui::Context::default();
        let mut output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        for _ in 0..2 {
            output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        }
        let model = text_center(&output, &Config::default().openai.transcription_model);

        click(&ctx, &mut app, model);
        let select_all = egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        };
        let typed = egui::Event::Text("my-transcriber".to_string());
        run_with_events(&ctx, &mut app, vec![select_all, typed]);

        assert_eq!(
            app.config_draft.openai.transcription_model,
            "my-transcriber"
        );
    }

    #[test]
    fn test_translation_model_preset_can_be_selected() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.model = "my-finetuned-model".to_string();
        let ctx = egui::Context::default();
        let mut output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        for _ in 0..2 {
            output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        }
        let model = text_center(&output, "my-finetuned-model");
        // The preset list button is the next widget right of the text field.
        let list_button = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Rect(rect)
                    if rect.rect.x_range().min > model.x
                        && rect.rect.y_range().contains(model.y) =>
                {
                    Some(rect.rect)
                }
                _ => None,
            })
            .min_by(|a, b| a.min.x.total_cmp(&b.min.x))
            .expect("no preset list button")
            .center();

        click(&ctx, &mut app, list_button);
        let output = ctx.run(raw_input(), |ctx| app.ui(ctx));
        click(&ctx, &mut app, text_center(&output, "gpt-6-sol"));

        assert_eq!(app.config_draft.openai.model, "gpt-6-sol");
    }

    #[test]
    fn test_model_names_are_trimmed_on_save() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = "test-key".to_string();
        app.config_draft.openai.model = "gpt-6-sol ".to_string();
        app.config_draft.openai.transcription_model = "\tgpt-transcribe\n".to_string();

        let saved = app.config_to_save().expect("config is valid");

        assert_eq!(saved.openai.model, "gpt-6-sol");
        assert_eq!(saved.openai.transcription_model, "gpt-transcribe");
        // The settings show what was saved.
        assert_eq!(app.config_draft, saved);
    }

    #[test]
    fn test_api_key_and_target_language_are_trimmed_on_save() {
        let (mut app, _log_tx) = test_app();
        // A pasted key often has a space or a line break at an end. The
        // Authorization header must not contain them.
        app.config_draft.openai.api_key = " sk-test \n".to_string();
        app.config_draft.translation.target_language = "\tFrench ".to_string();

        let saved = app.config_to_save().expect("config is valid");

        assert_eq!(saved.openai.api_key, "sk-test");
        assert_eq!(saved.translation.target_language, "French");
        // The settings show what was saved.
        assert_eq!(app.config_draft, saved);
    }

    /// Every free text field of the config, found through its serialized
    /// form, so that a text field added later is checked too.
    #[test]
    fn test_every_text_field_is_trimmed_on_save() {
        fn text_paths(value: &toml::Value, path: &str, out: &mut Vec<String>) {
            match value {
                toml::Value::String(_) => out.push(path.to_string()),
                toml::Value::Table(table) => {
                    for (key, child) in table {
                        let child_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{path}.{key}")
                        };
                        text_paths(child, &child_path, out);
                    }
                }
                _ => {}
            }
        }
        fn text_at<'a>(value: &'a mut toml::Value, path: &str) -> &'a mut String {
            let mut value = value;
            for key in path.split('.') {
                value = value.get_mut(key).expect("path exists");
            }
            match value {
                toml::Value::String(text) => text,
                _ => panic!("{path} is not text"),
            }
        }

        let mut valid = Config::default();
        valid.openai.api_key = "test-key".to_string();
        let valid = toml::Value::try_from(&valid).expect("config serializes");
        let mut paths = Vec::new();
        text_paths(&valid, "", &mut paths);

        let mut checked = Vec::new();
        for path in paths {
            let mut padded = valid.clone();
            let text = text_at(&mut padded, &path);
            *text = format!(" {text}\n");
            // A value that does not load with spaces around it, such as the
            // theme, is a fixed choice and not free text.
            let Ok(draft) = padded.try_into::<Config>() else {
                continue;
            };
            let (mut app, _log_tx) = test_app();
            app.config_draft = draft;

            let saved = app.config_to_save().expect("config is valid");

            let mut saved = toml::Value::try_from(&saved).expect("config serializes");
            let mut expected = valid.clone();
            assert_eq!(text_at(&mut saved, &path), text_at(&mut expected, &path));
            checked.push(path);
        }
        assert!(checked.contains(&"openai.api_key".to_string()));
    }

    #[test]
    fn test_blank_api_key_is_not_saved() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = " \n".to_string();

        assert!(app.config_to_save().is_err());
    }

    #[test]
    fn test_blank_transcription_model_is_not_saved() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = "test-key".to_string();
        app.config_draft.openai.transcription_model = " ".to_string();

        assert!(app.config_to_save().is_err());
    }

    #[test]
    fn test_blank_osc_address_is_not_saved() {
        let (mut app, _log_tx) = test_app();
        app.config_draft.openai.api_key = "test-key".to_string();
        app.config_draft.osc.address = " \n".to_string();

        let error = app.config_to_save().unwrap_err();
        assert!(error.contains("address"), "{:?}", error);
    }

    /// The settings window clamps a value to the range of its field when it
    /// draws the field, even if the user does not touch it. A config that
    /// loads unchanged must also stay unchanged in the window.
    #[test]
    fn test_high_limits_are_not_clamped_in_the_settings() {
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let config = Config {
            keep_audio_files: true,
            max_audio_files: 500,
            rate_limit: crate::config::RateLimitConfig {
                requests_per_minute: 500,
            },
            ..Config::default()
        };
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, config.clone(), log_rx, unused_config_file());
        // Tall enough that the settings below the log are not scrolled away
        let tall = |events| egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(400.0, 3000.0),
            )),
            events,
            ..Default::default()
        };
        let ctx = egui::Context::default();
        let mut output = ctx.run(tall(Vec::new()), |ctx| app.ui(ctx));

        // Open the two sections, which are closed at the start
        for header in ["Rate Limit", "Debug"] {
            let pos = text_center(&output, header);
            let button = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            let press = vec![egui::Event::PointerMoved(pos), button(true)];
            #[expect(
                clippy::let_underscore_must_use,
                reason = "only the output after the release is used"
            )]
            let _ = ctx.run(tall(press), |ctx| app.ui(ctx));
            output = ctx.run(tall(vec![button(false)]), |ctx| app.ui(ctx));
        }
        for _ in 0..3 {
            output = ctx.run(tall(Vec::new()), |ctx| app.ui(ctx));
        }

        // Both fields were drawn, so the clamp had its chance to run
        let text = painted_text(&output);
        assert!(text.contains(&"Max Audio Files:".to_string()), "{:?}", text);
        assert!(
            text.contains(&"Requests per Minute:".to_string()),
            "{:?}",
            text
        );
        assert_eq!(app.config_draft, config);
    }

    /// Press and release the primary button at `pos`, in two frames.
    fn click(ctx: &egui::Context, app: &mut BabbleBoopApp, pos: egui::Pos2) {
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        run_with_events(ctx, app, vec![egui::Event::PointerMoved(pos), button(true)]);
        run_with_events(ctx, app, vec![button(false)]);
    }

    // ===========================================================================
    // Test: The GUI shows that processing stopped
    // ===========================================================================

    #[test]
    fn test_stopped_processing_is_shown_instead_of_enabled() {
        let (mut app, app_state) = test_app_with_state();
        let ctx = egui::Context::default();
        app_state.gui_waker.attach(ctx.clone());
        let output = run_until_idle(&ctx, &mut app);
        assert!(painted_text(&output).contains(&"Enabled".to_string()));
        assert!(!ctx.has_requested_repaint());

        app_state.mark_processing_stopped();
        assert!(ctx.has_requested_repaint());
        let output = ctx.run(raw_input(), |ctx| app.ui(ctx));

        let text = painted_text(&output);
        assert!(
            text.contains(&"Processing stopped".to_string()),
            "{:?}",
            text
        );
        assert!(!text.contains(&"Enabled".to_string()), "{:?}", text);
    }

    // ===========================================================================
    // Test: A value that the loader replaced can be saved to the file
    // ===========================================================================

    const UNSAVED: &str = "● Unsaved changes";
    const NOT_APPLIED: &str = "● Settings not applied";

    /// The app after a start with the default config file, but with
    /// `osc.display_time` set to `display_time` in the file. The API key is
    /// blank, so a click on Save fails the validation and writes no file.
    fn app_from_file_with_display_time(display_time: i64) -> BabbleBoopApp {
        let mut file = toml::Value::try_from(Config::default()).unwrap();
        file["osc"]["display_time"] = toml::Value::Integer(display_time);
        let loaded = Config::from_toml(&toml::to_string(&file).unwrap()).unwrap();
        assert_eq!(
            loaded.config.openai.api_key, "",
            "a click on Save must fail the validation"
        );
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, loaded.config, log_rx, unused_config_file());
        app.note_replaced_values(&loaded.warnings);
        app
    }

    /// 999999 is above the maximum, so the loader uses 30000.
    fn app_with_replaced_display_time() -> BabbleBoopApp {
        app_from_file_with_display_time(999_999)
    }

    #[test]
    fn test_replaced_config_value_shows_unsaved_changes() {
        let mut app = app_with_replaced_display_time();

        let text = painted_text(&run_until_idle(&egui::Context::default(), &mut app));

        assert!(text.contains(&UNSAVED.to_string()), "{:?}", text);
        // Reset has nothing to discard, as the draft has the values in use
        assert!(!text.contains(&"Reset".to_string()), "{:?}", text);
    }

    #[test]
    fn test_replaced_config_value_can_be_saved() {
        let mut app = app_with_replaced_display_time();
        let ctx = egui::Context::default();
        let output = run_until_idle(&ctx, &mut app);

        click(&ctx, &mut app, text_center(&output, "Save Settings"));

        // The click reached the save handler, which refused the blank key
        let text = painted_text(&run_until_idle(&ctx, &mut app));
        assert!(
            text.contains(&"OpenAI API key is required".to_string()),
            "{:?}",
            text
        );
    }

    #[test]
    fn test_reset_keeps_the_replaced_value_and_the_unsaved_changes() {
        let mut app = app_with_replaced_display_time();
        let ctx = egui::Context::default();
        app.config_draft.osc.display_time = 5000;
        let output = run_until_idle(&ctx, &mut app);

        click(&ctx, &mut app, text_center(&output, "Reset"));

        let text = painted_text(&run_until_idle(&ctx, &mut app));
        assert!(
            text.contains(&"Changes discarded".to_string()),
            "{:?}",
            text
        );
        assert_eq!(app.config_draft.osc.display_time, 30000);
        assert!(text.contains(&UNSAVED.to_string()), "{:?}", text);
    }

    #[test]
    fn test_config_file_without_replaced_values_has_nothing_to_save() {
        let mut app = app_from_file_with_display_time(5000);
        let ctx = egui::Context::default();
        let output = run_until_idle(&ctx, &mut app);
        assert!(
            !painted_text(&output).contains(&UNSAVED.to_string()),
            "{:?}",
            painted_text(&output)
        );

        click(&ctx, &mut app, text_center(&output, "Save Settings"));

        let text = painted_text(&run_until_idle(&ctx, &mut app));
        assert!(
            !text.contains(&"OpenAI API key is required".to_string()),
            "{:?}",
            text
        );
    }

    // ===========================================================================
    // Test: Save writes the config file that the settings came from
    // ===========================================================================

    #[test]
    fn test_save_writes_the_config_file_that_was_loaded() {
        use std::fs;

        let dir = std::env::temp_dir().join(format!("babble_boop_gui_save_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        let config_file = dir.join("config.toml");
        let mut file_config = Config::default();
        file_config.openai.api_key = "sk-test".to_string();
        file_config.save(&config_file).unwrap();
        let loaded = Config::load(&config_file).unwrap();
        let (cmd_tx, _cmd_rx) = tokio::sync::mpsc::channel(10);
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, loaded.config, log_rx, config_file.clone());
        app.config_draft.translation.target_language = "German".to_string();
        let ctx = egui::Context::default();
        let output = run_until_idle(&ctx, &mut app);

        click(&ctx, &mut app, text_center(&output, "Save Settings"));

        let text = painted_text(&run_until_idle(&ctx, &mut app));
        let saved = Config::load(&config_file).map_err(|e| e.to_string());
        let files: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name())
            .collect();
        // Clean up before asserting, so a failure does not leave files behind
        fs::remove_dir_all(&dir).unwrap();

        assert!(
            text.contains(&"Settings saved successfully".to_string()),
            "{:?}",
            text
        );
        let saved = saved.unwrap().config;
        assert_eq!(saved.translation.target_language, "German");
        assert_eq!(saved.openai.api_key, "sk-test");
        assert_eq!(files, ["config.toml"]);
    }

    // ===========================================================================
    // Test: The GUI does not wait for a busy processing loop
    // ===========================================================================

    /// Run `body` on another thread, and fail if it does not return within
    /// 5 s. A GUI that waits for the processing loop blocks in it.
    fn without_blocking<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || done_tx.send(body()));
        done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the GUI waited for the processing loop")
    }

    /// A command channel with the room of the channel in `main.rs`, full of
    /// commands. The receiver reads none of them.
    fn full_command_channel() -> (
        tokio::sync::mpsc::Sender<AppCommand>,
        tokio::sync::mpsc::Receiver<AppCommand>,
    ) {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel(32);
        while cmd_tx.try_send(AppCommand::SetEnabled(true)).is_ok() {}
        assert_eq!(cmd_tx.capacity(), 0);
        (cmd_tx, cmd_rx)
    }

    #[test]
    fn test_a_command_to_a_full_channel_fails_without_waiting() {
        let (cmd_tx, cmd_rx) = full_command_channel();
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let app = BabbleBoopApp::new(app_state, Config::default(), log_rx, unused_config_file());

        let result = without_blocking(move || app.send_command(AppCommand::SetEnabled(false)));

        let error = result.unwrap_err();
        assert!(error.starts_with("Failed to send command: "), "{}", error);
        drop(cmd_rx);
    }

    #[test]
    fn test_save_with_a_full_channel_writes_the_file_and_says_it_is_not_applied() {
        use std::fs;

        let dir =
            std::env::temp_dir().join(format!("babble_boop_gui_save_full_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        let config_file = dir.join("config.toml");
        let mut file_config = Config::default();
        file_config.openai.api_key = "sk-test".to_string();
        file_config.save(&config_file).unwrap();
        let (cmd_tx, cmd_rx) = full_command_channel();
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, file_config, log_rx, config_file.clone());
        app.config_draft.translation.target_language = "German".to_string();

        let text = without_blocking(move || {
            let ctx = egui::Context::default();
            let output = run_until_idle(&ctx, &mut app);
            click(&ctx, &mut app, text_center(&output, "Save Settings"));
            painted_text(&run_until_idle(&ctx, &mut app))
        });
        let saved = Config::load(&config_file).map_err(|e| e.to_string());
        // Clean up before asserting, so a failure does not leave files behind
        fs::remove_dir_all(&dir).unwrap();

        assert!(
            text.iter().any(|line| line
                .starts_with("Settings saved to file, but not applied. Failed to send command: ")),
            "{:?}",
            text
        );
        assert_eq!(saved.unwrap().config.translation.target_language, "German");
        drop(cmd_rx);
    }

    /// The target language of each `UpdateConfig` in `commands`.
    fn sent_languages(commands: &[AppCommand]) -> Vec<String> {
        commands
            .iter()
            .filter_map(|command| match command {
                AppCommand::UpdateConfig(config) => {
                    Some(config.translation.target_language.clone())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_save_after_a_full_channel_sends_the_saved_settings_again() {
        use std::fs;

        let dir =
            std::env::temp_dir().join(format!("babble_boop_gui_save_retry_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        let config_file = dir.join("config.toml");
        let mut file_config = Config::default();
        file_config.openai.api_key = "sk-test".to_string();
        file_config.save(&config_file).unwrap();
        let (cmd_tx, mut cmd_rx) = full_command_channel();
        let (log_tx, log_rx) = tokio::sync::mpsc::channel(100);
        let app_state = Arc::new(AppState::new(cmd_tx, log_tx));
        let mut app = BabbleBoopApp::new(app_state, file_config, log_rx, config_file.clone());
        app.config_draft.translation.target_language = "German".to_string();

        let (queued, after_failure, sent_again, after_retry, sent_when_clean) =
            without_blocking(move || {
                let mut receive_all = || -> Vec<AppCommand> {
                    std::iter::from_fn(|| cmd_rx.try_recv().ok()).collect()
                };
                let ctx = egui::Context::default();
                let output = run_until_idle(&ctx, &mut app);
                click(&ctx, &mut app, text_center(&output, "Save Settings"));
                let after_failure = run_until_idle(&ctx, &mut app);
                // The loop reads the commands that filled the channel
                let queued = receive_all();
                click(&ctx, &mut app, text_center(&after_failure, "Save Settings"));
                let after_retry = run_until_idle(&ctx, &mut app);
                let sent_again = receive_all();
                click(&ctx, &mut app, text_center(&after_retry, "Save Settings"));
                run_until_idle(&ctx, &mut app);
                let sent_when_clean = receive_all();
                (
                    queued,
                    painted_text(&after_failure),
                    sent_again,
                    painted_text(&after_retry),
                    sent_when_clean,
                )
            });
        // Clean up before asserting, so a failure does not leave files behind
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(queued.len(), 32);
        assert_eq!(sent_languages(&queued), Vec::<String>::new());
        assert_eq!(sent_languages(&sent_again), ["German"]);
        assert!(
            after_failure.contains(&NOT_APPLIED.to_string()),
            "{:?}",
            after_failure
        );
        assert!(
            !after_failure.contains(&UNSAVED.to_string()),
            "{:?}",
            after_failure
        );
        assert!(
            after_retry.contains(&"Settings saved successfully".to_string()),
            "{:?}",
            after_retry
        );
        assert!(
            !after_retry.contains(&NOT_APPLIED.to_string()),
            "{:?}",
            after_retry
        );
        assert!(
            !after_retry.contains(&UNSAVED.to_string()),
            "{:?}",
            after_retry
        );
        assert_eq!(sent_languages(&sent_when_clean), Vec::<String>::new());
    }
}
