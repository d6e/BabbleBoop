use crate::config::ThemeMode;
use eframe::egui::{self, Color32};

/// Semantic color palette for the application
#[derive(Clone, Copy)]
pub struct AppColors {
    // Status colors
    pub success: Color32,
    pub error: Color32,
    pub warning: Color32,
    pub info: Color32,

    // Log colors
    pub log_timestamp: Color32,
    pub log_info: Color32,
    pub log_success: Color32,
    pub log_error: Color32,

    // Panel colors
    pub panel_background: Color32,
    pub text_muted: Color32,
    pub text_disabled: Color32,

    // Toggle switch colors
    pub toggle_on: Color32,
    pub toggle_off: Color32,
    pub toggle_knob: Color32,

    // Audio meter colors
    pub meter_background: Color32,
    pub meter_low: Color32,
    pub meter_medium: Color32,
    pub meter_high: Color32,
    pub meter_threshold: Color32,
    pub meter_threshold_active: Color32,

    // Status indicator colors
    pub enabled_text: Color32,
    pub disabled_text: Color32,
    pub cost_text: Color32,
    pub unsaved_indicator: Color32,
}

impl AppColors {
    pub fn dark() -> Self {
        Self {
            success: Color32::from_rgb(100, 200, 100),
            error: Color32::from_rgb(220, 80, 80),
            warning: Color32::from_rgb(255, 180, 0),
            info: Color32::from_rgb(150, 150, 220),

            log_timestamp: Color32::from_rgb(120, 120, 120),
            log_info: Color32::from_rgb(180, 180, 180),
            log_success: Color32::from_rgb(120, 200, 120),
            log_error: Color32::from_rgb(220, 100, 100),

            panel_background: Color32::from_rgb(30, 30, 30),
            text_muted: Color32::GRAY,
            text_disabled: Color32::from_rgb(140, 140, 140),

            toggle_on: Color32::from_rgb(100, 160, 100),
            toggle_off: Color32::from_rgb(60, 60, 60),
            toggle_knob: Color32::WHITE,

            meter_background: Color32::from_gray(40),
            meter_low: Color32::from_rgb(60, 180, 60),
            meter_medium: Color32::from_rgb(220, 180, 60),
            meter_high: Color32::from_rgb(220, 60, 60),
            meter_threshold: Color32::WHITE,
            meter_threshold_active: Color32::YELLOW,

            enabled_text: Color32::from_rgb(80, 160, 80),
            disabled_text: Color32::from_rgb(140, 140, 140),
            cost_text: Color32::from_rgb(180, 180, 100),
            unsaved_indicator: Color32::from_rgb(255, 180, 0),
        }
    }

    pub fn light() -> Self {
        Self {
            success: Color32::from_rgb(40, 160, 40),
            error: Color32::from_rgb(200, 50, 50),
            warning: Color32::from_rgb(200, 130, 0),
            info: Color32::from_rgb(80, 80, 180),

            log_timestamp: Color32::from_rgb(100, 100, 100),
            log_info: Color32::from_rgb(60, 60, 60),
            log_success: Color32::from_rgb(40, 140, 40),
            log_error: Color32::from_rgb(180, 50, 50),

            panel_background: Color32::from_rgb(235, 235, 235),
            text_muted: Color32::from_rgb(100, 100, 100),
            text_disabled: Color32::from_rgb(160, 160, 160),

            toggle_on: Color32::from_rgb(60, 140, 60),
            toggle_off: Color32::from_rgb(180, 180, 180),
            toggle_knob: Color32::WHITE,

            meter_background: Color32::from_rgb(200, 200, 200),
            meter_low: Color32::from_rgb(40, 160, 40),
            meter_medium: Color32::from_rgb(200, 160, 40),
            meter_high: Color32::from_rgb(200, 40, 40),
            meter_threshold: Color32::from_rgb(40, 40, 40),
            meter_threshold_active: Color32::from_rgb(200, 160, 0),

            enabled_text: Color32::from_rgb(40, 120, 40),
            disabled_text: Color32::from_rgb(120, 120, 120),
            cost_text: Color32::from_rgb(140, 140, 60),
            unsaved_indicator: Color32::from_rgb(180, 100, 0),
        }
    }
}

/// Get the appropriate egui Visuals for the given theme mode
pub fn get_visuals(mode: ThemeMode) -> egui::Visuals {
    match mode {
        ThemeMode::Dark => egui::Visuals::dark(),
        ThemeMode::Light => egui::Visuals::light(),
    }
}

/// Get the application color palette for the given theme mode
pub fn get_colors(mode: ThemeMode) -> AppColors {
    match mode {
        ThemeMode::Dark => AppColors::dark(),
        ThemeMode::Light => AppColors::light(),
    }
}
