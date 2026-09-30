pub mod api_client;
pub mod app_state;
pub mod audio_playback;
pub mod audio_recording;
pub mod chatbox;
pub mod config;
pub mod data_dir;
pub mod gui;
pub mod models;
pub mod pipeline;
pub mod price_estimator;
pub mod processing_loop;
pub mod rate_limiter;
pub mod recorder;
pub mod recording_manager;
mod resample;
pub mod shutdown;
mod stream_errors;
pub mod theme;
pub mod transcription;
pub mod translation;
pub mod types;
pub mod typing_indicator;
pub mod upload_audio;

#[cfg(test)]
mod test_support;
