use crate::models::{DEFAULT_CHAT_MODEL, DEFAULT_TRANSCRIPTION_MODEL};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::fs;
use std::ops::RangeInclusive;
use std::path::Path;

// The values the settings window accepts. `Config::load` moves a number from
// the file into the same range. A value that does not fit the type of its
// field, such as a negative count, port 70000 or a decimal number in an
// integer field, still stops the load with a TOML error.
pub const PORT_RANGE: RangeInclusive<u16> = 1..=65535;
pub const DISPLAY_TIME_MS_RANGE: RangeInclusive<u64> = 1000..=30000;
pub const MAX_MESSAGE_CHUNKS_RANGE: RangeInclusive<usize> = 1..=10;
// The maximums of these two are far above the values that users set. A
// lower maximum slows an API account with a higher limit, and a lower
// max_audio_files deletes saved recordings at the next recording.
pub const REQUESTS_PER_MINUTE_RANGE: RangeInclusive<usize> = 1..=10000;
pub const MAX_AUDIO_FILES_RANGE: RangeInclusive<usize> = 1..=10000;
// Seconds of silence after the noise gate closes that end a recording. The
// setting it replaces counted audio buffers from 1 to 200, about 0.01 to 2 s
// with 10 ms buffers. The settings window shows the silence with one
// decimal, so 0.1 s is the shortest value it can show. The silence is
// recorded and uploaded, so the maximum of 10 s fills a third of a 30 s part.
pub const SILENCE_DURATION_RANGE: RangeInclusive<f32> = 0.1..=10.0;
pub const NOISE_GATE_THRESHOLD_RANGE: RangeInclusive<f32> = 0.0..=1.0;
pub const NOISE_GATE_HOLD_TIME_RANGE: RangeInclusive<f32> = 0.0..=2.0;
pub const MIN_TRANSCRIPTION_DURATION_RANGE: RangeInclusive<f32> = 0.0..=10.0;

#[derive(Deserialize, Serialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThemeMode {
    #[default]
    Dark,
    Light,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct Config {
    pub osc: OscConfig,
    pub openai: OpenAiConfig,
    pub translation: TranslationConfig,
    pub audio: AudioConfig,
    pub rate_limit: RateLimitConfig,
    #[serde(alias = "debug", default)]
    pub keep_audio_files: bool,
    #[serde(default = "default_max_audio_files")]
    pub max_audio_files: usize,
    #[serde(default)]
    pub theme: ThemeMode,
}

fn default_max_audio_files() -> usize {
    10
}

impl Default for Config {
    fn default() -> Self {
        Self {
            osc: OscConfig::default(),
            openai: OpenAiConfig::default(),
            translation: TranslationConfig::default(),
            audio: AudioConfig::default(),
            rate_limit: RateLimitConfig::default(),
            keep_audio_files: false,
            max_audio_files: 10,
            theme: ThemeMode::default(),
        }
    }
}

/// A config read from a file, and the values in the file that it does not
/// use as written.
#[derive(Debug)]
pub struct LoadedConfig {
    pub config: Config,
    pub warnings: Vec<ConfigWarning>,
}

/// A value in the config file that the program replaced with a value it
/// can use.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfigWarning {
    /// The key in the config file, for example `osc.display_time`
    pub field: &'static str,
    /// The value in the file
    pub found: String,
    /// Why the program cannot use the value in the file
    pub reason: String,
    /// The value the program uses
    pub used: String,
}

impl fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Config file: {} = {}: {}. BabbleBoop uses {}. \
            Click Save Settings to write it to the file.",
            self.field, self.found, self.reason, self.used
        )
    }
}

impl Config {
    /// Read the config file. A value that the program cannot use is
    /// replaced, and each replacement gives a warning. The file is not
    /// changed.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<LoadedConfig, Box<dyn Error>> {
        let data = fs::read_to_string(path)?;
        Ok(Self::from_toml(&data)?)
    }

    /// Parse the text of a config file, with the replacements of `load`.
    /// A key that `Config` does not have is ignored. A key that an earlier
    /// version used gives a warning.
    pub fn from_toml(text: &str) -> Result<LoadedConfig, toml::de::Error> {
        let mut config: Config = toml::from_str(text)?;
        let mut warnings = config.normalize();
        let table: toml::Table = toml::from_str(text)?;
        // Up to v0.5.0 the silence length was a number of audio buffers,
        // and the buffer length depends on the device
        let old_silence = table
            .get("audio")
            .and_then(|audio| audio.get("silence_threshold"));
        if let Some(old) = old_silence {
            warnings.push(ConfigWarning {
                field: "audio.silence_threshold",
                found: old.to_string(),
                reason: "no longer used, audio.silence_duration in seconds replaces it".to_string(),
                used: format!("audio.silence_duration = {}", config.audio.silence_duration),
            });
        }
        Ok(LoadedConfig { config, warnings })
    }

    /// Load config from path, or create a default config file if it doesn't exist.
    /// Returns the config and a boolean indicating if a new file was created.
    pub fn load_or_create<P: AsRef<Path>>(path: P) -> Result<(LoadedConfig, bool), Box<dyn Error>> {
        let path = path.as_ref();
        if path.exists() {
            Ok((Self::load(path)?, false))
        } else {
            let config = Self::default();
            config.save(path)?;
            let loaded = LoadedConfig {
                config,
                warnings: Vec::new(),
            };
            Ok((loaded, true))
        }
    }

    /// Replace each value that the program cannot use, and return a
    /// warning for each replacement. A value out of range moves to the
    /// nearest end of the range. A number that is not finite, a blank text
    /// and port 0 become the default. A blank API key stays blank, and
    /// -0.0 stays as it is. A value that does not fit the type of its field
    /// does not get here: the TOML parser refuses it.
    fn normalize(&mut self) -> Vec<ConfigWarning> {
        let default = Config::default();
        let mut w = Warnings::default();

        let osc = &mut self.osc;
        w.text("osc.address", &mut osc.address, &default.osc.address);
        // Port 1 is the nearest, but no OSC app uses it
        let port_reason = "not a port that can be used";
        if !PORT_RANGE.contains(&osc.input_port) {
            w.replace(
                "osc.input_port",
                &mut osc.input_port,
                default.osc.input_port,
                port_reason,
            );
        }
        if !PORT_RANGE.contains(&osc.output_port) {
            w.replace(
                "osc.output_port",
                &mut osc.output_port,
                default.osc.output_port,
                port_reason,
            );
        }
        // The chatbox messages would go to the input port, not to VRChat.
        // Change the input port back to its default, or the output port if
        // the input port has its default.
        if osc.input_port == osc.output_port {
            if osc.input_port != default.osc.input_port {
                let reason = "the same as the osc.output_port in use";
                w.replace(
                    "osc.input_port",
                    &mut osc.input_port,
                    default.osc.input_port,
                    reason,
                );
            } else {
                let reason = "the same as the osc.input_port in use";
                w.replace(
                    "osc.output_port",
                    &mut osc.output_port,
                    default.osc.output_port,
                    reason,
                );
            }
        }
        w.clamp(
            "osc.display_time",
            &mut osc.display_time,
            DISPLAY_TIME_MS_RANGE,
        );
        w.clamp(
            "osc.max_message_chunks",
            &mut osc.max_message_chunks,
            MAX_MESSAGE_CHUNKS_RANGE,
        );

        let openai = &mut self.openai;
        w.secret("openai.api_key", &mut openai.api_key);
        w.text("openai.model", &mut openai.model, &default.openai.model);
        w.text(
            "openai.transcription_model",
            &mut openai.transcription_model,
            &default.openai.transcription_model,
        );
        w.text(
            "translation.target_language",
            &mut self.translation.target_language,
            &default.translation.target_language,
        );

        let audio = &mut self.audio;
        w.number(
            "audio.silence_duration",
            &mut audio.silence_duration,
            SILENCE_DURATION_RANGE,
            default.audio.silence_duration,
        );
        w.number(
            "audio.noise_gate_threshold",
            &mut audio.noise_gate_threshold,
            NOISE_GATE_THRESHOLD_RANGE,
            default.audio.noise_gate_threshold,
        );
        w.number(
            "audio.noise_gate_hold_time",
            &mut audio.noise_gate_hold_time,
            NOISE_GATE_HOLD_TIME_RANGE,
            default.audio.noise_gate_hold_time,
        );
        w.number(
            "audio.min_transcription_duration",
            &mut audio.min_transcription_duration,
            MIN_TRANSCRIPTION_DURATION_RANGE,
            default.audio.min_transcription_duration,
        );

        w.clamp(
            "rate_limit.requests_per_minute",
            &mut self.rate_limit.requests_per_minute,
            REQUESTS_PER_MINUTE_RANGE,
        );
        w.clamp(
            "max_audio_files",
            &mut self.max_audio_files,
            MAX_AUDIO_FILES_RANGE,
        );
        w.0
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let data = toml::to_string_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }
}

/// The warnings of `Config::normalize`, one for each value it replaces.
#[derive(Default)]
struct Warnings(Vec<ConfigWarning>);

impl Warnings {
    fn add(&mut self, field: &'static str, found: String, reason: &str, used: String) {
        self.0.push(ConfigWarning {
            field,
            found,
            reason: reason.to_string(),
            used,
        });
    }

    /// Set `value` to `used` and add a warning.
    fn replace<T: fmt::Display>(
        &mut self,
        field: &'static str,
        value: &mut T,
        used: T,
        reason: &str,
    ) {
        self.add(field, value.to_string(), reason, used.to_string());
        *value = used;
    }

    /// Move `value` to the nearest end of `range` if it is outside it.
    fn clamp<T: PartialOrd + Copy + fmt::Display>(
        &mut self,
        field: &'static str,
        value: &mut T,
        range: RangeInclusive<T>,
    ) {
        let (start, end) = (*range.start(), *range.end());
        if *value < start {
            self.replace(field, value, start, &format!("less than {}", start));
        } else if *value > end {
            self.replace(field, value, end, &format!("more than {}", end));
        }
    }

    /// `clamp`, but a value that is not a finite number becomes `default`.
    /// NaN is neither less nor more than any value, so `clamp` keeps it.
    fn number(
        &mut self,
        field: &'static str,
        value: &mut f32,
        range: RangeInclusive<f32>,
        default: f32,
    ) {
        if value.is_finite() {
            self.clamp(field, value, range);
        } else {
            self.replace(field, value, default, "not a finite number");
        }
    }

    /// Remove spaces and line breaks around `value`. A text that is then
    /// blank becomes `default`.
    fn text(&mut self, field: &'static str, value: &mut String, default: &str) {
        let trimmed = value.trim();
        let (used, reason) = if trimmed.is_empty() {
            (default, "blank")
        } else if trimmed.len() != value.len() {
            (trimmed, "spaces or line breaks around the text")
        } else {
            return;
        };
        let used = used.to_string();
        self.add(field, format!("{:?}", value), reason, format!("{:?}", used));
        *value = used;
    }

    /// Remove spaces and line breaks around `value`, which the warning does
    /// not show. A blank value stays blank.
    fn secret(&mut self, field: &'static str, value: &mut String) {
        let trimmed = value.trim();
        if trimmed.len() != value.len() {
            let used = if trimmed.is_empty() {
                "a blank text"
            } else {
                "the text without them"
            };
            *value = trimmed.to_string();
            let reason = "spaces or line breaks around the text";
            self.add(field, "(hidden)".to_string(), reason, used.to_string());
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct OscConfig {
    pub address: String,
    pub input_port: u16,
    pub output_port: u16,
    pub max_message_chunks: usize,
    pub display_time: u64,
}

impl Default for OscConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            input_port: 9001,
            output_port: 9000,
            max_message_chunks: 9,
            display_time: 3000,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_transcription_model")]
    pub transcription_model: String,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            model: DEFAULT_CHAT_MODEL.name.to_string(),
            transcription_model: default_transcription_model(),
        }
    }
}

fn default_transcription_model() -> String {
    DEFAULT_TRANSCRIPTION_MODEL.name.to_string()
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct TranslationConfig {
    pub target_language: String,
    pub include_original_message: bool,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            target_language: "Japanese".to_string(),
            include_original_message: false,
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct AudioConfig {
    /// Seconds of input with the noise gate closed that end a recording
    #[serde(default = "default_silence_duration")]
    pub silence_duration: f32,
    pub noise_gate_threshold: f32,
    pub noise_gate_hold_time: f32,
    pub min_transcription_duration: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            silence_duration: default_silence_duration(),
            noise_gate_threshold: 0.3,
            noise_gate_hold_time: 0.20,
            min_transcription_duration: 1.0,
        }
    }
}

fn default_silence_duration() -> f32 {
    1.0
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct RateLimitConfig {
    pub requests_per_minute: usize,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 50,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rate_limiter::RateLimiter;
    use crate::recording_manager::RecordingManager;

    /// Logger whose entries nobody reads.
    fn test_logger() -> crate::app_state::Logger {
        let (log_tx, _log_rx) = tokio::sync::mpsc::channel(10);
        crate::app_state::Logger::new(log_tx, Default::default())
    }

    /// `config.toml.example` of v0.5.0, the last release.
    const V0_5_0: &str = r#"
keep_audio_files = false  # if true, saves audio recordings to disk for troubleshooting
max_audio_files = 10      # maximum number of audio files to keep (oldest are deleted)
[osc]
address = "127.0.0.1"
input_port = 9001       # VRChat's output port
output_port = 9000      # VRChat's input port for chatbox
display_time = 3000     # time to display messages in milliseconds
max_message_chunks = 9  # large messages are split into chunks, this is the max it will split

[openai]
api_key = "YOUR API KEY"
model = "gpt-4o-mini"
transcription_model = "whisper-1"  # whisper-1, gpt-4o-transcribe, or gpt-4o-mini-transcribe

[translation]
target_language = "Japanese"
include_original_message = false

[audio]
silence_threshold = 100           # determines the time for the silence detection
noise_gate_threshold = 0.3        # adjust based on your microphone and environment
noise_gate_hold_time = 0.20       # adjust based on preference
min_transcription_duration = 1.0  # Minimum duration in seconds for transcription

[rate_limit]
requests_per_minute = 50          # adjust based on your API limits, it should continue to record even while waiting
"#;

    const EXAMPLE: &str = include_str!("../config.toml.example");

    /// The example config with the line of `key` replaced by `key = value`.
    fn example_with(key: &str, value: &str) -> String {
        let mut replaced = false;
        let lines: Vec<String> = EXAMPLE
            .lines()
            .map(|line| {
                if line.split('=').next().map(str::trim) == Some(key) {
                    replaced = true;
                    format!("{} = {}", key, value)
                } else {
                    line.to_string()
                }
            })
            .collect();
        assert!(replaced, "no line for {}", key);
        lines.join("\n")
    }

    fn example_config() -> Config {
        toml::from_str(EXAMPLE).unwrap()
    }

    /// The warning for `audio.silence_threshold`, the setting of v0.5.0
    /// that `audio.silence_duration` replaces.
    fn old_silence_warning(found: &str, used: &str) -> ConfigWarning {
        ConfigWarning {
            field: "audio.silence_threshold",
            found: found.to_string(),
            reason: "no longer used, audio.silence_duration in seconds replaces it".to_string(),
            used: format!("audio.silence_duration = {}", used),
        }
    }

    #[test]
    fn v0_5_0_config_loads_with_a_warning_for_the_old_silence_setting() {
        let loaded = Config::from_toml(V0_5_0).unwrap();
        assert_eq!(loaded.warnings, [old_silence_warning("100", "1")]);
        assert_eq!(loaded.config.audio.silence_duration, 1.0);
        // Every other value as in the file
        let mut expected: Config = toml::from_str(V0_5_0).unwrap();
        expected.audio.silence_duration = 1.0;
        assert_eq!(loaded.config, expected);
    }

    #[test]
    fn old_silence_setting_warning_names_the_setting_in_use() {
        let loaded = Config::from_toml(V0_5_0).unwrap();
        assert_eq!(
            loaded.warnings[0].to_string(),
            "Config file: audio.silence_threshold = 100: no longer used, \
            audio.silence_duration in seconds replaces it. BabbleBoop uses \
            audio.silence_duration = 1. Click Save Settings to write it to the file."
        );
    }

    #[test]
    fn both_silence_settings_use_the_new_one_with_a_warning() {
        let text = example_with("silence_duration", "2.5")
            .replace("[audio]", "[audio]\nsilence_threshold = 40");
        let loaded = Config::from_toml(&text).unwrap();
        assert_eq!(loaded.config.audio.silence_duration, 2.5);
        assert_eq!(loaded.warnings, [old_silence_warning("40", "2.5")]);
    }

    #[test]
    fn old_silence_setting_warning_shows_the_replaced_new_value() {
        let text = example_with("silence_duration", "nan")
            .replace("[audio]", "[audio]\nsilence_threshold = 40");
        let loaded = Config::from_toml(&text).unwrap();
        assert_eq!(loaded.config.audio.silence_duration, 1.0);
        let fields: Vec<&str> = loaded.warnings.iter().map(|w| w.field).collect();
        assert_eq!(
            fields,
            ["audio.silence_duration", "audio.silence_threshold"]
        );
        assert_eq!(loaded.warnings[1], old_silence_warning("40", "1"));
    }

    #[test]
    fn saved_config_does_not_keep_the_old_silence_setting() {
        let loaded = Config::from_toml(V0_5_0).unwrap();
        let saved = toml::to_string(&loaded.config).unwrap();
        let reloaded = Config::from_toml(&saved).unwrap();
        assert_eq!(reloaded.warnings, []);
        assert_eq!(reloaded.config, loaded.config);
    }

    #[test]
    fn example_and_default_configs_load_without_warnings() {
        let example = Config::from_toml(EXAMPLE).unwrap();
        assert_eq!(example.warnings, []);
        assert_eq!(example.config.audio.silence_duration, 1.0);

        let default = Config::default();
        let loaded = Config::from_toml(&toml::to_string(&default).unwrap()).unwrap();
        assert_eq!(loaded.warnings, []);
        assert_eq!(loaded.config, default);
    }

    /// One bad value in the file: the key, the value, and the setting the
    /// program must use instead. Every other setting must stay as in the
    /// file, and there must be exactly one warning, for this key.
    struct BadValue {
        field: &'static str,
        key: &'static str,
        value: &'static str,
        expect: fn(&mut Config),
    }

    const BAD_VALUES: &[BadValue] = &[
        // A NaN hold time keeps the gate open forever: no time is greater
        BadValue {
            field: "audio.noise_gate_hold_time",
            key: "noise_gate_hold_time",
            value: "nan",
            expect: |c| c.audio.noise_gate_hold_time = 0.2,
        },
        BadValue {
            field: "audio.noise_gate_hold_time",
            key: "noise_gate_hold_time",
            value: "inf",
            expect: |c| c.audio.noise_gate_hold_time = 0.2,
        },
        BadValue {
            field: "audio.noise_gate_hold_time",
            key: "noise_gate_hold_time",
            value: "5.0",
            expect: |c| c.audio.noise_gate_hold_time = 2.0,
        },
        // A NaN threshold never opens the gate: no level is greater
        BadValue {
            field: "audio.noise_gate_threshold",
            key: "noise_gate_threshold",
            value: "nan",
            expect: |c| c.audio.noise_gate_threshold = 0.3,
        },
        BadValue {
            field: "audio.noise_gate_threshold",
            key: "noise_gate_threshold",
            value: "-0.5",
            expect: |c| c.audio.noise_gate_threshold = 0.0,
        },
        BadValue {
            field: "audio.min_transcription_duration",
            key: "min_transcription_duration",
            value: "-1.0",
            expect: |c| c.audio.min_transcription_duration = 0.0,
        },
        BadValue {
            field: "audio.min_transcription_duration",
            key: "min_transcription_duration",
            value: "nan",
            expect: |c| c.audio.min_transcription_duration = 1.0,
        },
        BadValue {
            field: "audio.min_transcription_duration",
            key: "min_transcription_duration",
            value: "-inf",
            expect: |c| c.audio.min_transcription_duration = 1.0,
        },
        BadValue {
            field: "audio.min_transcription_duration",
            key: "min_transcription_duration",
            value: "60.0",
            expect: |c| c.audio.min_transcription_duration = 10.0,
        },
        // 0 stops every recording at the first quiet buffer
        BadValue {
            field: "audio.silence_duration",
            key: "silence_duration",
            value: "0.0",
            expect: |c| c.audio.silence_duration = 0.1,
        },
        BadValue {
            field: "audio.silence_duration",
            key: "silence_duration",
            value: "-2.0",
            expect: |c| c.audio.silence_duration = 0.1,
        },
        BadValue {
            field: "audio.silence_duration",
            key: "silence_duration",
            value: "60.0",
            expect: |c| c.audio.silence_duration = 10.0,
        },
        BadValue {
            field: "audio.silence_duration",
            key: "silence_duration",
            value: "nan",
            expect: |c| c.audio.silence_duration = 1.0,
        },
        BadValue {
            field: "audio.silence_duration",
            key: "silence_duration",
            value: "inf",
            expect: |c| c.audio.silence_duration = 1.0,
        },
        // 0 deletes every saved recording
        BadValue {
            field: "max_audio_files",
            key: "max_audio_files",
            value: "0",
            expect: |c| c.max_audio_files = 1,
        },
        BadValue {
            field: "max_audio_files",
            key: "max_audio_files",
            value: "20000",
            expect: |c| c.max_audio_files = 10000,
        },
        // The chatbox pauses this long after each part of a message
        BadValue {
            field: "osc.display_time",
            key: "display_time",
            value: "9223372036854775807",
            expect: |c| c.osc.display_time = 30000,
        },
        BadValue {
            field: "osc.display_time",
            key: "display_time",
            value: "0",
            expect: |c| c.osc.display_time = 1000,
        },
        // 0 sends nothing to the chatbox
        BadValue {
            field: "osc.max_message_chunks",
            key: "max_message_chunks",
            value: "0",
            expect: |c| c.osc.max_message_chunks = 1,
        },
        BadValue {
            field: "osc.max_message_chunks",
            key: "max_message_chunks",
            value: "50",
            expect: |c| c.osc.max_message_chunks = 10,
        },
        // Port 0 binds a random port that VRChat does not send to
        BadValue {
            field: "osc.input_port",
            key: "input_port",
            value: "0",
            expect: |c| c.osc.input_port = 9001,
        },
        BadValue {
            field: "osc.output_port",
            key: "output_port",
            value: "0",
            expect: |c| c.osc.output_port = 9000,
        },
        // The chatbox messages would go to the input port
        BadValue {
            field: "osc.output_port",
            key: "output_port",
            value: "9001",
            expect: |c| c.osc.output_port = 9000,
        },
        BadValue {
            field: "osc.input_port",
            key: "input_port",
            value: "9000",
            expect: |c| c.osc.input_port = 9001,
        },
        // 0 lets only one request through at the end of each minute
        BadValue {
            field: "rate_limit.requests_per_minute",
            key: "requests_per_minute",
            value: "0",
            expect: |c| c.rate_limit.requests_per_minute = 1,
        },
        BadValue {
            field: "rate_limit.requests_per_minute",
            key: "requests_per_minute",
            value: "20000",
            expect: |c| c.rate_limit.requests_per_minute = 10000,
        },
        BadValue {
            field: "osc.address",
            key: "address",
            value: r#""""#,
            expect: |c| c.osc.address = "127.0.0.1".to_string(),
        },
        BadValue {
            field: "osc.address",
            key: "address",
            value: r#"" \t ""#,
            expect: |c| c.osc.address = "127.0.0.1".to_string(),
        },
        BadValue {
            field: "osc.address",
            key: "address",
            value: r#"" 192.168.1.5\n""#,
            expect: |c| c.osc.address = "192.168.1.5".to_string(),
        },
        BadValue {
            field: "openai.api_key",
            key: "api_key",
            value: r#""sk-test\r\n""#,
            expect: |c| c.openai.api_key = "sk-test".to_string(),
        },
        BadValue {
            field: "openai.model",
            key: "model",
            value: r#"" gpt-4o-mini ""#,
            expect: |c| c.openai.model = "gpt-4o-mini".to_string(),
        },
        BadValue {
            field: "openai.model",
            key: "model",
            value: r#""""#,
            expect: |c| c.openai.model = Config::default().openai.model,
        },
        BadValue {
            field: "openai.transcription_model",
            key: "transcription_model",
            value: r#"" ""#,
            expect: |c| c.openai.transcription_model = Config::default().openai.transcription_model,
        },
        BadValue {
            field: "translation.target_language",
            key: "target_language",
            value: r#""Japanese\n""#,
            expect: |c| c.translation.target_language = "Japanese".to_string(),
        },
        BadValue {
            field: "translation.target_language",
            key: "target_language",
            value: r#""""#,
            expect: |c| {
                c.translation.target_language = Config::default().translation.target_language
            },
        },
    ];

    #[test]
    fn bad_values_load_as_usable_values_with_a_warning() {
        let mut failures = Vec::new();
        for bad in BAD_VALUES {
            let loaded = Config::from_toml(&example_with(bad.key, bad.value)).unwrap();
            let mut expected = example_config();
            (bad.expect)(&mut expected);
            let fields: Vec<&str> = loaded.warnings.iter().map(|w| w.field).collect();
            if loaded.config != expected || fields != [bad.field] {
                failures.push(format!(
                    "{} = {}: warnings for {:?}, config {:?}",
                    bad.key, bad.value, fields, loaded.config
                ));
            }
        }
        assert!(failures.is_empty(), "{:#?}", failures);
    }

    /// Values above the limits of earlier versions of the settings window
    /// (100 recordings, 120 requests per minute), as a user can have set
    /// them in the file.
    fn example_with_high_limits() -> LoadedConfig {
        let text = example_with("max_audio_files", "500")
            .replace("keep_audio_files = false", "keep_audio_files = true")
            .replace("requests_per_minute = 50", "requests_per_minute = 500");
        Config::from_toml(&text).unwrap()
    }

    #[test]
    fn high_limits_load_unchanged() {
        let loaded = example_with_high_limits();
        assert_eq!(loaded.warnings, []);
        assert_eq!(loaded.config.max_audio_files, 500);
        assert_eq!(loaded.config.rate_limit.requests_per_minute, 500);
    }

    #[tokio::test]
    async fn more_than_100_saved_recordings_are_kept() {
        let loaded = example_with_high_limits();
        let dir = std::env::temp_dir().join(format!(
            "babble_boop_more_than_100_recordings_{}",
            std::process::id()
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        for i in 0..150 {
            fs::write(dir.join(format!("{}_old.wav", i)), b"wav").unwrap();
        }

        let mut manager = RecordingManager::new(dir.clone(), loaded.config.max_audio_files);
        manager
            .save_recording(b"wav".to_vec(), "hello", &test_logger())
            .await;
        let kept = fs::read_dir(&dir).map(|entries| entries.count());
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(kept.unwrap(), 151);
    }

    #[tokio::test(start_paused = true)]
    async fn more_than_120_requests_per_minute_do_not_wait() {
        let loaded = example_with_high_limits();
        let mut limiter = RateLimiter::new(loaded.config.rate_limit.requests_per_minute);

        let start = tokio::time::Instant::now();
        for _ in 0..500 {
            limiter.wait().await;
        }

        // The paused clock moves only when the limiter sleeps
        assert_eq!(start.elapsed(), std::time::Duration::ZERO);
    }

    #[test]
    fn warning_names_the_field_the_value_found_and_the_value_used() {
        let loaded = Config::from_toml(&example_with("display_time", "999999")).unwrap();
        let message = loaded.warnings[0].to_string();
        for part in ["osc.display_time", "999999", "30000"] {
            assert!(message.contains(part), "{:?} not in {:?}", part, message);
        }
    }

    #[test]
    fn api_key_warning_does_not_show_the_key() {
        let loaded = Config::from_toml(&example_with("api_key", r#""sk-secret\n""#)).unwrap();
        assert_eq!(loaded.warnings.len(), 1);
        let message = loaded.warnings[0].to_string();
        assert!(!message.contains("sk-secret"), "{:?}", message);
    }

    #[test]
    fn api_key_of_only_spaces_warning_says_the_key_is_blank() {
        let loaded = Config::from_toml(&example_with("api_key", r#""  \n""#)).unwrap();
        assert_eq!(loaded.config.openai.api_key, "");
        let used: Vec<&str> = loaded.warnings.iter().map(|w| w.used.as_str()).collect();
        assert_eq!(used, ["a blank text"]);
    }

    #[tokio::test]
    async fn max_audio_files_zero_keeps_the_saved_recording() {
        let text = example_with("max_audio_files", "0")
            .replace("keep_audio_files = false", "keep_audio_files = true");
        let loaded = Config::from_toml(&text).unwrap();
        assert!(loaded.config.keep_audio_files);

        let dir = std::env::temp_dir().join(format!(
            "babble_boop_max_audio_files_zero_{}",
            std::process::id()
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        let mut manager = RecordingManager::new(dir.clone(), loaded.config.max_audio_files);
        manager
            .save_recording(b"wav".to_vec(), "hello", &test_logger())
            .await;
        let kept = fs::read_dir(&dir).map(|entries| entries.count());
        fs::remove_dir_all(&dir).unwrap();

        assert_eq!(kept.unwrap(), 1, "the recording just saved was deleted");
    }
}
