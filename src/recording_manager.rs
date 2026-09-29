use crate::app_state::{FailureLog, Logger};
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

pub struct RecordingManager {
    recordings_dir: PathBuf,
    max_recordings: usize,
    /// A failed debug recording save must not stop the translation, so the
    /// caller does not see the error. It goes to the activity log instead,
    /// once until a save works again or fails differently.
    save_failure: FailureLog,
}

impl RecordingManager {
    pub fn new(recordings_dir: PathBuf, max_recordings: usize) -> Self {
        RecordingManager {
            recordings_dir,
            max_recordings,
            save_failure: FailureLog::default(),
        }
    }

    /// Save a debug recording. A failure goes to the activity log, once
    /// until a save works again or fails differently; it does not stop the
    /// caller, which goes on to translate the transcription either way.
    pub async fn save_recording(
        &mut self,
        audio_data: Vec<u8>,
        transcription: &str,
        logger: &Logger,
    ) {
        match self.try_save_recording(audio_data, transcription).await {
            Ok(()) => {
                if self.save_failure.succeeded() {
                    logger.info(format!(
                        "Saved a debug recording to {} again",
                        self.recordings_dir.display()
                    ));
                }
            }
            Err(e) => self.save_failure.failed(
                logger,
                format!(
                    "Cannot save the debug recording to {}: {}",
                    self.recordings_dir.display(),
                    e
                ),
            ),
        }
    }

    async fn try_save_recording(
        &self,
        audio_data: Vec<u8>,
        transcription: &str,
    ) -> Result<(), Box<dyn Error>> {
        fs::create_dir_all(&self.recordings_dir)?;

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        // Take characters, not bytes, so the cut cannot fall inside a multibyte character
        let prefix: String = transcription.chars().take(50).collect();
        let slugified_transcription = self.slugify(&prefix);
        let filename = format!("{}_{}.wav", timestamp, slugified_transcription);
        let file_path = self.recordings_dir.join(filename);

        let mut file = File::create(&file_path).await?;
        file.write_all(&audio_data).await?;

        self.cleanup_old_recordings().await?;

        Ok(())
    }

    async fn cleanup_old_recordings(&self) -> Result<(), Box<dyn Error>> {
        // Skip entries whose modification time cannot be read (for example, a file
        // removed after read_dir listed it). They cannot be ordered, so they are kept.
        let mut entries: Vec<(SystemTime, PathBuf)> = fs::read_dir(&self.recordings_dir)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let modified = entry.metadata().and_then(|m| m.modified()).ok()?;
                Some((modified, entry.path()))
            })
            .collect();

        entries.sort_by_key(|(modified, _)| *modified);

        if entries.len() > self.max_recordings {
            for (_, path) in entries.iter().take(entries.len() - self.max_recordings) {
                fs::remove_file(path)?;
            }
        }

        Ok(())
    }

    fn slugify(&self, text: &str) -> String {
        text.chars()
            .filter_map(|c| {
                if c.is_alphanumeric() {
                    Some(c.to_ascii_lowercase())
                } else if c.is_whitespace() {
                    Some('-')
                } else {
                    None
                }
            })
            .collect::<String>()
            .split('-')
            .filter(|&s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }
}

#[cfg(test)]
mod tests {
    use super::RecordingManager;
    use crate::app_state::LogLevel;
    use crate::test_support::LogCapture;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    /// A failed debug recording save is logged once per distinct error and
    /// does not stop the caller; a save that works again is logged too.
    #[tokio::test]
    #[cfg(unix)]
    async fn test_a_failed_debug_recording_save_is_logged_once_per_distinct_error() {
        let dir = std::env::temp_dir().join(format!(
            "babble_boop_recording_save_failure_{}",
            std::process::id()
        ));
        if dir.exists() {
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        // Read and search but no write permission: the directory exists, so
        // create_dir_all succeeds, but creating a file in it fails.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let probe = dir.join("probe");
        let write_blocked = fs::write(&probe, b"x").is_err();
        _ = fs::remove_file(&probe);

        if !write_blocked {
            // Running as root: permissions do not block the write, so the
            // failure cannot be provoked here.
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
            fs::remove_dir_all(&dir).unwrap();
            eprintln!("skipped: directory permissions do not block writes");
            return;
        }

        let mut log = LogCapture::new();
        let logger = log.logger();
        let mut manager = RecordingManager::new(dir.clone(), 10);

        // The translation is not aborted: save_recording has no error to
        // propagate, so a caller that drives it (Pipeline::process) always
        // reaches the code after it.
        manager.save_recording(vec![0u8; 4], "first", &logger).await;
        let entries = log.entries();
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert_eq!(entries[0].0, LogLevel::Error, "{:?}", entries);
        assert!(
            entries[0].1.contains(&dir.display().to_string()),
            "{:?}",
            entries
        );

        // A second utterance fails the same way: no second log entry.
        manager
            .save_recording(vec![0u8; 4], "second", &logger)
            .await;
        assert_eq!(log.entries(), []);

        // The folder becomes writable again: the save works and a recovery
        // line is logged.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
        manager.save_recording(vec![0u8; 4], "third", &logger).await;
        let entries = log.entries();
        assert_eq!(entries.len(), 1, "{:?}", entries);
        assert_eq!(entries[0].0, LogLevel::Info, "{:?}", entries);

        // A save that works again logs nothing more.
        manager
            .save_recording(vec![0u8; 4], "fourth", &logger)
            .await;
        assert_eq!(log.entries(), []);

        fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_cleanup_skips_entries_without_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "babble_boop_cleanup_metadata_{}",
            std::process::id()
        ));
        // A previous run that panicked can leave the directory without permissions
        if dir.exists() {
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        // Two entries, so the sort has to compare them
        let files = [dir.join("1_old.wav"), dir.join("2_new.wav")];
        for file in &files {
            fs::write(file, b"wav").unwrap();
        }

        // Read but no search permission: read_dir lists the entry, but its
        // metadata cannot be read.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o400)).unwrap();
        let metadata_blocked = fs::metadata(&files[0]).is_err();
        let result = if metadata_blocked {
            Some(
                RecordingManager::new(dir.clone(), 0)
                    .cleanup_old_recordings()
                    .await,
            )
        } else {
            None
        };

        // Restore permissions and check the file before cleaning up
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
        let files_kept = files.iter().all(|file| file.exists());
        fs::remove_dir_all(&dir).unwrap();

        let Some(result) = result else {
            // Running as root: permissions do not block metadata, so the
            // failure cannot be provoked here.
            eprintln!("skipped: directory permissions do not block metadata");
            return;
        };
        assert!(result.is_ok(), "cleanup failed: {:?}", result.err());
        assert!(files_kept, "a recording with no readable age was deleted");
    }

    // ===========================================================================
    // Test: Recording file names cut the transcription by characters
    // ===========================================================================

    /// Saves one recording with `transcription` into a fresh directory and
    /// returns the file name without the timestamp prefix.
    async fn saved_recording_name(test_name: &str, transcription: &str) -> String {
        use crate::test_support::{silent_logger, TempDir};

        let dir = TempDir::new(test_name);

        RecordingManager::new(dir.path().to_path_buf(), 10)
            .save_recording(vec![0u8; 4], transcription, &silent_logger())
            .await;
        let names: Vec<String> = fs::read_dir(dir.path())
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();

        assert_eq!(names.len(), 1, "expected one recording, found {:?}", names);
        let (_timestamp, rest) = names[0]
            .split_once('_')
            .expect("file name has a timestamp prefix");
        rest.to_string()
    }

    #[tokio::test]
    async fn test_recording_name_ascii() {
        // The first 50 characters are kept, so " cut" is dropped.
        let transcription = "Hello, World! This is a test of the recording name cut";
        assert_eq!(
            saved_recording_name("ascii", transcription).await,
            "hello-world-this-is-a-test-of-the-recording-name.wav"
        );
    }

    #[tokio::test]
    async fn test_recording_name_multibyte() {
        // Byte 50 is inside the two byte 'é'.
        let accented = format!("{}é{}", "a".repeat(49), "b".repeat(20));
        assert_eq!(
            saved_recording_name("accented", &accented).await,
            format!("{}é.wav", "a".repeat(49))
        );

        // 60 characters of three bytes each; byte 50 is inside the 17th.
        let cjk = "日本語".repeat(20);
        assert_eq!(
            saved_recording_name("cjk", &cjk).await,
            format!("{}.wav", cjk.chars().take(50).collect::<String>())
        );
    }
}
