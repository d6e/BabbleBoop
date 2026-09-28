use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

pub struct RecordingManager {
    recordings_dir: PathBuf,
    max_recordings: usize,
}

impl RecordingManager {
    pub fn new(recordings_dir: PathBuf, max_recordings: usize) -> Self {
        RecordingManager {
            recordings_dir,
            max_recordings,
        }
    }

    pub async fn save_recording(
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

#[cfg(all(test, unix))]
mod tests {
    use super::RecordingManager;
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    #[tokio::test]
    async fn test_cleanup_skips_entries_without_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "babble_boop_cleanup_metadata_{}",
            std::process::id()
        ));
        // A previous run that panicked can leave the directory without permissions
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).ok();
        fs::remove_dir_all(&dir).ok();
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
        fs::remove_dir_all(&dir).ok();

        let Some(result) = result else {
            // Running as root: permissions do not block metadata, so the
            // failure cannot be provoked here.
            eprintln!("skipped: directory permissions do not block metadata");
            return;
        };
        assert!(result.is_ok(), "cleanup failed: {:?}", result.err());
        assert!(files_kept, "a recording with no readable age was deleted");
    }
}
