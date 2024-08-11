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
        let slugified_transcription =
            self.slugify(&transcription[..std::cmp::min(50, transcription.len())]);
        let filename = format!("{}_{}.wav", timestamp, slugified_transcription);
        let file_path = self.recordings_dir.join(filename);

        let mut file = File::create(&file_path).await?;
        file.write_all(&audio_data).await?;

        self.cleanup_old_recordings().await?;

        Ok(())
    }

    async fn cleanup_old_recordings(&self) -> Result<(), Box<dyn Error>> {
        let mut entries: Vec<_> = fs::read_dir(&self.recordings_dir)?
            .filter_map(|entry| entry.ok())
            .collect();

        entries.sort_by_key(|entry| entry.metadata().unwrap().modified().unwrap());

        if entries.len() > self.max_recordings {
            for entry in entries.iter().take(entries.len() - self.max_recordings) {
                fs::remove_file(entry.path())?;
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
