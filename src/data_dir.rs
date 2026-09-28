//! The folder of the files that BabbleBoop reads and writes: `config.toml`,
//! `total_cost.txt` and the `recordings` folder.

use std::fmt;
use std::path::{Path, PathBuf};

const CONFIG_FILE: &str = "config.toml";
const COST_FILE: &str = "total_cost.txt";
const RECORDINGS_DIR: &str = "recordings";

/// The folder that holds the config file, the total cost and the saved
/// recordings.
#[derive(Clone, Debug, PartialEq)]
pub struct DataDir(PathBuf);

impl DataDir {
    pub fn new(path: PathBuf) -> Self {
        Self(path)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn config_file(&self) -> PathBuf {
        self.0.join(CONFIG_FILE)
    }

    pub fn cost_file(&self) -> PathBuf {
        self.0.join(COST_FILE)
    }

    /// Saved recordings, when `keep_audio_files` is on.
    pub fn recordings_dir(&self) -> PathBuf {
        self.0.join(RECORDINGS_DIR)
    }
}

/// Why BabbleBoop uses its data folder.
#[derive(Clone, Debug, PartialEq)]
pub enum Reason {
    /// The folder of the executable. The config file is there, or it is
    /// in no folder yet.
    ExeDir,
    /// The config file is in the working directory and not next to the
    /// executable. Earlier versions always used the working directory.
    WorkingDir,
    /// The config file is next to the executable and in the working
    /// directory. The one next to the executable is used.
    ExeDirOverWorkingDir { working_dir: PathBuf },
    /// The folder of the executable is not known.
    ExeDirUnknown { error: String },
}

/// The data folder and why BabbleBoop uses it.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub dir: DataDir,
    pub reason: Reason,
}

impl Choice {
    /// The activity log shows the choice as an error only when the folder
    /// of the executable is not known.
    pub fn is_error(&self) -> bool {
        matches!(self.reason, Reason::ExeDirUnknown { .. })
    }
}

impl fmt::Display for Choice {
    /// The line for the activity log, so that users can find their files.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dir = self.dir.path().display();
        write!(
            f,
            "BabbleBoop keeps {}, {} and {} in {}",
            CONFIG_FILE, COST_FILE, RECORDINGS_DIR, dir
        )?;
        match &self.reason {
            Reason::ExeDir => write!(f, ", the folder of the executable"),
            Reason::WorkingDir => write!(
                f,
                ", the working directory, because {} is there and not next to the executable",
                CONFIG_FILE
            ),
            Reason::ExeDirOverWorkingDir { working_dir } => write!(
                f,
                ", the folder of the executable. It does not use the {} in the working directory {}",
                CONFIG_FILE,
                working_dir.display()
            ),
            Reason::ExeDirUnknown { error } => write!(
                f,
                ", the working directory, because the folder of the executable is not known: {}",
                error
            ),
        }
    }
}

/// Select the data folder. `exe_dir` is the folder of the executable, or
/// why it is not known. `has_config` tells if a folder has a config file.
///
/// The folder of the executable, unless only the working directory has a
/// config file: then the working directory, so that a user who started
/// BabbleBoop from the folder of the config file keeps the settings and
/// the total cost.
pub fn choose(
    working_dir: PathBuf,
    exe_dir: Result<PathBuf, String>,
    has_config: impl Fn(&Path) -> bool,
) -> Choice {
    let exe_dir = match exe_dir {
        Ok(exe_dir) => exe_dir,
        Err(error) => {
            return Choice {
                dir: DataDir::new(working_dir),
                reason: Reason::ExeDirUnknown { error },
            }
        }
    };
    let in_working_dir = working_dir != exe_dir && has_config(&working_dir);
    let (dir, reason) = match (has_config(&exe_dir), in_working_dir) {
        (true, true) => (exe_dir, Reason::ExeDirOverWorkingDir { working_dir }),
        (false, true) => (working_dir, Reason::WorkingDir),
        (_, false) => (exe_dir, Reason::ExeDir),
    };
    Choice {
        dir: DataDir::new(dir),
        reason,
    }
}

/// Select the data folder of this process. See `choose`.
pub fn locate() -> Choice {
    // Relative paths resolve against the working directory, so "." is the
    // same folder if its full path cannot be read.
    let working_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    locate_in(working_dir, exe_dir())
}

/// `choose`, with the config files that are on disk.
fn locate_in(working_dir: PathBuf, exe_dir: Result<PathBuf, String>) -> Choice {
    // A working directory that is the folder of the executable, for
    // example after a double click on the executable, can have another
    // spelling. It is only the folder of the executable.
    let working_dir = match &exe_dir {
        Ok(exe_dir) if same_dir(&working_dir, exe_dir) => exe_dir.clone(),
        _ => working_dir,
    };
    choose(working_dir, exe_dir, |dir| dir.join(CONFIG_FILE).exists())
}

/// The folder of the running executable. Not canonicalized: on Windows
/// that adds a `\\?\` prefix to the path that the activity log shows.
fn exe_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    match exe.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => Ok(dir.to_path_buf()),
        _ => Err(format!("{} has no parent folder", exe.display())),
    }
}

/// If two paths name the same folder. Canonicalized only to compare, as
/// both paths get the same form.
fn same_dir(a: &Path, b: &Path) -> bool {
    a == b
        || matches!(
            (a.canonicalize(), b.canonicalize()),
            (Ok(a), Ok(b)) if a == b
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn working_dir() -> PathBuf {
        PathBuf::from("/home/user/start")
    }

    fn exe_dir() -> PathBuf {
        PathBuf::from("/opt/babble_boop")
    }

    /// The choice when the config file is in the folders of `configs`.
    fn choice_with_configs(exe_dir: Result<PathBuf, String>, configs: &[PathBuf]) -> Choice {
        choose(working_dir(), exe_dir, |dir| {
            configs.iter().any(|config_dir| config_dir == dir)
        })
    }

    #[test]
    fn test_config_only_in_the_working_directory_keeps_the_working_directory() {
        let choice = choice_with_configs(Ok(exe_dir()), &[working_dir()]);
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(working_dir()),
                reason: Reason::WorkingDir
            }
        );
        assert!(!choice.is_error());
    }

    #[test]
    fn test_config_only_next_to_the_executable_uses_the_executable_folder() {
        let choice = choice_with_configs(Ok(exe_dir()), &[exe_dir()]);
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(exe_dir()),
                reason: Reason::ExeDir
            }
        );
        assert!(!choice.is_error());
    }

    #[test]
    fn test_config_in_both_folders_uses_the_executable_folder() {
        let choice = choice_with_configs(Ok(exe_dir()), &[working_dir(), exe_dir()]);
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(exe_dir()),
                reason: Reason::ExeDirOverWorkingDir {
                    working_dir: working_dir()
                }
            }
        );
        // The log line names the config file that is not used
        let line = choice.to_string();
        assert!(
            line.contains(&working_dir().display().to_string()),
            "{}",
            line
        );
        assert!(!choice.is_error());
    }

    #[test]
    fn test_no_config_uses_the_executable_folder() {
        let choice = choice_with_configs(Ok(exe_dir()), &[]);
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(exe_dir()),
                reason: Reason::ExeDir
            }
        );
    }

    #[test]
    fn test_working_directory_that_is_the_executable_folder_is_the_executable_folder() {
        let choice = choose(exe_dir(), Ok(exe_dir()), |dir| dir == exe_dir());
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(exe_dir()),
                reason: Reason::ExeDir
            }
        );
    }

    #[test]
    fn test_unknown_executable_folder_uses_the_working_directory() {
        for configs in [vec![], vec![working_dir()]] {
            let choice = choice_with_configs(Err("no such file".to_string()), &configs);
            assert_eq!(
                choice,
                Choice {
                    dir: DataDir::new(working_dir()),
                    reason: Reason::ExeDirUnknown {
                        error: "no such file".to_string()
                    }
                },
                "{:?}",
                configs
            );
            assert!(choice.is_error());
            let line = choice.to_string();
            assert!(line.contains("no such file"), "{}", line);
        }
    }

    #[test]
    fn test_log_line_names_the_folder() {
        let choice = choice_with_configs(Ok(exe_dir()), &[]);
        let line = choice.to_string();
        assert!(line.contains(&exe_dir().display().to_string()), "{}", line);
    }

    /// A new empty folder for a test, with the folders `start` and `exe`.
    fn temp_folders(test_name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "babble_boop_data_dir_{}_{}",
            test_name,
            std::process::id()
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(root.join("start")).unwrap();
        std::fs::create_dir_all(root.join("exe")).unwrap();
        root
    }

    #[test]
    fn test_locate_finds_the_config_file_on_disk() {
        let root = temp_folders("on_disk");
        std::fs::write(root.join("start").join("config.toml"), "").unwrap();

        let choice = locate_in(root.join("start"), Ok(root.join("exe")));
        std::fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(root.join("start")),
                reason: Reason::WorkingDir
            }
        );
    }

    #[test]
    fn test_locate_sees_the_executable_folder_in_another_spelling() {
        let root = temp_folders("spelling");
        std::fs::write(root.join("exe").join("config.toml"), "").unwrap();
        let working_dir = root.join("start").join("..").join("exe");

        let choice = locate_in(working_dir, Ok(root.join("exe")));
        std::fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(root.join("exe")),
                reason: Reason::ExeDir
            }
        );
    }

    #[test]
    fn test_files_are_in_the_data_folder() {
        let dir = DataDir::new(exe_dir());
        assert_eq!(dir.config_file(), exe_dir().join("config.toml"));
        assert_eq!(dir.cost_file(), exe_dir().join("total_cost.txt"));
        assert_eq!(dir.recordings_dir(), exe_dir().join("recordings"));
    }
}
