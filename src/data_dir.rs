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
    /// The config file is next to the executable. The working directory
    /// has no config file, or it is the folder of the executable.
    ExeDir,
    /// No folder has a config file yet. BabbleBoop creates it next to
    /// the executable.
    NewInExeDir,
    /// The config file is in the working directory and not next to the
    /// executable.
    WorkingDir,
    /// The config file is in the working directory and next to the
    /// executable. The one in the working directory is used.
    WorkingDirOverExeDir { exe_dir: PathBuf },
    /// No folder has a config file yet, and BabbleBoop cannot write to
    /// the folder of the executable. BabbleBoop creates the config file
    /// in the working directory.
    ExeDirNotWritable { exe_dir: PathBuf, error: String },
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
            Reason::NewInExeDir => write!(
                f,
                ", the folder of the executable, because no folder has {} yet",
                CONFIG_FILE
            ),
            Reason::WorkingDir => write!(
                f,
                ", the working directory, because {} is there and not next to the executable",
                CONFIG_FILE
            ),
            Reason::WorkingDirOverExeDir { exe_dir } => {
                let ignored = DataDir::new(exe_dir.clone());
                write!(
                    f,
                    ", the working directory, because {} is there. It does not use {} and {} next to the executable",
                    CONFIG_FILE,
                    ignored.config_file().display(),
                    ignored.cost_file().display()
                )
            }
            Reason::ExeDirNotWritable { exe_dir, error } => write!(
                f,
                ", the working directory, because no folder has {} yet and BabbleBoop cannot write to the folder of the executable {}: {}",
                CONFIG_FILE,
                exe_dir.display(),
                error
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
/// `can_write` tells if BabbleBoop can create a file in a folder, or why
/// not; it is called only when no folder has a config file.
///
/// 1. The working directory, if it has a config file, so that a user who
///    starts BabbleBoop from the folder of the config file keeps the
///    settings and the total cost.
/// 2. Else the folder of the executable, if it has a config file.
/// 3. Else the folder of the executable, if BabbleBoop can write to it;
///    if not, the working directory.
///
/// If the folder of the executable is not known, the working directory.
pub fn choose(
    working_dir: PathBuf,
    exe_dir: Result<PathBuf, String>,
    has_config: impl Fn(&Path) -> bool,
    can_write: impl Fn(&Path) -> Result<(), String>,
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
    let (dir, reason) = if working_dir == exe_dir {
        let reason = if has_config(&exe_dir) {
            Reason::ExeDir
        } else {
            Reason::NewInExeDir
        };
        (exe_dir, reason)
    } else if has_config(&working_dir) {
        if has_config(&exe_dir) {
            (working_dir, Reason::WorkingDirOverExeDir { exe_dir })
        } else {
            (working_dir, Reason::WorkingDir)
        }
    } else if has_config(&exe_dir) {
        (exe_dir, Reason::ExeDir)
    } else {
        match can_write(&exe_dir) {
            Ok(()) => (exe_dir, Reason::NewInExeDir),
            Err(error) => (working_dir, Reason::ExeDirNotWritable { exe_dir, error }),
        }
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

/// `choose`, with the config files that are on disk and a test file for
/// write access.
fn locate_in(working_dir: PathBuf, exe_dir: Result<PathBuf, String>) -> Choice {
    // A working directory that is the folder of the executable, for
    // example after a double click on the executable, can have another
    // spelling. It is only the folder of the executable.
    let working_dir = match &exe_dir {
        Ok(exe_dir) if same_dir(&working_dir, exe_dir) => exe_dir.clone(),
        _ => working_dir,
    };
    choose(
        working_dir,
        exe_dir,
        |dir| dir.join(CONFIG_FILE).exists(),
        can_write,
    )
}

/// If BabbleBoop can create a file in `dir`, or why not. Creates an empty
/// file with a name that no other file has, and removes it.
fn can_write(dir: &Path) -> Result<(), String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |time| time.subsec_nanos());
    let file = dir.join(format!(
        ".babble_boop_write_test_{}_{}",
        std::process::id(),
        nanos
    ));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&file)
        .map_err(|e| e.to_string())?;
    // The file was created, so the folder can be written. If the remove
    // fails, the empty file stays; the check still gives the answer.
    if std::fs::remove_file(&file).is_err() {
        eprintln!("Cannot remove the test file {}", file.display());
    }
    Ok(())
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

    fn writable(_: &Path) -> Result<(), String> {
        Ok(())
    }

    fn read_only(_: &Path) -> Result<(), String> {
        Err("permission denied".to_string())
    }

    /// The choice when the config file is in the folders of `configs`,
    /// and `can_write` tells if a folder can be written.
    fn choice_with(
        exe_dir: Result<PathBuf, String>,
        configs: &[PathBuf],
        can_write: impl Fn(&Path) -> Result<(), String>,
    ) -> Choice {
        choose(
            working_dir(),
            exe_dir,
            |dir| configs.iter().any(|config_dir| config_dir == dir),
            can_write,
        )
    }

    fn choice_with_configs(exe_dir: Result<PathBuf, String>, configs: &[PathBuf]) -> Choice {
        choice_with(exe_dir, configs, writable)
    }

    #[test]
    fn test_config_only_in_the_working_directory_keeps_the_working_directory() {
        for can_write in [writable, read_only] {
            let choice = choice_with(Ok(exe_dir()), &[working_dir()], can_write);
            assert_eq!(
                choice,
                Choice {
                    dir: DataDir::new(working_dir()),
                    reason: Reason::WorkingDir
                }
            );
            assert!(!choice.is_error());
        }
    }

    #[test]
    fn test_config_only_next_to_the_executable_uses_the_executable_folder() {
        for can_write in [writable, read_only] {
            let choice = choice_with(Ok(exe_dir()), &[exe_dir()], can_write);
            assert_eq!(
                choice,
                Choice {
                    dir: DataDir::new(exe_dir()),
                    reason: Reason::ExeDir
                }
            );
            assert!(!choice.is_error());
        }
    }

    #[test]
    fn test_config_in_both_folders_uses_the_working_directory() {
        let choice = choice_with_configs(Ok(exe_dir()), &[working_dir(), exe_dir()]);
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(working_dir()),
                reason: Reason::WorkingDirOverExeDir { exe_dir: exe_dir() }
            }
        );
        assert!(!choice.is_error());
    }

    #[test]
    fn test_log_line_names_the_files_next_to_the_executable_that_are_not_used() {
        let choice = choice_with_configs(Ok(exe_dir()), &[working_dir(), exe_dir()]);
        let line = choice.to_string();
        for ignored in [
            exe_dir().join("config.toml"),
            exe_dir().join("total_cost.txt"),
        ] {
            assert!(
                line.contains(&ignored.display().to_string()),
                "{:?} not in {}",
                ignored,
                line
            );
        }
        assert!(
            line.contains(&working_dir().display().to_string()),
            "{}",
            line
        );
    }

    #[test]
    fn test_no_config_and_a_writable_executable_folder_uses_the_executable_folder() {
        let choice = choice_with_configs(Ok(exe_dir()), &[]);
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(exe_dir()),
                reason: Reason::NewInExeDir
            }
        );
        assert!(!choice.is_error());
    }

    #[test]
    fn test_no_config_and_an_unwritable_executable_folder_uses_the_working_directory() {
        let choice = choice_with(Ok(exe_dir()), &[], read_only);
        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(working_dir()),
                reason: Reason::ExeDirNotWritable {
                    exe_dir: exe_dir(),
                    error: "permission denied".to_string()
                }
            }
        );
        assert!(!choice.is_error());
        // The log line tells why the folder of the executable is not used
        let line = choice.to_string();
        assert!(line.contains(&exe_dir().display().to_string()), "{}", line);
        assert!(line.contains("permission denied"), "{}", line);
    }

    #[test]
    fn test_write_access_is_checked_only_for_the_executable_folder_without_a_config() {
        let checked = std::cell::RefCell::new(Vec::new());
        let can_write = |dir: &Path| {
            checked.borrow_mut().push(dir.to_path_buf());
            Ok(())
        };
        choice_with(Ok(exe_dir()), &[], can_write);
        assert_eq!(*checked.borrow(), vec![exe_dir()]);

        for configs in [
            vec![working_dir()],
            vec![exe_dir()],
            vec![working_dir(), exe_dir()],
        ] {
            checked.borrow_mut().clear();
            choice_with(Ok(exe_dir()), &configs, can_write);
            assert!(checked.borrow().is_empty(), "{:?}", configs);
        }
    }

    #[test]
    fn test_working_directory_that_is_the_executable_folder_is_the_executable_folder() {
        for can_write in [writable, read_only] {
            let with_config = choose(exe_dir(), Ok(exe_dir()), |dir| dir == exe_dir(), can_write);
            assert_eq!(
                with_config,
                Choice {
                    dir: DataDir::new(exe_dir()),
                    reason: Reason::ExeDir
                }
            );
            // No other folder to fall back to: the load error names the
            // file if it cannot be created
            let without_config = choose(exe_dir(), Ok(exe_dir()), |_| false, can_write);
            assert_eq!(
                without_config,
                Choice {
                    dir: DataDir::new(exe_dir()),
                    reason: Reason::NewInExeDir
                }
            );
        }
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

    fn entries(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect()
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
    fn test_locate_writes_no_file_when_it_checks_the_executable_folder() {
        let root = temp_folders("write_check");

        let choice = locate_in(root.join("start"), Ok(root.join("exe")));
        let left = entries(&root.join("exe"));
        std::fs::remove_dir_all(&root).unwrap();

        assert_eq!(
            choice,
            Choice {
                dir: DataDir::new(root.join("exe")),
                reason: Reason::NewInExeDir
            }
        );
        assert!(left.is_empty(), "{:?}", left);
    }

    #[test]
    fn test_can_write_fails_for_a_missing_folder() {
        let root = temp_folders("missing");
        let result = can_write(&root.join("missing"));
        std::fs::remove_dir_all(&root).unwrap();
        assert!(result.is_err(), "{:?}", result);
    }

    /// Other platforms: no portable way to make a folder that the test
    /// cannot write, so only `test_can_write_fails_for_a_missing_folder`
    /// covers the failure there.
    #[cfg(unix)]
    #[test]
    fn test_locate_uses_the_working_directory_when_the_executable_folder_is_read_only() {
        use std::os::unix::fs::PermissionsExt;
        let root = temp_folders("read_only");
        let exe = root.join("exe");
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o555)).unwrap();
        // root can write to a read only folder
        let bypassed = std::fs::write(exe.join("probe"), "").is_ok();

        let choice = locate_in(root.join("start"), Ok(exe.clone()));
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::remove_dir_all(&root).unwrap();

        if bypassed {
            eprintln!("skipped: this user can write to a read only folder");
            return;
        }
        assert_eq!(choice.dir, DataDir::new(root.join("start")));
        assert!(
            matches!(&choice.reason, Reason::ExeDirNotWritable { exe_dir, .. } if *exe_dir == exe),
            "{:?}",
            choice.reason
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
