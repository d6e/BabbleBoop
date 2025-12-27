//! Investigation tests for issues found in commit review
//!
//! These tests demonstrate and verify the issues identified.

#[cfg(test)]
mod shutdown_investigation {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc;

    /// Issue 1: Processing thread blocks forever when GUI closes
    ///
    /// The tokio::select! loop in run_processing_loop waits on:
    /// 1. rx.recv() - audio events
    /// 2. cmd_rx.recv() - commands from GUI
    ///
    /// When GUI closes:
    /// - cmd_tx is dropped (GUI's app_state goes out of scope)
    /// - cmd_rx.recv() returns None
    /// - BUT rx (audio events) is still open because audio thread is parked
    /// - So tokio::select! keeps waiting on rx.recv()
    ///
    /// The `else` branch only triggers when BOTH channels are closed.
    #[tokio::test]
    async fn test_select_blocks_when_one_channel_open() {
        let (audio_tx, mut audio_rx) = mpsc::channel::<String>(10);
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<String>(10);

        // Simulate GUI closing - drop cmd_tx
        drop(cmd_tx);

        // audio_tx is still alive, simulating parked audio thread holding tx
        let _audio_tx = audio_tx;

        // This select would block forever in the real code
        // We use timeout to prove it
        let result = tokio::time::timeout(Duration::from_millis(100), async {
            tokio::select! {
                Some(msg) = audio_rx.recv() => {
                    format!("audio: {}", msg)
                }
                Some(msg) = cmd_rx.recv() => {
                    format!("cmd: {}", msg)
                }
                else => {
                    "both closed".to_string()
                }
            }
        })
        .await;

        // This will timeout because:
        // - cmd_rx returns None immediately (closed)
        // - audio_rx blocks forever (tx still exists)
        // - else branch never triggers (needs BOTH to be closed)
        assert!(
            result.is_err(),
            "Expected timeout because select blocks when one channel is open"
        );
    }

    /// Demonstrates correct behavior: both channels closed
    #[tokio::test]
    async fn test_select_exits_when_both_channels_closed() {
        let (audio_tx, mut audio_rx) = mpsc::channel::<String>(10);
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<String>(10);

        // Close both channels
        drop(cmd_tx);
        drop(audio_tx);

        let result = tokio::time::timeout(Duration::from_millis(100), async {
            tokio::select! {
                Some(msg) = audio_rx.recv() => {
                    format!("audio: {}", msg)
                }
                Some(msg) = cmd_rx.recv() => {
                    format!("cmd: {}", msg)
                }
                else => {
                    "both closed".to_string()
                }
            }
        })
        .await;

        // This completes immediately
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "both closed");
    }

    /// Issue 2: Audio thread parked forever
    ///
    /// The audio thread does:
    /// ```ignore
    /// loop { std::thread::park(); }
    /// ```
    ///
    /// This keeps the Stream alive (which is needed for audio capture).
    /// However, when the app closes:
    /// 1. The thread is never unparked
    /// 2. The thread holds `tx` (audio event sender)
    /// 3. This prevents the processing thread from exiting (see Issue 1)
    ///
    /// The fix would be to use a shutdown signal instead of park().
    #[test]
    fn test_parked_thread_blocks_channel_close() {
        let (tx, mut rx) = std::sync::mpsc::channel::<()>();

        let handle = std::thread::spawn(move || {
            let _tx = tx; // Hold the sender
            loop {
                std::thread::park(); // Park forever like the audio thread
            }
        });

        // Give thread time to start
        std::thread::sleep(Duration::from_millis(50));

        // Try to receive - this will block because tx is held by parked thread
        let result = rx.recv_timeout(Duration::from_millis(100));

        // Timeout because the channel is never closed (tx still held)
        assert!(result.is_err());

        // Note: We can't clean up this thread - it's parked forever
        // This is exactly the problem in the real code
        // We just drop the handle and let the test end
        drop(handle);
    }
}

#[cfg(test)]
mod typing_indicator_config_investigation {
    use std::sync::Arc;

    /// Issue 3: TypingIndicator holds stale config
    ///
    /// When AppCommand::UpdateConfig is received:
    /// - rate_limiter is recreated with new config
    /// - recording_manager is recreated with new config
    /// - BUT typing_indicator still holds Arc<Config> from startup
    ///
    /// This means OSC address/port changes won't take effect for typing indicator.
    #[derive(Clone)]
    struct MockConfig {
        osc_port: u16,
    }

    struct MockTypingIndicator {
        config: Arc<MockConfig>,
    }

    impl MockTypingIndicator {
        fn new(config: Arc<MockConfig>) -> Self {
            Self { config }
        }

        fn get_port(&self) -> u16 {
            self.config.osc_port
        }
    }

    #[test]
    fn test_typing_indicator_holds_stale_config() {
        // Initial config
        let config = Arc::new(MockConfig { osc_port: 9000 });

        // Create typing indicator with initial config
        let indicator = MockTypingIndicator::new(Arc::clone(&config));

        // "Update" the config (in real code, this is what happens in UpdateConfig handler)
        // The app_state.config is updated, but typing_indicator still holds old Arc
        let new_config = Arc::new(MockConfig { osc_port: 9001 });

        // Typing indicator still sees old port
        assert_eq!(indicator.get_port(), 9000);

        // New config has different port
        assert_eq!(new_config.osc_port, 9001);

        // This proves the typing indicator won't see config updates
        assert_ne!(
            indicator.get_port(),
            new_config.osc_port,
            "TypingIndicator should be stale after config update"
        );
    }
}

#[cfg(test)]
mod rwlock_investigation {
    use std::sync::{Arc, RwLock};

    /// Issue 4: RwLock unwrap without error handling
    ///
    /// The code uses `.unwrap()` on RwLock guards:
    /// ```ignore
    /// app_state.config.read().unwrap().clone()
    /// ```
    ///
    /// RwLock is only poisoned if a thread panics while holding the lock.
    /// In practice, this is rare but possible.
    ///
    /// VERDICT: LOW risk - panics during lock are rare, and if they happen,
    /// the app is likely in a bad state anyway. Using .expect() with a message
    /// would be slightly better for debugging.
    #[test]
    fn test_rwlock_poisoning() {
        let lock = Arc::new(RwLock::new(42));
        let lock_clone = Arc::clone(&lock);

        // Spawn thread that will panic while holding write lock
        let handle = std::thread::spawn(move || {
            let _guard = lock_clone.write().unwrap();
            panic!("intentional panic while holding lock");
        });

        // Wait for thread to panic
        let _ = handle.join();

        // Lock is now poisoned
        let result = lock.read();
        assert!(result.is_err(), "Lock should be poisoned after panic");

        // .unwrap() would panic here - the code should use .expect() or handle error
        // But we can still access the data if we want:
        let guard = lock.read().unwrap_or_else(|e| e.into_inner());
        assert_eq!(*guard, 42);
    }
}

#[cfg(test)]
mod config_path_investigation {
    /// Issue 5: Duplicate CONFIG_PATH constant
    ///
    /// main.rs:18 defines: const CONFIG_PATH: &str = "config.toml";
    /// gui.rs:7 defines:   const CONFIG_PATH: &str = "config.toml";
    ///
    /// VERDICT: TRUE issue - violates DRY principle.
    /// If someone changes one and not the other, bugs ensue.
    ///
    /// Should be defined in config.rs or lib.rs and imported.
    #[test]
    fn test_constants_match() {
        // These are the literal values from the code
        let main_config_path = "config.toml";
        let gui_config_path = "config.toml";

        // They match NOW, but if someone changes one...
        assert_eq!(
            main_config_path, gui_config_path,
            "CONFIG_PATH should be defined once and shared"
        );
    }
}

#[cfg(test)]
mod blocking_send_investigation {
    use std::time::Duration;
    use tokio::sync::mpsc;

    /// Issue 6: blocking_send in GUI thread
    ///
    /// The GUI uses blocking_send on tokio channel:
    /// ```ignore
    /// self.app_state.command_tx.blocking_send(AppCommand::SetEnabled(new_state));
    /// ```
    ///
    /// This will block the GUI thread if the channel buffer (32) is full.
    ///
    /// VERDICT: LOW risk in practice because:
    /// 1. Channel buffer is 32, commands are processed quickly
    /// 2. User can't click fast enough to fill buffer
    /// 3. try_send would lose messages, which might be worse
    ///
    /// However, silent error ignoring (let _ = ...) is still a concern.
    #[tokio::test]
    async fn test_blocking_send_behavior() {
        let (tx, mut rx) = mpsc::channel::<i32>(2); // Small buffer

        // Fill the buffer
        tx.send(1).await.unwrap();
        tx.send(2).await.unwrap();

        // Next send would block... but we're testing blocking_send from sync context
        // In a real scenario, the GUI would freeze here

        // Drain one to make room
        assert_eq!(rx.recv().await, Some(1));

        // Now we can send again
        tx.send(3).await.unwrap();
    }

    /// Test silent error ignoring
    #[tokio::test]
    async fn test_silent_error_on_dropped_receiver() {
        let (tx, rx) = mpsc::channel::<i32>(10);

        // Drop receiver
        drop(rx);

        // Send fails silently with `let _ = `
        let result = tx.send(1).await;
        assert!(result.is_err(), "Send should fail when receiver is dropped");

        // In the real code, this error is ignored:
        // let _ = self.app_state.command_tx.blocking_send(...);
        // The user would never know their command wasn't processed
    }
}
