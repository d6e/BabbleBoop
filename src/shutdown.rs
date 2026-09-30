use std::future::Future;
use std::sync::Arc;
use tokio::sync::watch;

/// Shutdown request shared by the GUI, the processing loop and the audio
/// thread. Synchronous code polls `is_requested`; async code awaits
/// `requested` or wraps work in `run_until`.
#[derive(Clone)]
pub struct Shutdown {
    tx: Arc<watch::Sender<bool>>,
}

impl Default for Shutdown {
    fn default() -> Self {
        Self::new()
    }
}

impl Shutdown {
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(false);
        Self { tx: Arc::new(tx) }
    }

    pub fn request(&self) {
        self.tx.send_replace(true);
    }

    pub fn is_requested(&self) -> bool {
        *self.tx.borrow()
    }

    /// Resolves when shutdown is requested, at once if it already was.
    pub async fn requested(&self) {
        let mut rx = self.tx.subscribe();
        #[expect(
            clippy::let_underscore_must_use,
            reason = "wait_for fails only when the sender is dropped, and self holds it"
        )]
        let _ = rx.wait_for(|requested| *requested).await;
    }

    /// Run `work` until it finishes or shutdown is requested. Returns `None`
    /// if shutdown stopped the work first; `work` is then dropped.
    pub async fn run_until<F: Future>(&self, work: F) -> Option<F::Output> {
        tokio::select! {
            biased;
            _ = self.requested() => None,
            output = work => Some(output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn test_shutdown_requested_before_wait_is_seen() {
        use std::time::Duration;

        let shutdown = Shutdown::new();
        shutdown.request();

        assert!(shutdown.is_requested());
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            shutdown.run_until(std::future::pending::<()>()),
        )
        .await;
        assert_eq!(result, Ok(None));
    }

    #[tokio::test]
    async fn test_run_until_returns_output_without_shutdown() {
        let shutdown = Shutdown::new();
        assert_eq!(shutdown.run_until(async { 7 }).await, Some(7));
        assert!(!shutdown.is_requested());
    }
}
