//! Stream errors that cpal reports on an audio thread.

use tokio::sync::mpsc;

/// Passes stream errors from a cpal error callback to the processing side,
/// which logs them. The callback runs on an audio thread, so it does not
/// log itself (`Logger` prints and takes the egui context lock); a report
/// is a `try_send`, which does not wait.
///
/// cpal can call the error callback in a loop with the same error, for
/// example while a device is unplugged, so an error is sent again only
/// after a different one. An error that did not fit in the channel is sent
/// when cpal reports it again.
pub(crate) struct StreamErrorReporter<E> {
    tx: mpsc::Sender<E>,
    /// Starts each message, for example "Audio input error"
    label: &'static str,
    /// Makes the value to send from the message
    wrap: fn(String) -> E,
    last_sent: Option<String>,
}

impl<E> StreamErrorReporter<E> {
    pub(crate) fn new(tx: mpsc::Sender<E>, label: &'static str, wrap: fn(String) -> E) -> Self {
        Self {
            tx,
            label,
            wrap,
            last_sent: None,
        }
    }

    pub(crate) fn report(&mut self, error: impl std::fmt::Display) {
        let message = format!("{}: {}", self.label, error);
        if self.last_sent.as_ref() == Some(&message) {
            return;
        }
        if self.tx.try_send((self.wrap)(message.clone())).is_ok() {
            self.last_sent = Some(message);
        }
    }
}
