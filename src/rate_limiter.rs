use std::time::{Duration, Instant};
use tokio::time::sleep;

pub struct RateLimiter {
    last_request: Instant,
    request_count: usize,
    max_requests: usize,
}

impl RateLimiter {
    pub fn new(max_requests: usize) -> Self {
        RateLimiter {
            last_request: Instant::now(),
            request_count: 0,
            max_requests,
        }
    }

    pub async fn wait(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_request);

        if elapsed < Duration::from_secs(60) {
            if self.request_count >= self.max_requests {
                let wait_time = Duration::from_secs(60) - elapsed;
                sleep(wait_time).await;
                self.request_count = 0;
                self.last_request = Instant::now();
            }
        } else {
            self.request_count = 0;
            self.last_request = now;
        }

        self.request_count += 1;
    }
}
