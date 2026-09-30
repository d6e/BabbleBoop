use std::time::Duration;
use tokio::time::{sleep, Instant};

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

    /// Change the limit. Requests already made in the current minute still
    /// count against it.
    pub fn set_max_requests(&mut self, max_requests: usize) {
        self.max_requests = max_requests;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn test_rate_limiter_waits_only_when_the_budget_is_used_up() {
        let mut limiter = RateLimiter::new(2);
        let mut waits = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            limiter.wait().await;
            waits.push(start.elapsed());
        }

        assert_eq!(
            waits,
            [Duration::ZERO, Duration::ZERO, Duration::from_secs(60)]
        );
    }
}
