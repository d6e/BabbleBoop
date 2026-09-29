//! Sample rate conversion for mono audio, without a crate: a windowed sinc
//! low-pass filter when the rate goes down, then linear interpolation.

use std::f64::consts::PI;

/// Filter taps on each side of the centre, per input sample per output
/// sample. More taps give a steeper low-pass filter.
const TAPS_PER_RATIO: f64 = 16.0;

/// Low-pass cutoff as a fraction of the output Nyquist frequency. The
/// filter rolls off around it, so the margin keeps most content just below
/// the output Nyquist frequency out of the aliased band.
const CUTOFF: f64 = 0.85;

/// Resample mono audio. When the rate goes down, a windowed sinc low-pass
/// filter first removes the frequencies that the output rate cannot hold,
/// so they do not fold back into lower frequencies. Linear interpolation
/// between the filtered samples then gives the values at the output times.
pub(crate) fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if input.is_empty() || from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    // Input samples per output sample
    let step = f64::from(from_rate) / f64::from(to_rate);
    let output_len = (input.len() as f64 / step).round() as usize;
    let kernel = (step > 1.0).then(|| low_pass_kernel(step));
    let value_at = |index: usize| -> f64 {
        match &kernel {
            Some(kernel) => filtered_sample(input, kernel, index),
            None => f64::from(input[index]),
        }
    };

    let last = input.len() - 1;
    (0..output_len)
        .map(|k| {
            let position = k as f64 * step;
            let index = (position.floor() as usize).min(last);
            let fraction = position - index as f64;
            let current = value_at(index);
            let value = if fraction > 0.0 && index < last {
                current + (value_at(index + 1) - current) * fraction
            } else {
                current
            };
            value as f32
        })
        .collect()
}

/// Blackman windowed sinc low-pass filter for downsampling by `step`,
/// normalised to unity gain at 0 Hz.
fn low_pass_kernel(step: f64) -> Vec<f64> {
    let half_width = (TAPS_PER_RATIO * step).ceil() as usize;
    let len = 2 * half_width + 1;
    // Cutoff in cycles per input sample
    let cutoff = CUTOFF * 0.5 / step;
    let mut kernel: Vec<f64> = (0..len)
        .map(|n| {
            let x = n as f64 - half_width as f64;
            let sinc = if x == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * PI * cutoff * x).sin() / (PI * x)
            };
            let phase = 2.0 * PI * n as f64 / (len - 1) as f64;
            let window = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
            sinc * window
        })
        .collect();
    let sum: f64 = kernel.iter().sum();
    for tap in &mut kernel {
        *tap /= sum;
    }
    kernel
}

/// Output of the filter at `index`, treating samples outside the input
/// as silence.
fn filtered_sample(input: &[f32], kernel: &[f64], index: usize) -> f64 {
    let half_width = kernel.len() / 2;
    let start = index.saturating_sub(half_width);
    let end = (index + half_width + 1).min(input.len());
    input
        .get(start..end)
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(offset, &sample)| {
            // Tap for the sample at start + offset; the kernel is symmetric
            let tap = start + offset + half_width - index;
            kernel[tap] * f64::from(sample)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f32, sample_rate: u32, seconds: f32, amplitude: f32) -> Vec<f32> {
        let len = (sample_rate as f32 * seconds) as usize;
        (0..len)
            .map(|n| {
                let t = n as f32 / sample_rate as f32;
                amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_resampled_sine_matches_the_ideal_samples() {
        // 44.1 kHz to 16 kHz: the output times fall between input samples
        let resampled = resample(&sine(1000.0, 44_100, 0.5, 0.5), 44_100, 16_000);
        let ideal = sine(1000.0, 16_000, 0.5, 0.5);
        assert_eq!(resampled.len(), ideal.len());
        // The filter treats samples before the start and after the end as
        // silence, so skip the edges.
        let edge = 100;
        let error = resampled[edge..ideal.len() - edge]
            .iter()
            .zip(&ideal[edge..ideal.len() - edge])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 0.01, "max error {}", error);
    }
}
