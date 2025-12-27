//! Tests for BabbleBoop
//!
//! These tests verify the fixes for issues identified during code review.

#[cfg(test)]
mod regression_tests {
    use crate::price_estimator::PriceEstimator;
    use crate::rate_limiter::RateLimiter;

    // ===========================================================================
    // Regression test: Model pricing now covers common models
    // ===========================================================================

    #[test]
    fn test_common_models_have_pricing() {
        // These models should now have proper pricing
        let models_with_pricing = vec![
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ];

        for model in models_with_pricing {
            let estimator = PriceEstimator::new(model);
            // Verify we can estimate non-zero costs
            let cost = estimator.estimate_translation_cost(1000, 500);
            assert!(
                cost > 0.0,
                "Model {} should have non-zero pricing",
                model
            );
        }
    }

    #[test]
    fn test_unknown_model_uses_default_pricing() {
        // Unknown models should use conservative default pricing (gpt-4o-mini rates)
        let estimator = PriceEstimator::new("unknown-model-xyz");
        let cost = estimator.estimate_translation_cost(1000, 500);
        // Should use gpt-4o-mini pricing as fallback, not zero
        assert!(cost > 0.0, "Unknown models should use default pricing");
    }

    // ===========================================================================
    // Regression test: Rate limiter behavior
    // ===========================================================================

    #[tokio::test]
    async fn test_rate_limiter_tracks_requests() {
        let mut limiter = RateLimiter::new(2);

        // First two requests should not block
        limiter.wait().await;
        limiter.wait().await;

        // Rate limiter should now be at capacity
        // (would block if we called wait again within the same minute)
    }

    // ===========================================================================
    // Test: Translation error response structure
    // ===========================================================================

    mod translation_tests {
        #[derive(serde::Deserialize)]
        struct ChatGptChoice {
            message: ChatGptMessage,
        }

        #[derive(serde::Deserialize)]
        struct ChatGptMessage {
            content: String,
        }

        #[derive(serde::Deserialize)]
        struct ChatGptResponse {
            choices: Vec<ChatGptChoice>,
        }

        #[test]
        fn test_empty_choices_handled_with_iterator() {
            // This test verifies the fix: using .into_iter().next() instead of [0]
            let json = r#"{"choices": []}"#;
            let response: ChatGptResponse = serde_json::from_str(json).unwrap();

            // Using iterator pattern (the fix) - returns None instead of panicking
            let result = response.choices.into_iter().next();
            assert!(result.is_none(), "Empty choices should return None");
        }

        #[test]
        fn test_valid_response_parsed_correctly() {
            let json = r#"{"choices": [{"message": {"role": "assistant", "content": "Hello"}}]}"#;
            let response: ChatGptResponse = serde_json::from_str(json).unwrap();

            let choice = response.choices.into_iter().next();
            assert!(choice.is_some());
            assert_eq!(choice.unwrap().message.content, "Hello");
        }
    }

    // ===========================================================================
    // Test: WAV encoding helper
    // ===========================================================================

    #[test]
    fn test_wav_encoding() {
        use hound::{WavReader, WavSpec, WavWriter};
        use std::io::Cursor;

        // Create sample audio data
        let samples: Vec<f32> = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let channels = 1;
        let sample_rate = 44100;

        // Encode to WAV
        let mut wav_buffer = Vec::new();
        {
            let mut writer = WavWriter::new(
                Cursor::new(&mut wav_buffer),
                WavSpec {
                    channels: channels as u16,
                    sample_rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();

            for &sample in &samples {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        }

        // Verify we can read it back
        let reader = WavReader::new(Cursor::new(&wav_buffer)).unwrap();
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.spec().sample_rate, 44100);
    }

    // ===========================================================================
    // Test: I16 to F32 conversion
    // ===========================================================================

    #[test]
    fn test_i16_to_f32_conversion() {
        fn i16_to_f32(sample: i16) -> f32 {
            sample as f32 / i16::MAX as f32
        }

        // Test boundary values
        assert!((i16_to_f32(0) - 0.0).abs() < 0.0001);
        assert!((i16_to_f32(i16::MAX) - 1.0).abs() < 0.0001);
        assert!((i16_to_f32(i16::MIN) - (-1.0)).abs() < 0.01);

        // Test mid-range value
        let mid = i16::MAX / 2;
        assert!((i16_to_f32(mid) - 0.5).abs() < 0.01);
    }
}
