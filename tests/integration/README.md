# OSC Passthrough Integration Tests

These tests verify the OSC passthrough functionality of BabbleBoop.

## Running Tests

Run all tests:
```bash
cargo test
```

Run only OSC passthrough tests:
```bash
cargo test --test osc_passthrough_test
```

Run with single thread to avoid port conflicts:
```bash
cargo test --test osc_passthrough_test -- --test-threads=1
```

## Test Coverage

- **test_basic_osc_forwarding**: Verifies simple messages are forwarded from passthrough port to output port
- **test_facetracking_parameters**: Tests forwarding of typical facetracking OSC messages
- **test_invalid_packet_dropping**: Ensures non-OSC packets are filtered and not forwarded
- **test_concurrent_operations**: Tests high-throughput message forwarding with multiple senders
- **test_graceful_shutdown**: Verifies the passthrough can be cleanly shut down
- **test_passthrough_disabled**: Ensures passthrough doesn't start when disabled in config
- **test_shared_socket_source_port**: Verifies forwarded messages come from the main socket's port
- **test_socket_bind_failure**: Tests error handling when passthrough port is already in use
- **test_osc_bundle_forwarding**: Verifies OSC bundles are properly forwarded

## Port Usage

Tests use high port numbers (19000-19040) to avoid conflicts with running services.