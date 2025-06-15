# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Build: `cargo build` (development) or `cargo build --release` (optimized)
- Run: `cargo run --release`
- Format: `cargo fmt`
- Lint: `cargo clippy`
- Test: `cargo test` (specific test: `cargo test test_name`)

## Code Style Guidelines
- **Imports**: Group by source (crate-level first, then external, then std)
- **Formatting**: Follow standard Rust formatting (rustfmt)
- **Types**: Use descriptive struct definitions with appropriate derive traits; use Serde for serialization
- **Naming**: snake_case for variables/functions, CamelCase for types/structs
- **Error handling**: Use Result<T, Box<dyn Error>> with ? operator; provide descriptive error messages
- **Async**: Utilize Tokio runtime with async/await syntax
- **Documentation**: Maintain clear function names and add comments for complex logic
- **Configuration**: Use TOML for configuration with structured types

## Project Overview
BabbleBoop is a VRChat audio translation utility using OpenAI APIs for transcription and translation.