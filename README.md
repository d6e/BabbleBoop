# VRChat OSC LLM Translator

This is a fun little project I put together to experiment with real-time translation in VRChat using Large Language Models (LLMs). It's not perfect, but it's been interesting to play around with!

## What It Does

- Listens to your speech in VRChat
- Transcribes what you say using OpenAI's Whisper
- Translates the text using GPT models
- Sends the translation to VRChat's chat box via OSC

## Why LLMs?

I wanted to see if using LLMs like GPT could provide more context-aware translations compared to traditional methods. It's not always better, but it can handle some VRChat-specific lingo pretty well!

## What You Need

- A PC running VRChat
- A microphone
- VRChat with OSC enabled
- An OpenAI account (for API access)

## Getting Started

## Running It

1. Start VRChat and enable OSC
2. Populate the `config.toml` file with your OpenAI API key and target language. Make sure it's in the same folder as the executable. I left an example config file in the repo.
3. Run the translator:
   - If using a [pre-built release](https://github.com/d6e/vrchat_osc_llm/releases), just double-click the executable
   - If you've built from source, use `cargo run --release`
4. Start chatting in VRChat!

## Some Cool Things It Does

- Uses a noise gate to ignore background noise
- Waits for pauses in speech before translating
- Shows the "typing" indicator in VRChat while it's working
- Limits API requests to avoid burning through your OpenAI credits too fast

## Known Issues

- Sometimes struggles with very short phrases
- Can be a bit slow if you talk for a long time without pausing
- Occasionally makes weird translations (but that can be funny too)

## Troubleshooting

If it's not working:
1. Double-check your VRChat OSC settings
2. Make sure your mic is working
3. Check your `config.toml` file for typos
4. Verify your OpenAI API key is valid

## Want to Tinker?

Feel free to fork the project and make changes! If you come up with any cool improvements, I'd love to see them. If you want to build from source:

1. Make sure you have Rust installed
2. Clone this repo
3. Run `cargo build --release`

## Disclaimer

This is just a personal project and isn't officially associated with VRChat or OpenAI. Use at your own risk!
