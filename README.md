# Character Conversational AI with Voice Cloning

##### 🚧 **Work in Progress**

- **Current focus**: Developing a single-agent conversational AI tailored for Japanese.
- **Ongoing efforts**: Optimizing loading and inference speeds.

This project aims to build a multi-agent conversational AI for characters. Allowing users to interact with them or observe the interaction between characters.

## Quick Install
<!-- TODO: edit compose and .env -->

Clone required repositories:

```
git clone https://github.com/cocrs/character_chatbot.git
cd character_chatbot
git clone -b v1.4.2 https://github.com/fishaudio/fish-speech.git
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4/
```

- Install with [pdm](https://pdm-project.org/en/latest/):

```
conda create -n character-chatbot python=3.10
conda activate character-chatbot
pdm install
```

## How to run

- Path of audio for cloning: `voices/ref.wav`
- Edit `prompt_text` in `configs/fish.yaml` for the reference audio

```
bash scripts/run.sh
```

<!-- ## Roadmap

#### Single-Agent Conversational AI (🚧 In Progress)
  - [ ] Chat as a character
  - [ ] TTS

#### Multi-Agent Conversational AI -->

## Credits

- [Chainlit](https://github.com/chainlit/chainlit)
- [Langchain](https://github.com/hwchase17/lanlchain)
- [Fish-speech](https://github.com/fishaudio/fish-speech)
- [Whisper](https://github.com/openai/whisper)
- [Unsloth](https://github.com/unslothai/unsloth)
