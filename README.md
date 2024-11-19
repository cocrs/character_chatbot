# Character Conversational AI with Voice Cloning

This project aims to build a multi-agent conversational AI for characters. Allowing users to interact with them or observe the interaction between characters.

##### 🚧 In Progress: Single-Agent Conversational AI

## Quick Install

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