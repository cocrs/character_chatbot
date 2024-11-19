import os
from dotenv import load_dotenv
import torch
from dataclasses import dataclass


load_dotenv()


@dataclass
class Config:
    # google_api_key: str = os.environ.get("GOOGLE_API_KEY")
    # engine_id: str = os.environ.get("ENGINE_ID")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mode: str = "langchain"

    # for langchain
    use_chat_history: bool = True

    # tts
    tts: str | None = "fish" # bark or fish
    audio_output_path: str = "../output/clone.wav"


config = Config()
