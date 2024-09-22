import os
from dotenv import load_dotenv
import torch
from dataclasses import dataclass


load_dotenv()


@dataclass
class Config:
    google_api_key: str = os.environ.get("GOOGLE_API_KEY")
    engine_id: str = os.environ.get("ENGINE_ID")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mode: str = "langchain"  # langchain or llama_index

    # for langchain
    document_path: str = "raw_subtitles/16bits_01_JP_Konoha.json"
    use_chat_history: bool = True

    # tts
    tts: str | None = None # bark or fish
    audio_output_path: str = "../output/clone.wav"


config = Config()
