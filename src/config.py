import os
from dotenv import load_dotenv
import torch


class Config:
    def __init__(self):
        load_dotenv()
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.engine_id = os.environ.get("ENGINE_ID")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = "langchain"  # langchain or llama_index
        self.audio_output_path = "output.mp3"


config = Config()
