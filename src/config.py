import os
from dotenv import load_dotenv

import torch

# ******* Load environment variables *******
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ENGINE_ID = os.environ.get("ENGINE_ID")

# ******* Set up audio output path *******
AUDIO_OUTPUT_PATH = "output.mp3"

device = "cuda" if torch.cuda.is_available() else "cpu"
mode = "langchain"  # langchain or llama_index
