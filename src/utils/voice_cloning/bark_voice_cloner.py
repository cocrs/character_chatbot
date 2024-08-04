from .bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio

import torchaudio
import torch
import numpy as np

from .hubert.hubert_manager import HuBERTManager
from .hubert.pre_kmeans_hubert import CustomHubert
from .hubert.customtokenizer import CustomTokenizer

from .bark.api import generate_audio
from .bark.generation import (
    SAMPLE_RATE,
    preload_models,
    codec_decode,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
)
from scipy.io.wavfile import write as write_wav
import os

AUDIO_OUTPUT_PATH = "output.mp3"


# Based on https://github.com/serp-ai/bark-with-voice-clone/blob/main/clone_voice.ipynb
class BarkVoiceCloner:
    def __init__(self, voice_name="output"):
        self.voice_name = voice_name  # whatever you want the name of the voice to be
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # download and load all models
        preload_models(
            text_use_gpu=True,
            text_use_small=False,
            coarse_use_gpu=True,
            coarse_use_small=False,
            fine_use_gpu=True,
            fine_use_small=False,
            codec_use_gpu=True,
            force_reload=False,
            path="models",
        )

    # only needs to be called if you want to clone a new voice
    def clone_and_save(self, audio_filepath):
        ref_audio_path = "utils/voice_cloning/bark/assets/prompts/" + self.voice_name + ".npz"

        model = load_codec_model(use_gpu=True if self.device == "cuda" else False)

        # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer

        hubert_manager = HuBERTManager()
        hubert_manager.make_sure_hubert_installed()
        hubert_manager.make_sure_tokenizer_installed(
            "japanese-HuBERT-quantizer_24_epoch.pth",
            "junwchina/bark-voice-cloning-japanese-HuBERT-quantizer",
        )

        # Load the HuBERT model
        hubert_model = CustomHubert(checkpoint_path="data/models/hubert/hubert.pt").to(
            self.device
        )

        # Load the CustomTokenizer model
        tokenizer = CustomTokenizer.load_from_checkpoint(
            "data/models/hubert/tokenizer.pth"
        ).to(
            self.device
        )  # Automatically uses the right layers

        # Load and pre-process the audio waveform
        # the audio you want to clone (under 13 seconds)
        wav, sr = torchaudio.load(audio_filepath)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.to(self.device)

        semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
        semantic_tokens = tokenizer.get_token(semantic_vectors)

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(wav.unsqueeze(0))
        codes = torch.cat(
            [encoded[0] for encoded in encoded_frames], dim=-1
        ).squeeze()  # [n_q, T]

        # move codes to cpu
        codes = codes.cpu().numpy()
        # move semantic tokens to cpu
        semantic_tokens = semantic_tokens.cpu().numpy()

        np.savez(
            ref_audio_path,
            fine_prompt=codes,
            coarse_prompt=codes[:2, :],
            semantic_prompt=semantic_tokens,
        )

    def generate(self, text_prompt):
        # simple generation
        audio_array = generate_audio(
            text_prompt,
            history_prompt=self.voice_name,
            text_temp=0.7,
            waveform_temp=0.7,
        )

        # generation with more control
        x_semantic = generate_text_semantic(
            text_prompt,
            history_prompt=self.voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )

        x_coarse_gen = generate_coarse(
            x_semantic,
            history_prompt=self.voice_name,
            temp=0.7,
            top_k=50,
            top_p=0.95,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            history_prompt=self.voice_name,
            temp=0.5,
        )
        audio_array = codec_decode(x_fine_gen)

        write_wav(AUDIO_OUTPUT_PATH, SAMPLE_RATE, audio_array)
