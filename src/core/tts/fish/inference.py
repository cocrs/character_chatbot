# based on https://github.com/fishaudio/fish-speech/blob/main/tools/vqgan/inference.py
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger

from fish_speech.utils.file import AUDIO_EXTENSIONS


def load_model(config_name: str, checkpoint_path: str, device="cuda"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(
        version_base="1.3", config_path="../../../../fish-speech/fish_speech/configs"
    ):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model: {result}")
    return model


class VQGANInference:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        config_name: str,
        checkpoint_path: str,
        device: str,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.config_name = config_name
        self.checkpoint_path = checkpoint_path
        self.device = device

        self.model = load_model(config_name, checkpoint_path, device=device)

    @torch.no_grad()
    def run(self):
        if self.input_path.suffix in AUDIO_EXTENSIONS:
            logger.info(f"Processing in-place reconstruction of {self.input_path}")

            # Load audio
            audio, sr = torchaudio.load(str(self.input_path))
            if audio.shape[0] > 1:
                audio = audio.mean(0, keepdim=True)
            audio = torchaudio.functional.resample(
                audio, sr, self.model.spec_transform.sample_rate
            )

            audios = audio[None].to(self.device)
            logger.info(
                f"Loaded audio with {audios.shape[2] / self.model.spec_transform.sample_rate:.2f} seconds"
            )

            # VQ Encoder
            audio_lengths = torch.tensor(
                [audios.shape[2]], device=self.device, dtype=torch.long
            )
            indices = self.model.encode(audios, audio_lengths)[0][0]

            logger.info(f"Generated indices of shape {indices.shape}")

            # Save indices
            np.save(self.output_path.with_suffix(".npy"), indices.cpu().numpy())
        elif self.input_path.suffix == ".npy":
            logger.info(f"Processing precomputed indices from {self.input_path}")
            indices = np.load(self.input_path)
            indices = torch.from_numpy(indices).to(self.device).long()
            assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
        else:
            raise ValueError(f"Unknown input type: {self.input_path}")

        # Restore
        feature_lengths = torch.tensor([indices.shape[1]], device=self.device)
        fake_audios = self.model.decode(
            indices=indices[None], feature_lengths=feature_lengths
        )
        audio_time = fake_audios.shape[-1] / self.model.spec_transform.sample_rate

        logger.info(
            f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
        )

        # Save audio
        fake_audio = fake_audios[0, 0].float().cpu().numpy()
        sf.write(self.output_path, fake_audio, self.model.spec_transform.sample_rate)
        logger.info(f"Saved audio to {self.output_path}")
