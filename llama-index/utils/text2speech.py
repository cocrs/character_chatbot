import torch
from transformers import AutoProcessor, BarkModel
import scipy

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)

voice_preset = "v2/ja_speaker_4"

model.enable_cpu_offload()


def text2speech(text: str, file_path: str):

    inputs = processor(text, voice_preset=voice_preset)
    audio_array = model.generate(**inputs.to(device))
    audio_array = audio_array.cpu().numpy().squeeze()

    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(file_path, rate=sample_rate, data=audio_array)
