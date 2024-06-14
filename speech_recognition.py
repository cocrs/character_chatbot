from transformers import WhisperProcessor, WhisperForConditionalGeneration
from io import BytesIO
import soundfile as sf
from pydub import AudioSegment
import librosa

# Initialize Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

def transcribe_audio(audio_data: bytes, mime_type: str) -> str:
    with BytesIO(audio_data) as audio_file:
        audio_file.name = f"input_audio.{mime_type.split('/')[1]}"
        audio_file.seek(0)

        # Load audio using pydub
        audio = AudioSegment.from_file(audio_file)

        # Export to wav format (pydub requires wav format)
        wav_file = BytesIO()
        audio.export(wav_file, format="wav")
        wav_file.seek(0)

        # Read audio data and convert to waveform array
        waveform, sample_rate = sf.read(wav_file)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

        input_features = processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).input_features
        predicted_ids = model.generate(input_features, is_multilingual=True)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]
