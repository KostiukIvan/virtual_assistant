import io
import os

import numpy as np

# New imports for remote models
import requests
import sounddevice as sd
import torch

# New import for remote TTS audio processing
from scipy.io import wavfile
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from pkg.ai.models.aspd_detector import AdvancedSpeechPauseDetector

# Assume these classes are defined in their respective files as before
from pkg.ai.models.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.ai.models.ttt_model import LocalTextToTextModel, RemoteTextToTextModel
from pkg.config import (
    HF_API_TOKEN,
    STT_MODE,
    STT_MODEL_LOCAL,
    STT_MODEL_REMOTE,
    TTS_MODE,
    TTS_MODEL_LOCAL,
    TTS_MODEL_REMOTE,
    TTT_MODE,
    TTT_MODEL_LOCAL,
    TTT_MODEL_REMOTE,
    device,
)


# ===== Base TTS =====
class TextToSpeechModel:
    def __init__(self, model: str, sample_rate: int, device: int = 0) -> None:
        self.device = device
        self.model = model
        self.sample_rate = sample_rate

    def text_to_speech(self, text: str) -> np.ndarray:
        raise NotImplementedError


# ===== Local HuggingFace TTS (Corrected with Placeholder) =====
class LocalTextToSpeechModel(TextToSpeechModel):
    def __init__(
        self,
        model: str = TTS_MODEL_LOCAL,
        sample_rate: int = 16000,
        device: int = 0,
    ) -> None:
        super().__init__(model, sample_rate, device)
        self.processor = SpeechT5Processor.from_pretrained(model)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model)  # .to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
        )  # .to(self.device)

        # --- FIX: Generate a generic speaker embedding to avoid network errors ---
        # This creates a placeholder tensor. The voice will be robotic but functional.
        self.speaker_embeddings = torch.randn(1, 512)  # .to(self.device)
        # --------------------------------------------------------------------

    def text_to_speech(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt")  # .to(self.device)
        speech_output = self.model.generate(
            **inputs,
            speaker_embeddings=self.speaker_embeddings,
            vocoder=self.vocoder,
        )
        return speech_output.cpu().numpy()


# ===== Remote HuggingFace TTS (New Class) =====
class RemoteTextToSpeechModel(TextToSpeechModel):
    def __init__(
        self,
        model: str = TTS_MODEL_REMOTE,
        sample_rate: int = 22050,
        hf_token: str | None = None,
    ) -> None:
        super().__init__(model, sample_rate)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            msg = "Hugging Face API token not provided."
            raise ValueError(msg)

        self.api_url = model
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

    def text_to_speech(self, text: str) -> np.ndarray:
        """Sends text to the API and returns the audio as a NumPy float array."""
        payload = {"inputs": text}

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            error_info = response.json()
            msg = f"API request failed: {error_info.get('error', 'Unknown error')}"
            raise ConnectionError(
                msg,
            )

        # The successful response is raw audio bytes (a .wav file in memory)
        audio_bytes = response.content

        # Use scipy to read the in-memory WAV file and get the sample rate and data
        rate, data = wavfile.read(io.BytesIO(audio_bytes))
        self.sample_rate = rate  # Update sample rate to the actual rate from the audio file

        # Convert audio data from integer to float format for playback
        if data.dtype == np.int16:
            audio_float = data.astype(np.float32) / 32768.0
        else:
            audio_float = data.astype(
                np.float32,
            )  # Assume it's already in a compatible format if not int16

        return audio_float


# ===== Main Conversational Loop =====
def main() -> None:
    sample_rate = 16000
    frame_duration = 30
    frame_samples = int(sample_rate * frame_duration / 1000)

    detector = AdvancedSpeechPauseDetector(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration,
        vad_level=3,
        short_pause_ms=250,
        long_pause_ms=1000,
    )

    stt = (
        LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device)
        if STT_MODE == "local"
        else RemoteSpeechToTextModel(STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    ttt = (
        LocalTextToTextModel(TTT_MODEL_LOCAL, device=device)
        if TTT_MODE == "local"
        else RemoteTextToTextModel(TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    tts = (
        LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device)
        if TTS_MODE == "local"
        else RemoteTextToSpeechModel(TTS_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    buffer = []
    recording = False

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_chunk, _ = stream.read(frame_samples)
            if detector.is_speech(audio_chunk.flatten()):
                if not recording:
                    pass
                buffer.extend(audio_chunk.flatten())
                recording = True
            elif recording and len(buffer) > 5000:
                audio_np = np.array(buffer, dtype=np.float32)
                text = stt.audio_to_text(audio_np, sample_rate)

                if text and text.strip() and len(text.strip()) > 1:
                    reply = ttt.text_to_text(text)

                    audio_reply = tts.text_to_speech(reply)
                    sd.play(audio_reply, samplerate=tts.sample_rate)
                    sd.wait()
                else:
                    pass

                buffer = []
                recording = False
            elif recording:  # Reset if speech was too short
                buffer = []
                recording = False


if __name__ == "__main__":
    main()
