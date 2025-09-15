import io
import os

import numpy as np

# New imports for remote models
import requests

# New import for remote TTS audio processing
from scipy.io import wavfile

from pkg.ai.models.tts.tts_interface import TextToSpeechModel
from pkg.config import (
    TTS_MODEL_REMOTE,
)


# ===== Remote HuggingFace TTS (New Class) =====
class RemoteTextToSpeechModel(TextToSpeechModel):
    def __init__(
        self,
        model: str = TTS_MODEL_REMOTE,
        hf_token: str | None = None,
    ) -> None:
        super().__init__(model)
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

        # Convert audio data from integer to float format for playback
        if data.dtype == np.int16:
            audio_float = data.astype(np.float32) / 32768.0
        else:
            audio_float = data.astype(
                np.float32,
            )  # Assume it's already in a compatible format if not int16

        return audio_float
