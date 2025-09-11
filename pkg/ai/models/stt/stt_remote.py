import io
import os
import wave

import numpy as np
import requests
import sounddevice as sd
from transformers import pipeline

from pkg.ai.models.aspd.aspd_detector import AdvancedSpeechPauseDetector
from pkg.ai.models.stt.stt_interface import SpeechToTextModel
from pkg.config import HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, device
from pkg.utils import float_to_pcm16



# ===== Remote HuggingFace STT (New Class) =====
class RemoteSpeechToTextModel(SpeechToTextModel):
    def __init__(
        self,
        model: str = STT_MODEL_REMOTE,
        hf_token: str | None = None,
    ) -> None:
        super().__init__(model)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            msg = (
                "Hugging Face API token not found. " "Pass it as an argument or set the HF_TOKEN environment variable."
            )
            raise ValueError(
                msg,
            )

        self.api_url = model
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "audio/wav",
        }

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int = 16000) -> str:
        """buffer: numpy array of PCM float32 [-1,1]
        sample_rate: must match model's expected sample rate.
        """
        # 1. Convert float audio to 16-bit PCM bytes, as before.
        pcm_data = float_to_pcm16(buffer)

        # 2. Create a virtual WAV file in memory.
        with io.BytesIO() as wav_file:
            with wave.open(wav_file, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit (2 bytes)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data)
            wav_data = wav_file.getvalue()

        # 3. Send the complete WAV data (header + PCM).
        response = requests.post(self.api_url, headers=self.headers, data=wav_data)

        if response.status_code != 200:
            return f"Error: API returned status {response.status_code} - {response.text}"

        result = response.json()

        if "error" in result:
            if "is currently loading" in result["error"]:
                estimated_time = result.get("estimated_time", 0)
                return f"Model is loading, please try again in {estimated_time:.0f} seconds."
            return f"API Error: {result['error']}"

        return result.get("text", "No transcription available.")


