import os

import numpy as np
import requests
import sounddevice as sd
from transformers import pipeline

from pkg.ai.models.aspd_detector import AdvancedSpeechPauseDetector
from pkg.ai.models.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.config import (
    HF_API_TOKEN,
    STT_MODE,
    STT_MODEL_LOCAL,
    STT_MODEL_REMOTE,
    TTT_MODE,
    TTT_MODEL_LOCAL,
    TTT_MODEL_REMOTE,
    device,
)


# ===== Base TTT =====
class TextToTextModel:
    def __init__(self, model: str, device: int = 0) -> None:
        self.device = device
        self.model = model

    def text_to_text(self, message: str) -> str:
        raise NotImplementedError


# ===== Local HuggingFace TTT =====
class LocalTextToTextModel(TextToTextModel):
    def __init__(self, model: str = TTT_MODEL_LOCAL, device: int = 0) -> None:
        super().__init__(model, device)
        self.generator = pipeline(
            "text2text-generation",
            model=self.model,
            device=self.device,
        )

    def text_to_text(self, message: str) -> str:
        output = self.generator(
            message,
            max_length=128,
            clean_up_tokenization_spaces=True,
        )
        return output[0]["generated_text"]


# ===== Remote HuggingFace TTT (New Class) =====
class RemoteTextToTextModel(TextToTextModel):
    def __init__(
        self,
        model: str = TTT_MODEL_REMOTE,
        hf_token: str | None = None,
    ) -> None:
        # We don't need the 'device' parameter for remote models
        super().__init__(model)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            msg = "Hugging Face API token not found. " "Import it from config or set the HF_TOKEN environment variable."
            raise ValueError(
                msg,
            )

        self.api_url = model
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

    def text_to_text(self, message: str) -> str:
        """Sends text to the Hugging Face Inference API for a response."""
        payload = {
            "inputs": message,
            "parameters": {
                "max_new_tokens": 128,  # Limit the length of the reply
                "return_full_text": False,  # Only get the model's reply
            },
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            return f"Error: API returned status {response.status_code} - {response.text}"

        result = response.json()

        # Handle potential errors from the API
        if "error" in result:
            if "is currently loading" in result["error"]:
                estimated_time = result.get("estimated_time", 0)
                return f"Model is loading, please try again in {estimated_time:.0f} seconds."
            return f"API Error: {result['error']}"

        # Parse the successful response
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        return "Error: Could not parse the API response."


# ===== Main Conversational Loop =====
def main() -> None:
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_samples = int(sample_rate * frame_duration / 1000)

    detector = AdvancedSpeechPauseDetector(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration,
        vad_level=3,
        short_pause_ms=250,
        long_pause_ms=1000,
    )

    # Initialize Speech-to-Text model based on config
    if STT_MODE == "local":
        stt = LocalSpeechToTextModel(model=STT_MODEL_LOCAL, device=device)
    else:
        stt = RemoteSpeechToTextModel(
            model_name=STT_MODEL_REMOTE,
            hf_token=HF_API_TOKEN,
        )

    # Initialize Text-to-Text model based on config
    if TTT_MODE == "local":
        ttt = LocalTextToTextModel(model=TTT_MODEL_LOCAL, device=device)
    else:
        ttt = RemoteTextToTextModel(model=TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)

    buffer = []
    recording = False

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_float, _ = stream.read(frame_samples)
            audio_chunk = audio_float.flatten()

            if detector.is_speech(audio_chunk):
                if not recording:
                    pass
                buffer.extend(audio_chunk)
                recording = True
            elif recording and len(buffer) > 5000:
                audio_np = np.array(buffer, dtype=np.float32)
                text = stt.audio_to_text(audio_np, sample_rate=sample_rate)

                if text and text.strip() and len(text.strip()) > 1:
                    ttt.text_to_text(text)
                else:
                    pass

                buffer = []
                recording = False
            elif recording:
                buffer = []
                recording = False


if __name__ == "__main__":
    main()
