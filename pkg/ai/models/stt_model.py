import io
import os
import wave

import numpy as np
import requests
import sounddevice as sd
from transformers import pipeline

from pkg.ai.models.aspd_detector import AdvancedSpeechPauseDetector
from pkg.config import HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, device
from pkg.utils import float_to_pcm16


# ===== Base STT =====
class SpeechToTextModel:
    def __init__(self, model: str, device: int = 0) -> None:
        self.device = device
        self.model = model

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError


# ===== Local HuggingFace STT =====
class LocalSpeechToTextModel(SpeechToTextModel):
    def __init__(self, model: str = STT_MODEL_LOCAL, device: int = 0) -> None:
        super().__init__(model, device)
        # load pipeline once (local execution)
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            device=self.device,
        )

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int = 16000) -> str:
        """buffer: numpy array of PCM float32 [-1,1]
        sample_rate: must match pipeline (default 16k).
        """
        result = self.asr({"array": buffer, "sampling_rate": sample_rate})
        return result["text"]


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


# ===== Main test loop =====
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

    if STT_MODE == "local":
        stt = LocalSpeechToTextModel(model=STT_MODEL_LOCAL, device=device)
    else:
        stt = RemoteSpeechToTextModel(model=STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)

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
            else:
                if recording and len(buffer) > 5000:  # Transcribe if speech is long enough
                    audio_np = np.array(buffer, dtype=np.float32)
                    stt.audio_to_text(audio_np, sample_rate=sample_rate)
                    buffer = []  # Clear buffer after transcription

                # If it was recording but the audio is too short, just reset
                if recording:
                    recording = False
                    buffer = []


if __name__ == "__main__":
    main()
