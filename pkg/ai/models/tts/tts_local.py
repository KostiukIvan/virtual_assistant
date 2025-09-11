import io
import os

import numpy as np


import torch

# New import for remote TTS audio processing
from scipy.io import wavfile
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from pkg.ai.models.tts.tts_interface import TextToSpeechModel
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

