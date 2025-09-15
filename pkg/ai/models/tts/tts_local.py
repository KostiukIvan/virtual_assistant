import numpy as np
import torch

# New import for remote TTS audio processing
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

import pkg.config as config
from pkg.ai.models.tts.tts_interface import TextToSpeechModel
from pkg.config import (
    TTS_MODEL_LOCAL,
)


# ===== Local HuggingFace TTS (Corrected with Placeholder) =====
class LocalTextToSpeechModel(TextToSpeechModel):
    def __init__(
        self,
        model: str = TTS_MODEL_LOCAL,
    ) -> None:
        super().__init__(model, config.DEVICE_CUDA_OR_CPU)
        self.processor = SpeechT5Processor.from_pretrained(model)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
        ).to(self.device)

        # --- FIX: Generate a generic speaker embedding to avoid network errors ---
        # This creates a placeholder tensor. The voice will be robotic but functional.
        self.speaker_embeddings = torch.randn(1, 512).to(self.device)
        # --------------------------------------------------------------------

    def text_to_speech(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        speech_output = self.model.generate(
            **inputs,
            speaker_embeddings=self.speaker_embeddings,
            vocoder=self.vocoder,
        )
        return speech_output.cpu().numpy()
