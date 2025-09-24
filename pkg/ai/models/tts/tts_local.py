import numpy as np
import torch

# New import for remote TTS audio processing
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

import pkg.config as config
from pkg.ai.models.tts.tts_interface import TextToSpeechModel
from pkg.config import (
    TTS_MODEL,
)


# ===== Local HuggingFace TTS (Corrected with Placeholder) =====
class LocalTextToSpeechModel(TextToSpeechModel):
    def __init__(
        self,
        model: str = TTS_MODEL,
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
        """
        Convert text to speech with safe chunking to avoid max-length errors.
        Returns: numpy array of waveform.
        """
        # Split text into smaller chunks (config.TTS_MAX_CHARS, e.g., 200–400)
        chunks = [text[i : i + config.TTS_MAX_CHARS] for i in range(0, len(text), config.TTS_MAX_CHARS)]
        outputs = []

        for chunk_text in chunks:
            inputs = self.processor(text=chunk_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                speech_chunk = self.model.generate(
                    **inputs,
                    speaker_embeddings=self.speaker_embeddings,
                    vocoder=self.vocoder,
                )
            outputs.append(speech_chunk.cpu().numpy())

        # Concatenate all chunks into one waveform
        return np.concatenate(outputs, axis=-1)
