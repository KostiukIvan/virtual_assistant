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
        max_len = self.model.config.max_position_embeddings  # model max tokens/frames
        # Tokenize text
        tokens = self.processor.tokenizer(text).input_ids
        chunks = [tokens[i : i + max_len] for i in range(0, len(tokens), max_len)]

        outputs = []

        for chunk_tokens in chunks:
            inputs = self.processor(tokenizer=chunk_tokens, return_tensors="pt").to(self.device)
            with torch.no_grad():
                speech_chunk = self.model.generate(
                    **inputs,
                    speaker_embeddings=self.speaker_embeddings,
                    vocoder=self.vocoder,
                )
            outputs.append(speech_chunk.cpu().numpy())  # TODO: Potential for speedup !!!

        # Concatenate all chunks into one waveform
        return np.concatenate(outputs, axis=-1)
