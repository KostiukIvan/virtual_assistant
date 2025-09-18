import numpy as np
from faster_whisper import WhisperModel

import pkg.config as config
from pkg.ai.models.stt.stt_interface import SpeechToTextModel


class FasterWhisper(SpeechToTextModel):
    def __init__(self, model: str = "large-v3") -> None:
        super().__init__(model, config.DEVICE_CUDA_OR_CPU)

        device = "cuda" if config.DEVICE_CUDA_OR_CPU == "cuda" else "cpu"
        # Use float16 on GPU, int8 on CPU for efficiency
        compute_type = "float16" if device == "cuda" else "int8"

        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int):
        """
        Convert audio buffer to text with confidence.

        Args:
            buffer: np.ndarray (float32 PCM samples)
            sample_rate: int, sampling rate of buffer

        Returns:
            (text: str, confidence: float)
        """
        # Run transcription (beam search improves confidence reliability)
        segments, info = self.model.transcribe(
            buffer, beam_size=5, language="en"  # force English; remove if you want auto-detect
        )

        text = []
        confidences = []

        for seg in segments:
            text.append(seg.text)
            if seg.avg_logprob is not None:
                # Map avg_logprob → [0,1] using sigmoid
                conf = 1 / (1 + np.exp(-seg.avg_logprob))
                confidences.append(conf)

        full_text = " ".join(text).strip()
        confidence = float(np.mean(confidences)) if confidences else 0.0

        return full_text, confidence
