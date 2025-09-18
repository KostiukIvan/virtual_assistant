import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import pkg.config as config
from pkg.ai.models.stt.stt_interface import SpeechToTextModel

# ----------------- ASR + Confidence -----------------


class Wav2Vec2ASRModel(SpeechToTextModel):
    """Real-time ASR using Wav2Vec2 with utterance-level confidence."""

    def __init__(self, model="facebook/wav2vec2-base-960h"):
        super().__init__(model, config.DEVICE_CUDA_OR_CPU)

        self.processor = Wav2Vec2Processor.from_pretrained(model)
        self.model = Wav2Vec2ForCTC.from_pretrained(model).to(self.device)

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int):
        """
        Args:
            buffer (np.ndarray): PCM float32 [-1,1]
            sample_rate (int): sample rate of buffer
        Returns:
            (text, confidence): text string, confidence float [0..1]
        """
        # Resample if needed
        if sample_rate != 16000:
            import librosa

            buffer = librosa.resample(buffer, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Preprocess
        input_values = self.processor(buffer, sampling_rate=sample_rate, return_tensors="pt").input_values
        input_values = input_values.to(self.device)

        # Forward
        with torch.no_grad():
            logits = self.model(input_values).logits  # (batch, seq_len, vocab)

        # Predicted IDs
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0].lower().strip()

        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        # Gather max prob at each timestep
        max_probs = torch.max(probs, dim=-1).values  # shape: (1, seq_len)
        confidence = float(torch.mean(max_probs))  # average over time
        confidence = min(1.0, max(0.0, confidence))

        return text, confidence
