import numpy as np
from transformers import pipeline

import pkg.config as config
from pkg.ai.models.stt.stt_interface import SpeechToTextModel


# ===== Local HuggingFace STT =====
class LocalSpeechToTextModel(SpeechToTextModel):
    def __init__(self, model: str = config.STT_MODEL_LOCAL) -> None:
        super().__init__(model, config.DEVICE_CUDA_OR_CPU)
        # load pipeline once (local execution)
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            device=self.device,
            generate_kwargs={"language": "english"},
        )

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int) -> str:
        """buffer: numpy array of PCM float32 [-1,1]
        sample_rate: must match pipeline (default 16k).
        """
        result = self.asr({"array": buffer, "sampling_rate": sample_rate})
        return result["text"]
