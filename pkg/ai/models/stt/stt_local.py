import numpy as np
from transformers import pipeline

from pkg.ai.models.stt.stt_interface import SpeechToTextModel
from pkg.config import STT_MODEL_LOCAL


# ===== Local HuggingFace STT =====
class LocalSpeechToTextModel(SpeechToTextModel):
    def __init__(self, model: str = STT_MODEL_LOCAL, device: int = 0) -> None:
        super().__init__(model, device)
        # load pipeline once (local execution)
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            device=self.device,
            generate_kwargs={"language": "english"},
        )

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int = 16000) -> str:
        """buffer: numpy array of PCM float32 [-1,1]
        sample_rate: must match pipeline (default 16k).
        """
        result = self.asr({"array": buffer, "sampling_rate": sample_rate})
        return result["text"]
