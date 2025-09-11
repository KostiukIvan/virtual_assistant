import numpy as np


class SpeechToTextModel:
    def __init__(self, model: str, device: int = 0) -> None:
        self.device = device
        self.model = model

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError
