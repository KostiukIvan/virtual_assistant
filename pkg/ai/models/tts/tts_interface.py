import numpy as np


class TextToSpeechModel:
    def __init__(self, model: str, sample_rate: int, device: int = 0) -> None:
        self.device = device
        self.model = model
        self.sample_rate = sample_rate

    def text_to_speech(self, text: str) -> np.ndarray:
        raise NotImplementedError
