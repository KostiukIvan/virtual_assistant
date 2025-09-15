import numpy as np


class TextToSpeechModel:
    def __init__(self, model: str, device: int = 0) -> None:
        self.device = device
        self.model = model

    def text_to_speech(self, text: str) -> np.ndarray:
        raise NotImplementedError
