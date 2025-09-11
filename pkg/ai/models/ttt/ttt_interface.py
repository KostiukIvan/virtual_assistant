

class TextToTextModel:
    def __init__(self, model: str, device: int = 0) -> None:
        self.device = device
        self.model = model

    def text_to_text(self, message: str) -> str:
        raise NotImplementedError


