from transformers import Pipeline, pipeline

from pkg.ai.models.ttt.ttt_interface import TextToTextModel
from pkg.config import (
    TTT_MODEL_LOCAL,
)


class LocalTextToTextModel(TextToTextModel):
    def __init__(
        self,
        model: str = TTT_MODEL_LOCAL,
        device: int = 0,
        max_length: int = 256,
        num_return_sequences: int = 1,
    ) -> None:
        super().__init__(model, device)
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.generator: Pipeline | None = None

    def _load_pipeline(self) -> None:
        """Lazy load the HuggingFace pipeline (only once)."""
        if self.generator is None:
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                device=self.device,
            )

    def text_to_text(
        self,
        message: str | list[str],
        **generate_kwargs,
    ) -> str | list[str]:
        """
        Run text-to-text inference.

        Args:
            message: A single string or list of strings.
            **generate_kwargs: Extra HF generation params (e.g., temperature, top_p).

        Returns:
            Generated text or list of texts.
        """
        self._load_pipeline()

        try:
            outputs = self.generator(
                message,
                max_length=self.max_length,
                num_return_sequences=self.num_return_sequences,
                clean_up_tokenization_spaces=True,
                **generate_kwargs,
            )
            if isinstance(message, str):
                return outputs[0]["generated_text"]
            else:
                return [o[0]["generated_text"] for o in outputs]

        except Exception as e:
            # Graceful fallback
            return f"[LocalTextToTextModel error: {e}]"
