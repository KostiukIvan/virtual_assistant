class TTTModelSelector:
    @staticmethod
    def get_ttt_model(model_name: str):
        if model_name in [
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot-3B",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
        ]:
            from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel

            return LocalTextToTextModel(model=model_name)

        else:
            raise ValueError(f"Unknown STT model: {model_name}")
