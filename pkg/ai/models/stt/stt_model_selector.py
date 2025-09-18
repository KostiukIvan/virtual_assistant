class STTModelSelector:
    @staticmethod
    def get_stt_model(model_name: str):
        if model_name == "openai/whisper-small":
            from pkg.ai.models.stt.whisper import Whisper

            return Whisper(model=model_name)
        elif model_name == "facebook/wav2vec2-base-960h":
            from pkg.ai.models.stt.wave_to_vec import Wav2Vec2ASRModel

            return Wav2Vec2ASRModel(model=model_name)
        elif model_name in ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]:
            from pkg.ai.models.stt.faster_whisper_model import FasterWhisper

            return FasterWhisper(model=model_name)
        else:
            raise ValueError(f"Unknown STT model: {model_name}")
