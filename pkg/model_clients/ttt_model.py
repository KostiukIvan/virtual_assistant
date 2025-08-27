import sounddevice as sd
import numpy as np
import webrtcvad
from transformers import pipeline
from pkg.utils import float_to_pcm16
from pkg.model_clients.vad_model import VAD
from pkg.model_clients.stt_model import LocalSpeechToTextModel


# ===== Base TTT =====
class TextToTextModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def text_to_text(self, message: str) -> str:
        raise NotImplementedError


# ===== Local HuggingFace TTT =====
class LocalTextToTextModel(TextToTextModel):
    def __init__(self, model_name: str = "google/flan-t5-small"):
        super().__init__(model_name)
        self.generator = pipeline("text2text-generation", model=self.model_name)

    def text_to_text(self, message: str) -> str:
        output = self.generator(message, max_length=128, clean_up_tokenization_spaces=True)
        return output[0]["generated_text"]


# ===== Main Conversational Loop =====
def main():
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_samples = int(sample_rate * frame_duration / 1000)

    vad = VAD(vad_level=3)
    stt = LocalSpeechToTextModel("openai/whisper-small")
    ttt = LocalTextToTextModel("google/flan-t5-small")

    buffer = []
    recording = False

    print("🎤 Speak into the microphone... (Ctrl+C to stop)")

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_float, _ = stream.read(frame_samples)

            if vad.is_speech(audio_float.flatten(), sample_rate):
                buffer.extend(audio_float.flatten())
                recording = True
                print("...listening...")
            else:
                if recording and len(buffer) > 0:
                    print("🔎 Transcribing...")
                    audio_np = np.array(buffer, dtype=np.float32)
                    text = stt.audio_to_text(audio_np, sample_rate=sample_rate)
                    print("🗣 You said:", text)

                    reply = ttt.text_to_text(text)
                    print("🤖 Bot:", reply)

                    buffer = []
                    recording = False


if __name__ == "__main__":
    main()
