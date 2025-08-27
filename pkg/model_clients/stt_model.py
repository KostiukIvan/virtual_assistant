import sounddevice as sd
import numpy as np
import webrtcvad
from transformers import pipeline
from pkg.utils import float_to_pcm16
from pkg.model_clients.vad_model import VAD


# ===== Base STT =====
class SpeechToTextModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def audio_to_text(self, buffer: np.ndarray) -> str:
        raise NotImplementedError


# ===== Local HuggingFace STT =====
class LocalSpeechToTextModel(SpeechToTextModel):
    def __init__(self, model_name: str = "openai/whisper-small"):
        super().__init__(model_name)
        # load pipeline once (local execution)
        self.asr = pipeline("automatic-speech-recognition", model=self.model_name)

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int = 16000) -> str:
        """
        buffer: numpy array of PCM float32 [-1,1]
        sample_rate: must match pipeline (default 16k)
        """
        result = self.asr({"array": buffer, "sampling_rate": sample_rate})
        return result["text"]



# ===== Main test loop =====
def main():
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_samples = int(sample_rate * frame_duration / 1000)

    vad = VAD(vad_level=3)
    stt = LocalSpeechToTextModel("openai/whisper-small")

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
                    print("📝 Recognized:", text)
                    buffer = []
                    recording = False


if __name__ == "__main__":
    main()