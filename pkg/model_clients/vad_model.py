from pkg.utils import float_to_pcm16
import sounddevice as sd
import numpy as np
import webrtcvad
import struct
import time


# --- VAD class (from above) ---
class VoiceActiveDetection:
    def __init__(self):
        pass
    
    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        raise NotImplementedError


class VAD(VoiceActiveDetection):
    def __init__(self, vad_level: int = 3):
        super().__init__()
        self.vad_level = vad_level
        self.client: webrtcvad.Vad = webrtcvad.Vad(self.vad_level)

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        try:
            pcm16 = float_to_pcm16(frame)
            return self.client.is_speech(pcm16, sample_rate)
        except Exception as e:
            print("VAD error: ", str(e))
            return False

# --- main test loop ---
def main():
    sample_rate = 16000  # must be 8k/16k/32k/48k
    frame_duration = 30  # ms (10/20/30 allowed)
    vad = VAD(vad_level=3, sample_rate=sample_rate)

    frame_samples = int(sample_rate * frame_duration / 1000)

    print("🎤 Speak into the microphone... (Ctrl+C to stop)")
    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_float, _ = stream.read(frame_samples)
            print(audio_float)
            is_speech = vad.is_speech(audio_float)
            print("Speech" if is_speech else "Silence")


if __name__ == "__main__":
    main()