import numpy as np
import sounddevice as sd

from pkg.ai.models.aspd.aspd_detector import AdvancedSpeechPauseDetector
from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel
from pkg.config import HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, device


# ===== Main test loop =====
def main() -> None:
    sample_rate = 16000
    frame_duration = 30  # ms
    frame_samples = int(sample_rate * frame_duration / 1000)

    detector = AdvancedSpeechPauseDetector(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration,
        vad_level=3,
        short_pause_ms=250,
        long_pause_ms=1000,
    )

    if STT_MODE == "local":
        stt = LocalSpeechToTextModel(model=STT_MODEL_LOCAL, device=device)
    else:
        stt = RemoteSpeechToTextModel(model=STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)

    buffer = []
    recording = False

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_float, _ = stream.read(frame_samples)
            audio_chunk = audio_float.flatten()

            buffer.extend(audio_chunk)
            if not detector.is_speech(audio_chunk):
                if recording and len(buffer) > 5000:  # Transcribe if speech is long enough
                    audio_np = np.array(buffer, dtype=np.float32)
                    text = stt.audio_to_text(audio_np, sample_rate=sample_rate)
                    buffer = []  # Clear buffer after transcription
                    print(text)

                # If it was recording but the audio is too short, just reset
                if recording:
                    recording = False
                    buffer = []


if __name__ == "__main__":
    main()
