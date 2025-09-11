import numpy as np
import sounddevice as sd

from pkg.ai.models.aspd.aspd_detector import AdvancedSpeechPauseDetector
from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel
from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel
from pkg.ai.models.ttt.ttt_remote import RemoteTextToTextModel
from pkg.config import (
    HF_API_TOKEN,
    STT_MODE,
    STT_MODEL_LOCAL,
    STT_MODEL_REMOTE,
    TTT_MODE,
    TTT_MODEL_LOCAL,
    TTT_MODEL_REMOTE,
    device,
)


# ===== Main Conversational Loop =====
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

    # Initialize Speech-to-Text model based on config
    if STT_MODE == "local":
        stt = LocalSpeechToTextModel(model=STT_MODEL_LOCAL, device=device)
    else:
        stt = RemoteSpeechToTextModel(
            model_name=STT_MODEL_REMOTE,
            hf_token=HF_API_TOKEN,
        )

    # Initialize Text-to-Text model based on config
    if TTT_MODE == "local":
        ttt = LocalTextToTextModel(model=TTT_MODEL_LOCAL, device=device)
    else:
        ttt = RemoteTextToTextModel(model=TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)

    buffer = []
    recording = False

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_float, _ = stream.read(frame_samples)
            audio_chunk = audio_float.flatten()

            if detector.is_speech(audio_chunk):
                if not recording:
                    pass
                buffer.extend(audio_chunk)
                recording = True
            elif recording and len(buffer) > 5000:
                audio_np = np.array(buffer, dtype=np.float32)
                text = stt.audio_to_text(audio_np, sample_rate=sample_rate)

                if text and text.strip() and len(text.strip()) > 1:
                    ttt.text_to_text(text)
                else:
                    pass

                buffer = []
                recording = False
            elif recording:
                buffer = []
                recording = False


if __name__ == "__main__":
    main()
