import numpy as np
import sounddevice as sd

import pkg.config as config
from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel


# ===== Main test loop =====
def main() -> None:
    print("Starting STT test...")
    print("DEVICE:", config.DEVICE_CUDA_OR_CPU)
    if config.STT_MODE == "local":
        stt = LocalSpeechToTextModel(model=config.STT_MODEL_LOCAL)
    else:
        stt = RemoteSpeechToTextModel(model=config.STT_MODEL_REMOTE, hf_token=config.HF_API_TOKEN)

    buffer = []
    recording = False
    with sd.InputStream(
        channels=config.AUDIO_CHANNELS, samplerate=config.AUDIO_SAMPLE_RATE, dtype=config.AUDIO_DTYPE
    ) as stream:
        while True:
            audio_float, _ = stream.read(config.AUDIO_FRAME_SAMPLES)
            audio_chunk = audio_float.flatten()
            print("Audio chunk received:", audio_chunk.shape)
            buffer.extend(audio_chunk)
            if recording and len(buffer) > 5000:  # Transcribe if speech is long enough
                audio_np = np.array(buffer, dtype=np.float32)
                text = stt.audio_to_text(audio_np, sample_rate=config.AUDIO_SAMPLE_RATE)
                buffer = []  # Clear buffer after transcription
                print(text)

            # If it was recording but the audio is too short, just reset
            if recording:
                recording = False
                buffer = []


if __name__ == "__main__":
    main()
