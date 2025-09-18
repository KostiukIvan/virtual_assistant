import numpy as np

# New imports for remote models
import sounddevice as sd
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel
from pkg.ai.models.tts.tts_remote import RemoteTextToSpeechModel
from pkg.ai.models.ttt.ttt_remote import RemoteTextToTextModel

import pkg.config as config

# Assume these classes are defined in their respective files as before
from pkg.ai.models.stt.whisper import LocalSpeechToTextModel
from pkg.ai.models.tts.tts_local import LocalTextToSpeechModel
from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel


# ===== Main Conversational Loop =====
def main() -> None:
    # Init models
    stt = (
        LocalSpeechToTextModel(config.STT_MODEL)  # drop device if not supported
        if config.STT_MODE == "local"
        else RemoteSpeechToTextModel(config.STT_MODEL_REMOTE, hf_token=config.HF_API_TOKEN)
    )

    ttt = (
        LocalTextToTextModel(config.TTT_MODEL)
        if config.TTT_MODE == "local"
        else RemoteTextToTextModel(config.TTT_MODEL_REMOTE, hf_token=config.HF_API_TOKEN)
    )

    tts = (
        LocalTextToSpeechModel(config.TTS_MODEL)
        if config.TTS_MODE == "local"
        else RemoteTextToSpeechModel(config.TTS_MODEL_REMOTE, hf_token=config.HF_API_TOKEN)
    )

    buffer = []
    FRAME_SAMPLES = config.AUDIO_FRAME_SAMPLES
    SAMPLE_RATE = config.AUDIO_SAMPLE_RATE
    CHUNK_SIZE = SAMPLE_RATE  # ~1s of audio

    with sd.InputStream(
        channels=config.AUDIO_CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=config.AUDIO_DTYPE,
        blocksize=FRAME_SAMPLES,
    ) as stream:
        print("Listening... (Ctrl+C to stop)")
        while True:
            audio_chunk, _ = stream.read(FRAME_SAMPLES)
            buffer.extend(audio_chunk.flatten())

            if len(buffer) >= CHUNK_SIZE:
                audio_np = np.array(buffer, dtype=np.float32)
                buffer = []  # reset buffer

                # STT
                text = stt.audio_to_text(audio_np, SAMPLE_RATE)
                if not text or len(text.strip()) < 2:
                    continue

                print(f"User: {text}")

                # TTT
                reply = ttt.text_to_text(text)
                print(f"Bot: {reply}")

                # TTS
                audio_reply = tts.text_to_speech(reply)

                # Pause mic while speaking (avoid echo capture)
                stream.stop()
                sd.play(audio_reply, samplerate=SAMPLE_RATE)
                sd.wait()
                stream.start()


if __name__ == "__main__":
    main()
