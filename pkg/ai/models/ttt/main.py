import numpy as np
import sounddevice as sd

import pkg.config as config
from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel
from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel
from pkg.ai.models.ttt.ttt_remote import RemoteTextToTextModel

# ===== Main Conversational Loop =====


def main() -> None:
    # Initialize STT
    if config.STT_MODE == "local":
        stt = LocalSpeechToTextModel(model=config.STT_MODEL_LOCAL)
    else:
        stt = RemoteSpeechToTextModel(
            model_name=config.STT_MODEL_REMOTE,
            hf_token=config.HF_API_TOKEN,
        )

    # Initialize TTT
    if config.TTT_MODE == "local":
        ttt = LocalTextToTextModel(model=config.TTT_MODEL_LOCAL)
    else:
        ttt = RemoteTextToTextModel(
            model=config.TTT_MODEL_REMOTE,
            hf_token=config.HF_API_TOKEN,
        )

    buffer = []
    FRAME_SAMPLES = config.AUDIO_FRAME_SAMPLES
    SAMPLE_RATE = config.AUDIO_SAMPLE_RATE

    # target: ~1 sec audio per STT call
    CHUNK_SIZE = SAMPLE_RATE

    with sd.InputStream(
        channels=config.AUDIO_CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype=config.AUDIO_DTYPE,
        blocksize=FRAME_SAMPLES,
    ) as stream:
        print("Listening... (Ctrl+C to stop)")
        while True:
            audio_float, _ = stream.read(FRAME_SAMPLES)
            buffer.extend(audio_float.flatten())

            if len(buffer) >= CHUNK_SIZE:
                audio_np = np.array(buffer, dtype=np.float32)
                buffer = []  # reset buffer

                # Run STT
                text = stt.audio_to_text(audio_np, sample_rate=SAMPLE_RATE)
                if text and text.strip():
                    print(f"User: {text}")
                    reply = ttt.text_to_text(text)
                    print(f"Bot: {reply}")


if __name__ == "__main__":
    main()
