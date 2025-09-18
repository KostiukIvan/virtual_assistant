import asyncio

import numpy as np

import pkg.config as config
from pkg.ai.models.stt.stt_model_selector import STTModelSelector
from pkg.ai.models.ttt.ttt_model_selector import TTTModelSelector
from pkg.ai.models.utils import mic_producer
from pkg.voice_app.aspd_worker import (
    AdvancedSpeechPauseDetectorAsyncStream,
)


async def main():
    print("Starting STT test...")
    print("DEVICE:", config.DEVICE_CUDA_OR_CPU)
    stt = STTModelSelector.get_stt_model("small.en")  # "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
    ttt = TTTModelSelector.get_stt_model(
        "facebook/blenderbot-400M-distill"
    )  # "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"
    mic_queue = asyncio.Queue(maxsize=1)
    event_queue = asyncio.Queue(maxsize=1)

    detector = AdvancedSpeechPauseDetectorAsyncStream(mic_queue, event_queue)
    detector.start()

    mic_task = asyncio.create_task(mic_producer(mic_queue))

    try:
        audio_chunks = []
        while True:
            data = await event_queue.get()
            if data["event"] == "p":  # Long pause detected
                chunk = data["data"]
                audio_chunks.extend(chunk)
            elif data["event"] == "s" or data["event"] == "L":  # Long pause detected
                chunk = data["data"]

                text, conf = stt.audio_to_text(np.array(audio_chunks).flatten(), sample_rate=config.AUDIO_SAMPLE_RATE)
                if conf > 0.3:
                    response = ttt.text_to_text(text)
                    print(f"TTT Response: {response}")

                print(f"Transcription: {text} (Confidence: {conf})")
                audio_chunks = []  # reset for next chunk

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic_task.cancel()
        await detector.stop()


if __name__ == "__main__":
    asyncio.run(main())
