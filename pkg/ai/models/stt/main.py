import asyncio
import logging

import numpy as np

import pkg.config as config
from pkg.ai.models.stt.stt_model_selector import STTModelSelector
from pkg.ai.models.utils import mic_producer
from pkg.voice_app.aspd_worker import (
    AdvancedSpeechPauseDetectorAsyncStream,
)

logger = logging.getLogger(__name__)


async def main():
    logger.info("Starting STT test...")
    logger.info("DEVICE:", config.DEVICE_CUDA_OR_CPU)
    stt = STTModelSelector.get_stt_model("small.en")  # "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
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
                logger.info(f"Transcription: {text} (Confidence: {conf})")
                audio_chunks = []  # reset for next chunk

    except KeyboardInterrupt:
        logger.info("\nStopping...")
    finally:
        mic_task.cancel()
        await detector.stop()


if __name__ == "__main__":
    asyncio.run(main())
