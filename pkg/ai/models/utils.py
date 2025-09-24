import asyncio
import logging

import sounddevice as sd

import pkg.config as config

logger = logging.getLogger(__name__)


async def mic_producer(queue: asyncio.Queue):
    """Reads audio frames from microphone and puts them into the queue."""
    with sd.InputStream(
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=config.AUDIO_CHANNELS,
        dtype=config.AUDIO_DTYPE,
        blocksize=config.AUDIO_FRAME_SAMPLES,
    ) as stream:
        logger.info("🎙️ Microphone stream started. Speak into the mic...")
        try:
            while True:
                audio_chunk, _ = stream.read(config.AUDIO_FRAME_SAMPLES)
                await queue.put(audio_chunk)
        except asyncio.CancelledError:
            logger.info("Mic producer cancelled.")
