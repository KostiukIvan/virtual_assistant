import asyncio
import logging

import sounddevice as sd

import pkg.config as config

logger = logging.getLogger(__name__)


class LocalAudioStream:
    """An asynchronous wrapper for sounddevice's input stream."""

    def __init__(
        self,
        output_queue: asyncio.Queue,
    ):
        self.output_queue = output_queue
        self.stream = None
        self.loop = None

    def start(self):
        self.loop = asyncio.get_running_loop()

        def callback(indata, frames, callback_time, status):
            if status:
                logger.info(f"[LocalAudioStream] Status: {status}")
            try:
                self.loop.call_soon_threadsafe(self.output_queue.put_nowait, indata.copy())
            except Exception:
                logger.exception("[LocalAudioStream] Callback error")

        self.stream = sd.InputStream(
            channels=config.AUDIO_CHANNELS,
            samplerate=config.AUDIO_SAMPLE_RATE,
            dtype=config.AUDIO_DTYPE,
            blocksize=config.AUDIO_FRAME_SAMPLES,
            callback=callback,
        )
        self.stream.start()
        logger.info("LocalAudioStream started.")

    def stop(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            logger.info("LocalAudioStream stopped.")
