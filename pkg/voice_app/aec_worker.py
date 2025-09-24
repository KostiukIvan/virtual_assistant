import asyncio
import logging

import pkg.config as config

logger = logging.getLogger(__name__)


class AECWorker:
    """Async AEC worker that outputs mic frames only when the speaker is silent."""

    def __init__(self, mic_queue, playback_ref_queue, output_queue):
        self.mic_queue = mic_queue
        self.playback_ref_queue = playback_ref_queue
        self.output_queue = output_queue

        self.running = True
        self._task: asyncio.Task | None = None

    async def run(self):
        try:
            while self.running:

                # Drain playback queue
                try:
                    sound_frame = self.playback_ref_queue.get_nowait()
                    await asyncio.sleep(config.AUDIO_FRAME_DURATION_MS / 1000)  # tune as needed
                except asyncio.QueueEmpty:
                    sound_frame = None

                # Handle mic frame
                try:
                    mic_frame = self.mic_queue.get_nowait()
                except asyncio.QueueEmpty:
                    mic_frame = None

                if mic_frame is not None and sound_frame is None:
                    try:
                        await self.output_queue.put(mic_frame)
                    except asyncio.QueueFull:
                        # Drop frame if output is overloaded
                        pass

                await asyncio.sleep(0.005)  # tune as needed
        except asyncio.CancelledError:
            logger.info("AEC worker was cancelled.")
        except Exception:
            logger.exception("AEC got exception")

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
