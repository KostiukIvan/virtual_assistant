import asyncio
import logging
import queue

import numpy as np
from fastapi import WebSocket

import pkg.config as config

logger = logging.getLogger(__name__)


class RemoteAudioStreamProducer:
    """Consumes audio frames from an input queue and streams them to the WebSocket.
    Also pushes copies of frames into playback_ref_queue for AEC.
    Runs as a background asyncio Task with start/stop controls.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        ws: WebSocket = None,
    ) -> None:
        self.input_queue = input_queue
        self.ws = ws

        self.is_running = False
        self.task: asyncio.Task | None = None

    async def _production_loop(self) -> None:
        try:
            leftover = np.array([], dtype=config.AUDIO_DTYPE)
            buffer = []

            logger.info("RemoteAudioStreamProducer loop started")

            while self.is_running:
                try:
                    data = self.input_queue.get(timeout=1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if data is None:
                    continue

                audio_chunk = data.get("data")
                event = data.get("event")

                if audio_chunk is not None:
                    buffer.append(audio_chunk)

                if event == "L" and buffer:
                    # TODO: Think about to move this logic to CLIENT side
                    # Merge chunks + leftover
                    audio_chunk = np.concatenate([leftover] + buffer)
                    num_samples = len(audio_chunk)
                    cursor = 0

                    while cursor + config.AUDIO_FRAME_SAMPLES <= num_samples:
                        frame = audio_chunk[cursor : cursor + config.AUDIO_FRAME_SAMPLES]
                        cursor += config.AUDIO_FRAME_SAMPLES

                        # send to remote client
                        await self.ws.send_text(frame.tobytes())

                    # keep tail for next cycle
                    leftover = audio_chunk[cursor:]
                    buffer.clear()

        except asyncio.CancelledError:
            logger.info("Cancelled")
        except Exception:
            logger.exception("[RemoteAudioStreamProducer] Error:")
        finally:
            logger.info("RemoteAudioStreamProducer stopped")

    def start(self) -> None:
        if self.is_running:
            logger.warning("RemoteAudioStreamProducer already started")
            return
        self.is_running = True
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self._production_loop())
        logger.info("RemoteAudioStreamProducer started")

    async def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("RemoteAudioStreamProducer stopped")
