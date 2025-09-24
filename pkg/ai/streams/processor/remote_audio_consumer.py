import asyncio
import json
import logging
import queue

import numpy as np
from fastapi import WebSocket

import pkg.config as config

logger = logging.getLogger(__name__)


class RemoteAudioStreamConsumer:
    """Consumes audio frames and events from a WebSocket and pushes them into a queue.
    Runs as an asyncio task with a synchronous output queue.
    """

    def __init__(self, output_queue: queue.Queue, ws: WebSocket) -> None:
        self.output_queue = output_queue
        self.ws = ws

        self.is_running = False
        self.task: asyncio.Task | None = None

    async def _ingestion_loop(self) -> None:
        """
        The main ingestion loop to be run as an asyncio task.
        It handles WebSocket receive operations and places data into the sync queue.
        """
        try:
            current_buffer = []
            while self.is_running:
                message = await self.ws.receive()

                # Handle disconnects gracefully
                if message["type"] == "websocket.disconnect":
                    logger.info("WebSocket disconnected")
                    break

                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        data = message["bytes"]

                        # Convert to float32 numpy, then extend buffer
                        frames = np.frombuffer(data, dtype=config.AUDIO_DTYPE)
                        current_buffer.extend(frames.flatten())
                        logger.debug("Received %d audio frames (buffer=%d)", len(frames), len(current_buffer))

                    elif "text" in message:
                        message_text = message["text"]
                        try:
                            event_data = json.loads(message_text)
                        except json.JSONDecodeError as e:
                            logger.warning("Invalid JSON message: %s (error=%s)", message_text, e)
                            continue

                        event = event_data.get("event")
                        if event == "s":
                            if current_buffer:
                                self.output_queue.put(
                                    {"data": np.array(current_buffer, dtype=config.AUDIO_DTYPE), "event": "s"}
                                )
                                logger.debug("Pushed %d frames to output_queue (event=s)", len(current_buffer))
                                current_buffer = []
                        elif event == "L":
                            self.output_queue.put({"data": None, "event": "L"})
                            logger.info("End-of-stream event (L) pushed to output_queue")

        except asyncio.CancelledError:
            logger.info("Ingestion loop cancelled")
        except Exception:
            logger.exception("Ingestion loop error")

    def start(self) -> None:
        if self.is_running:
            logger.warning("RemoteAudioStreamConsumer already running, ignoring start()")
            return
        self.is_running = True
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self._ingestion_loop())
        logger.info("RemoteAudioStreamConsumer started")

    async def stop(self) -> None:
        if not self.is_running:
            logger.warning("RemoteAudioStreamConsumer already stopped, ignoring stop()")
            return
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                logger.debug("Task cancelled cleanly")
        logger.info("RemoteAudioStreamConsumer stopped")
