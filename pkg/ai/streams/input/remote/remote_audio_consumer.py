import asyncio
import json
import queue
import time

import numpy as np
from fastapi import WebSocket


class RemoteAudioStreamConsumer:
    """Consumes audio frames from a WebSocket and pushes them into a queue.
    Runs as a background asyncio Task with start/stop controls.
    """

    def __init__(self, output_queue: queue.Queue, ws: WebSocket, dtype: str = "float32") -> None:
        self.output_queue = output_queue
        self.ws = ws
        self.dtype = dtype

        self.is_running = False
        self.task: asyncio.Task | None = None

    async def _ingestion_loop(self) -> None:
        try:
            while self.is_running:
                msg = await self.ws.receive_text()
                frame = json.loads(msg)
                frame = np.array(frame, dtype=self.dtype)
                self.output_queue.put((time.monotonic_ns(), frame.copy()))
        except asyncio.CancelledError:
            print("[RemoteAudioStreamConsumer] Cancelled")
        except Exception as e:
            print("[RemoteAudioStreamConsumer] Error:", e)

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self._ingestion_loop())
        print("RemoteAudioStreamConsumer started")

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
        print("RemoteAudioStreamConsumer stopped")
