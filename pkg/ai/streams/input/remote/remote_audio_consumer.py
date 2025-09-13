import asyncio
import json
import queue

import numpy as np
from fastapi import WebSocket


class RemoteAudioStreamConsumer:
    """Consumes audio frames and events from a WebSocket and pushes them into a queue.
    Runs as an asyncio task with a synchronous output queue.
    """

    def __init__(self, output_queue: queue.Queue, ws: WebSocket, dtype: str = "float32") -> None:
        self.output_queue = output_queue
        self.ws = ws
        self.dtype = dtype

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
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Handle binary data (audio frames)
                        data = message["bytes"]
                        frames = np.frombuffer(data, dtype=self.dtype)
                        current_buffer.extend(frames.flatten())
                    elif "text" in message:
                        # Handle text data (JSON event)
                        message_text = message["text"]
                        event_data = json.loads(message_text)
                        event = event_data.get("event")
                        if event in "s":
                            if current_buffer:
                                self.output_queue.put(
                                    {"data": np.array(current_buffer, dtype=np.float32), "event": "s"}
                                )
                                current_buffer = []
                        if event in "L":
                            self.output_queue.put({"data": None, "event": "L"})

                        print(f"[RemoteAudioStreamConsumer] Received event: {event}")

        except asyncio.CancelledError:
            print("[RemoteAudioStreamConsumer] Ingestion loop cancelled.")
        except Exception as e:
            print(f"[RemoteAudioStreamConsumer] Ingestion loop error: {e}")

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
