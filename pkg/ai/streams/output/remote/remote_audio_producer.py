import asyncio
import queue

import numpy as np
from fastapi import WebSocket


class RemoteAudioStreamProducer:
    """Consumes audio frames from an input queue and streams them to the WebSocket.
    Also pushes copies of frames into playback_ref_queue for AEC.
    Runs as a background asyncio Task with start/stop controls.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        frame_duration_ms: int = 30,
        sample_rate: int = 16000,
        channels: int = 1,
        ws: WebSocket = None,
        dtype: str = "float32",
    ) -> None:
        self.input_queue = input_queue
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.ws = ws

        self.is_running = False
        self.task: asyncio.Task | None = None

    async def _production_loop(self) -> None:
        try:
            frame_size = int(self.sample_rate * self.frame_duration_ms * self.channels / 1000)  # 30 ms
            leftover = np.array([], dtype=self.dtype)
            buffer = []

            print("RemoteAudioStreamProducer loop started")

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

                    while cursor + frame_size <= num_samples:
                        frame = audio_chunk[cursor : cursor + frame_size]
                        cursor += frame_size

                        # send to remote client
                        await self.ws.send_text(frame.tobytes())

                    # keep tail for next cycle
                    leftover = audio_chunk[cursor:]
                    buffer.clear()

        except asyncio.CancelledError:
            print("[RemoteAudioStreamProducer] Cancelled")
        except Exception as e:
            print("[RemoteAudioStreamProducer] Error:", e)
        finally:
            print("RemoteAudioStreamProducer stopped")

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        loop = asyncio.get_event_loop()
        self.task = loop.create_task(self._production_loop())
        print("RemoteAudioStreamProducer started")

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
        print("RemoteAudioStreamProducer stopped")
