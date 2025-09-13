import asyncio

import sounddevice as sd


class LocalAudioProducer:
    """A worker that plays audio frames from an async queue to the speakers."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
    ):
        self.input_queue = input_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.task = None
        self.running = False

    async def _loop(self):
        frame_size = int(self.sample_rate * 0.03) * self.channels
        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=frame_size // self.channels,
        ) as stream:
            while self.running:
                frame = await self.input_queue.get()
                if frame is None:
                    continue
                stream.write(frame)

    def start(self):
        self.running = True
        self.task = asyncio.create_task(self._loop())
        print("LocalAudioProducer started.")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print("LocalAudioProducer stopped.")
