import asyncio

import sounddevice as sd

import pkg.config as config


class LocalAudioProducer:
    """Async worker that plays audio frames from an async queue to the speakers."""

    def __init__(self, input_queue: asyncio.Queue):
        self.input_queue = input_queue
        self.task: asyncio.Task | None = None

    async def _loop(self):
        try:
            with sd.OutputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                channels=config.AUDIO_CHANNELS,
                dtype=config.AUDIO_DTYPE,
                blocksize=config.AUDIO_FRAME_SAMPLES,
            ) as stream:
                while True:
                    frame = await self.input_queue.get()

                    if frame is None:
                        continue

                    try:
                        stream.write(frame)
                    except Exception as e:
                        print(f"[LocalAudioProducer] Stream write error: {e}")
        except asyncio.CancelledError:
            print("[LocalAudioProducer] Playback loop cancelled.")
            raise

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._loop())
            print("LocalAudioProducer started.")

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        print("LocalAudioProducer stopped.")
