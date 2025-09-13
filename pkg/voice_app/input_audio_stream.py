import asyncio
import sys
import time

import sounddevice as sd


class LocalAudioStream:
    """An asynchronous wrapper for sounddevice's input stream."""

    def __init__(
        self,
        output_queue: asyncio.Queue,
        sample_rate=16000,
        frame_duration_ms=30,
        channels=1,
        dtype="float32",
    ):
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.stream = None
        self.loop = None

    def start(self):
        self.loop = asyncio.get_running_loop()

        def callback(indata, frames, callback_time, status):
            if status:
                print(f"[LocalAudioStream] Status: {status}", file=sys.stderr)
            # This is the crucial part: use threadsafe to put data into the queue from the callback.
            self.loop.call_soon_threadsafe(self.output_queue.put_nowait, (time.monotonic_ns(), indata.copy()))

        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype=self.dtype,
            blocksize=self.frame_samples,
            callback=callback,
        )
        self.stream.start()
        print("LocalAudioStream started.")

    def stop(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            print("LocalAudioStream stopped.")
