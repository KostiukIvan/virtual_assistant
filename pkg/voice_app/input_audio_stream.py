import asyncio
import sys

import sounddevice as sd

import pkg.config as config


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
                print(f"[LocalAudioStream] Status: {status}", file=sys.stderr)
            try:
                self.loop.call_soon_threadsafe(self.output_queue.put_nowait, indata.copy())
            except Exception as e:
                print(f"[LocalAudioStream] Callback error: {e}", file=sys.stderr)

        self.stream = sd.InputStream(
            channels=config.AUDIO_CHANNELS,
            samplerate=config.AUDIO_SAMPLE_RATE,
            dtype=config.AUDIO_DTYPE,
            blocksize=config.AUDIO_FRAME_SAMPLES,
            callback=callback,
        )
        self.stream.start()
        print("LocalAudioStream started.")

    def stop(self):
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            print("LocalAudioStream stopped.")
