import queue
import threading
import time

import numpy as np
import sounddevice as sd

from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream


# ---------------------------
# LocalAudioProducer (modified to publish playback reference)
# ---------------------------
class LocalAudioProducer:
    """Consumes audio frames from an input queue and plays them through the local speakers.
    Additionally, it publishes a copy of each played chunk to a `playback_ref_queue`
    so the AEC has access to the reference (what's being played).
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        playback_ref_queue: queue.Queue,  # will receive copies of frames we play
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        self.input_queue = input_queue
        self.playback_ref_queue = playback_ref_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.is_running = False
        self.thread = None
        self.stream = None

    def _production_loop(self) -> None:
        try:
            frame_size = int(self.sample_rate * 0.03) * self.channels  # 30ms
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=frame_size // self.channels,
            )
            self.stream.start()
            print("LocalAudioProducer output stream started")

            buffer = []  # hold chunks until "L"
            leftover = np.array([], dtype=self.dtype)

            while self.is_running:
                try:
                    data = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue

                if data is None:
                    continue

                audio_chunk = data.get("data")
                event = data.get("event")

                if audio_chunk is not None:
                    buffer.append(audio_chunk)

                print("buffer size:", len(buffer), "event:", event)

                if event == "L" and buffer:
                    # Combine all buffered chunks + leftover
                    audio_chunk = np.concatenate([leftover] + buffer)
                    num_samples = len(audio_chunk)
                    cursor = 0

                    while cursor + frame_size <= num_samples:
                        frame = audio_chunk[cursor : cursor + frame_size]
                        cursor += frame_size

                        self.playback_ref_queue.put_nowait((time.monotonic_ns(), frame.copy()))
                        self.stream.write(frame)

                    # Save any incomplete tail for next iteration
                    leftover = audio_chunk[cursor:]
                    buffer.clear()  # clear once processed

                if audio_chunk is None:
                    # optional end-of-stream marker
                    break

        finally:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
            print("LocalAudioProducer stopped")

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._production_loop, daemon=True)
        self.thread.start()
        print("LocalAudioProducer started")

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("LocalAudioProducer stopped")


def main() -> None:
    """Directly tests the LocalAudioProducer with a single mock audio chunk."""
    audio_queue = queue.Queue()

    audio_producer = LocalAudioProducer(
        input_queue=audio_queue,
    )

    # Generate a mock audio chunk

    # 2. Initialize the audio stream component
    audio_stream = LocalAudioStream(output_queue=audio_queue)

    # 3. Start capturing audio
    audio_stream.start()
    audio_producer.start()

    try:
        # 4. Main loop to consume frames from the queue
        while True:
            pass

    except KeyboardInterrupt:
        pass
    finally:

        audio_producer.stop()
        audio_stream.stop()


if __name__ == "__main__":
    main()
