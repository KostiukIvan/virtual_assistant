import queue
import threading
import time

import numpy as np
import sounddevice as sd


class LocalAudioStream:
    """Ingests audio from the local microphone and puts the audio chunks (frames)
    into an output queue. (Unchanged behavior except now it writes into mic_queue
    which the AEC may consume.)
    """

    def __init__(
        self,
        output_queue: queue.Queue,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        self.output_queue = output_queue  # mic queue (raw mic frames)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.channels = channels
        self.dtype = dtype

        # number of samples per frame
        self.frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)

        self.is_running = False
        self.thread = None

    def _ingestion_loop(self) -> None:
        frame_samples = self.frame_samples
        try:
            with sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                blocksize=frame_samples,
            ) as stream:
                while self.is_running:
                    try:
                        frame, overflowed = stream.read(frame_samples)
                        # sounddevice.read returns (data, overflowed) in this context
                        # but different versions behave differently; to be robust:
                        if isinstance(frame, tuple) or isinstance(frame, list):
                            frame = np.asarray(frame[0])
                        self.output_queue.put((time.monotonic_ns(), frame.copy()))
                    except Exception as e:
                        print("[LocalAudioStream] read error: %s", e)
                        break
        except Exception as e:
            print("[LocalAudioStream] InputStream error: %s", e)

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._ingestion_loop, daemon=True)
        self.thread.start()
        print("LocalAudioStream started")

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("LocalAudioStream stopped")


def main() -> None:
    """Example of using the LocalAudioStream to capture microphone audio
    and process it from a queue.
    """
    # 1. Create a queue to hold the audio frames
    audio_frames_queue = queue.Queue()

    # 2. Initialize the audio stream component
    audio_stream = LocalAudioStream(
        output_queue=audio_frames_queue, sample_rate=32000, frame_duration_ms=60, channels=1
    )

    # 3. Start capturing audio
    audio_stream.start()

    try:
        # 4. Main loop to consume frames from the queue
        while True:
            try:
                # Get a frame from the queue
                timestamp, frame = audio_frames_queue.get(timeout=1.0)
                print(len(frame), frame.min(), frame.max())

                # In a real application, you would process the frame here.
                # For this demo, we just print a dot to show that frames are being received.

            except queue.Empty:
                # This can happen if the main loop is faster than the audio stream.
                continue

    except KeyboardInterrupt:
        pass
    finally:
        # 5. Stop the audio stream gracefully
        audio_stream.stop()


if __name__ == "__main__":
    # To run this, you need sounddevice and numpy:
    # pip install sounddevice numpy
    main()
