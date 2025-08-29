import queue
import threading

import numpy as np
import sounddevice as sd


class LocalAudioStream:
    """Ingests audio from the local microphone and puts the audio chunks (frames)
    into an output queue.
    """

    def __init__(
        self,
        output_queue: queue.Queue,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        """Initializes the LocalAudioStream.

        Args:
            output_queue (queue.Queue): The queue to which audio frames will be sent.
            sample_rate (int): The sample rate of the audio stream.
            frame_duration_ms (int): The duration of each audio chunk in milliseconds.
            channels (int): The number of audio channels.
            dtype (str): The data type of the audio samples (e.g., "float32").

        """
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.channels = channels
        self.dtype = dtype

        # Calculate the number of samples per frame
        self.frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)

        self.is_running = False
        self.thread = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """This function is called by the sounddevice stream for each new audio block.
        It puts the audio data into the output queue.
        """
        if status:
            pass
        self.output_queue.put(indata.copy())

    def _ingestion_loop(self) -> None:
        """The main loop for capturing and processing audio."""
        frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype="float32",
        ) as stream:
            while self.is_running:
                try:
                    # Read a chunk of audio data equal to the frame step size
                    frame_bytes, _ = stream.read(frame_samples)
                    self.output_queue.put(frame_bytes)

                except Exception:
                    break

    def start(self) -> None:
        """Starts the audio ingestion in a separate thread."""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._ingestion_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops the audio ingestion thread."""
        if not self.is_running:
            return

        self.is_running = False
        if self.thread:
            self.thread.join()  # Wait for the thread to finish


def main() -> None:
    """Example of using the LocalAudioStream to capture microphone audio
    and process it from a queue.
    """
    # 1. Create a queue to hold the audio frames
    audio_frames_queue = queue.Queue()

    # 2. Initialize the audio stream component
    audio_stream = LocalAudioStream(output_queue=audio_frames_queue)

    # 3. Start capturing audio
    audio_stream.start()

    try:
        # 4. Main loop to consume frames from the queue
        while True:
            try:
                # Get a frame from the queue
                audio_frames_queue.get(timeout=1.0)

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
