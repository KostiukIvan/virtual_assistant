import queue
import threading
import time

import numpy as np
import sounddevice as sd


class LocalAudioProducer:
    """Consumes audio frames from an input queue and plays them through the local speakers."""

    def __init__(
        self,
        input_queue: queue.Queue,
        speak_callback: callable,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        self.input_queue = input_queue
        self.speak_callback = speak_callback
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.is_running = False
        self.thread = None
        self.stream = None

    def _production_loop(self) -> None:
        """The main loop for playing audio from the queue."""
        is_speaking = False
        try:
            # The stream is created here and remains open
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
            )
            self.stream.start()
            while self.is_running:
                try:
                    audio_chunk = self.input_queue.get(timeout=0.1)
                    if not is_speaking:
                        self.speak_callback(True)
                        is_speaking = True
                    self.stream.write(audio_chunk)
                except queue.Empty:
                    if is_speaking:
                        self.speak_callback(False)
                        is_speaking = False
        except Exception:
            pass
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def start(self) -> None:
        """Starts the audio production in a separate thread."""
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._production_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops the audio production thread."""
        if not self.is_running:
            return
        self.is_running = False
        if self.thread:
            self.thread.join()


def main() -> None:
    """Directly tests the LocalAudioProducer with a single mock audio chunk."""
    audio_queue = queue.Queue()

    audio_producer = LocalAudioProducer(
        input_queue=audio_queue,
        speak_callback=lambda is_speaking: print(
            f"[Playback Status: {'SPEAKING' if is_speaking else 'IDLE'}]",
        ),
    )

    # Generate a mock audio chunk
    SAMPLE_RATE = 16000
    DURATION = 2
    FREQUENCY = 440

    t = np.linspace(0.0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    # Generate the sine wave with amplitude between -1.0 and 1.0, standard for float audio
    amplitude = 0.5
    data = amplitude * np.sin(2.0 * np.pi * FREQUENCY * t)

    # --- FIX: Convert the audio to a float32 NumPy array, NOT bytes ---
    audio_chunk_numpy = data.astype(np.float32)

    audio_producer.start_audio_production()
    time.sleep(0.5)

    # --- FIX: Put the NumPy array into the queue ---
    audio_queue.put(audio_chunk_numpy)

    audio_producer.thread.join()


if __name__ == "__main__":
    main()
