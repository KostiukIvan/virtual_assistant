import pyaudio
import numpy as np
import threading
import queue
import time
import collections
from pkg.model_clients.vad_model import VAD
import sounddevice as sd


class VoiceFrameIngestor:
    """
    Ingests audio from the microphone, segments it into frames, performs VAD,
    and calls a callback when a speech pause is detected.
    """
    def __init__(self,
                 vad: object,
                 stream_queue: queue.Queue,
                 pause_callback: callable,
                 frame_ms: int = 30,
                 overlap_ms: int = 10,
                 pause_threshold_ms: int = 300,
                 sample_rate: int = 16000):
        """
        Initializes the VoiceFrameIngestor.

        Args:
            vad (object): An object with an `is_speech(frame)` method.
            stream_queue (queue.Queue): Queue to put the audio frames into.
            pause_callback (callable): Function to call when speech pauses.
            frame_ms (int): The duration of each audio frame in milliseconds.
            overlap_ms (int): The overlap between consecutive frames in milliseconds.
            pause_threshold_ms (int): Milliseconds of silence to trigger the pause callback.
            rate (int): The sample rate of the audio.
        """
        self.vad = vad
        self.stream_queue = stream_queue
        self.pause_callback = pause_callback
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_ms
        self.overlap_duration_ms = overlap_ms

        # Calculate frame and overlap sizes in samples
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        self.overlap_size = int(self.sample_rate * (self.overlap_duration_ms / 1000.0))
        self.step_size = self.frame_size - self.overlap_size

        self.is_running = False
        self.thread = None

        # State management for pause detection
        self.is_speaking = False
        self.pause_threshold_frames = pause_threshold_ms // (frame_ms - overlap_ms)
        self.silent_frames_count = 0

        self.p = pyaudio.PyAudio()

        print("[AudioProcessor Config]")
        print(f"  sample_rate: {self.sample_rate}")
        print(f"  frame_duration_ms: {self.frame_duration_ms}")
        print(f"  overlap_duration_ms: {self.overlap_duration_ms}")
        print(f"  frame_size (samples): {self.frame_size}")
        print(f"  overlap_size (samples): {self.overlap_size}")
        print(f"  step_size (samples): {self.step_size}")
        print(f"  pause_threshold_ms: {pause_threshold_ms}")
        print(f"  pause_threshold_frames: {self.pause_threshold_frames}")
        print()

    def start(self):
        """Starts the audio ingestion in a separate thread."""
        if self.is_running:
            print("Ingestor is already running.")
            return

        print("Starting voice frame ingestor...")
        self.is_running = True
        self.thread = threading.Thread(target=self._ingestion_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the audio ingestion thread."""
        print("Stopping voice frame ingestor...")
        self.is_running = False
        if self.thread:
            self.thread.join() # Wait for the thread to finish
        self.p.terminate()
        print("Ingestor stopped.")

    def _ingestion_loop(self):
        """The main loop for capturing and processing audio."""
        frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        with sd.InputStream(channels=1, samplerate=self.sample_rate, dtype="float32") as stream:
            while self.is_running:
                try:
                    # Read a chunk of audio data equal to the frame step size
                    frame_bytes, _ = stream.read(frame_samples)

                    # Perform VAD
                    if self.vad.is_speech(frame_bytes):
                        self.is_speaking = True
                        self.silent_frames_count = 0
                        # Put the raw frame into the stream queue for external use
                        self.stream_queue.put(frame_bytes)
                    else:
                        if self.is_speaking:
                            self.silent_frames_count += 1
                            if self.silent_frames_count >= self.pause_threshold_frames:
                                print("\n--- Pause detected! ---")
                                self.pause_callback()
                                self.is_speaking = False # Reset state after callback
                                self.silent_frames_count = 0

                except Exception as e:
                    print(f"An error occurred in the ingestion loop: {e}")
                    break


# --- Example Usage ---
if __name__ == '__main__':
    # 1. Define a callback function to handle speech pauses
    def on_speech_paused():
        """This function will be called when a pause is detected."""
        print("Callback: Speech has paused. You can perform an action here, like sending data.")

    # 2. Initialize the components
    output_queue = queue.Queue()
    vad = VAD(vad_level=3)

    # 3. Create and start the ingestor
    ingestor = VoiceFrameIngestor(
        vad=vad,
        stream_queue=output_queue,
        pause_callback=on_speech_paused,
        frame_ms=30,
        overlap_ms=10,
        pause_threshold_ms=500 # Consider it a pause after 500ms of silence
    )
    ingestor.start()

    print("\nMicrophone is active. Speak and then pause to trigger the callback.")
    print("Press Ctrl+C to stop.")

    try:
        # You can process the frames from the queue in your main thread
        while True:
            try:
                frame = output_queue.get(timeout=1.0)
                # For this demo, we'll just show that frames are being produced
                print(".", end="", flush=True)
            except queue.Empty:
                # If the queue is empty, just continue waiting
                continue
    except KeyboardInterrupt:
        print("\nStopping application.")
    finally:
        ingestor.stop()