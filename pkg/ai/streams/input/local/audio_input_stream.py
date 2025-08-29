import sounddevice as sd
import queue
import threading
import numpy as np
import time

class LocalAudioStream:
    """
    Ingests audio from the local microphone and puts the audio chunks (frames)
    into an output queue.
    """
    def __init__(self,
                 output_queue: queue.Queue,
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 channels: int = 1,
                 dtype: str = "float32"):
        """
        Initializes the LocalAudioStream.

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

        print("LocalAudioStream initialized with:")
        print(f" - Sample Rate: {self.sample_rate}")
        print(f" - Frame Duration: {self.frame_duration_ms} ms")
        print(f" - Frame Samples: {self.frame_samples}\n")

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """
        This function is called by the sounddevice stream for each new audio block.
        It puts the audio data into the output queue.
        """
        if status:
            print(f"Stream status: {status}", flush=True)
        self.output_queue.put(indata.copy())

    def _stream_thread_loop(self):
        """
        The main loop for the audio stream, run in a separate thread.
        It uses a sounddevice InputStream with a callback to continuously
        capture audio.
        """
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback
            ):
                # The stream is active in this context. We just need to keep the
                # thread alive while is_running is True.
                while self.is_running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"An error occurred in the audio stream: {e}")

    def start(self):
        """Starts the audio ingestion in a separate thread."""
        if self.is_running:
            print("Audio stream is already running.")
            return

        print("Starting local audio stream...")
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_thread_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the audio ingestion thread."""
        if not self.is_running:
            print("Audio stream is not running.")
            return
            
        print("Stopping local audio stream...")
        self.is_running = False
        if self.thread:
            self.thread.join() # Wait for the thread to finish
        print("Audio stream stopped.")

def main():
    """
    Example of using the LocalAudioStream to capture microphone audio
    and process it from a queue.
    """
    # 1. Create a queue to hold the audio frames
    audio_frames_queue = queue.Queue()

    # 2. Initialize the audio stream component
    audio_stream = LocalAudioStream(output_queue=audio_frames_queue)

    # 3. Start capturing audio
    audio_stream.start()

    print("🎤 Microphone is active. Audio frames are being captured.")
    print("   Press Ctrl+C to stop the application.")

    try:
        # 4. Main loop to consume frames from the queue
        while True:
            try:
                # Get a frame from the queue
                frame = audio_frames_queue.get(timeout=1.0)
                
                # In a real application, you would process the frame here.
                # For this demo, we just print a dot to show that frames are being received.
                print(".", end="", flush=True)
                
            except queue.Empty:
                # This can happen if the main loop is faster than the audio stream.
                print("Queue is empty, waiting for audio...")
                continue
                
    except KeyboardInterrupt:
        print("\nInterruption detected. Stopping application...")
    finally:
        # 5. Stop the audio stream gracefully
        audio_stream.stop()
        print("Application finished.")

if __name__ == "__main__":
    # To run this, you need sounddevice and numpy:
    # pip install sounddevice numpy
    main()
