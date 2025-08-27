import queue
import threading
import numpy as np
import time
from pkg.model_clients.vad_model import VAD
from pkg.model_clients.stt_model import LocalSpeechToTextModel
from pkg.streams.local_voice_stream_ingestor import VoiceFrameIngestor

class SpeechToTextStreamProcessor:
    """
    Processes a stream of audio frames for speech-to-text transcription.
    It consumes audio frames from an input queue and places transcribed text
    into an output queue.
    """
    def __init__(self, stt_model: object, input_stream_queue: queue.Queue, output_stream_queue: queue.Queue, sample_rate: int = 16000):
        """
        Initializes the SpeechToTextStreamProcessor.

        Args:
            stt_model (object): An object with an `audio_to_text(buffer, sample_rate)` method.
            input_stream_queue (queue.Queue): The queue to get audio frames from.
            output_stream_queue (queue.Queue): The queue to put transcribed text into.
            sample_rate (int): The sample rate of the audio.
        """
        self.stt_model = stt_model
        self.input_stream_queue = input_stream_queue
        self.output_stream_queue = output_stream_queue
        self.sample_rate = sample_rate
        
        self.audio_buffer = []
        self.is_running = False
        self.thread = None

    def start(self):
        """Starts the processor in a separate thread."""
        if self.is_running:
            print("Processor is already running.")
            return
        
        print("Starting STT Stream Processor...")
        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the processor thread."""
        print("Stopping STT Stream Processor...")
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Processor stopped.")

    def _processing_loop(self):
        """The main loop for consuming and buffering audio frames."""
        while self.is_running:
            try:
                # Get a frame from the input queue
                frame = self.input_stream_queue.get(timeout=1.0)
                self.audio_buffer.append(frame)
                print(".", end="", flush=True) # Visual feedback
            except queue.Empty:
                continue
            except Exception as e:
                print(f"An error occurred in the processing loop: {e}")

    def process_audio(self):
        """
        Processes the currently buffered audio, transcribes it, places the
        result in the output queue, and clears the buffer.
        """
        if not self.audio_buffer:
            return

        print("\n🔎 Transcribing...")
        
        # Safely copy the buffer and clear the original for the next utterance
        current_buffer = self.audio_buffer
        self.audio_buffer = []

        # Concatenate all frames into a single audio clip
        audio_clip = np.concatenate([frame.flatten() for frame in current_buffer])
        
        # Perform speech-to-text
        text = self.stt_model.audio_to_text(audio_clip, sample_rate=self.sample_rate)
        
        if text:
            # Place the final text into the output queue
            print(f"📝 Recognized: {text}")
            self.output_stream_queue.put(text)


if __name__ == '__main__':
    # 1. Initialize the core components and both queues
    SAMPLE_RATE = 16000
    AUDIO_QUEUE = queue.Queue()  # For audio frames from ingestor to processor
    TEXT_QUEUE = queue.Queue()   # For text from processor to main thread
    VAD_MODEL = VAD(vad_level=3)
    STT_MODEL = LocalSpeechToTextModel()

    # 2. Initialize the updated SpeechToTextStreamProcessor
    processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=AUDIO_QUEUE,
        output_stream_queue=TEXT_QUEUE,
        sample_rate=SAMPLE_RATE
    )

    # 3. Initialize the VoiceFrameIngestor
    ingestor = VoiceFrameIngestor(
        vad=VAD_MODEL,
        stream_queue=AUDIO_QUEUE, # Ingestor outputs to the AUDIO_QUEUE
        pause_callback=processor.process_audio,
        sample_rate=SAMPLE_RATE,
        frame_ms=30,
        pause_threshold_ms=800
    )

    # 4. Start both components
    processor.start()
    ingestor.start()

    print("\n🎤 Microphone is active. Speak and then pause to trigger transcription.")
    print("Press Ctrl+C to stop.")

    try:
        # The main thread now listens for results from the TEXT_QUEUE
        while True:
            try:
                transcribed_text = TEXT_QUEUE.get(timeout=1.0)
                print(f"\n[Final Output] -> {transcribed_text}")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nStopping application.")
    finally:
        # 5. Stop the components gracefully
        ingestor.stop()
        processor.stop()