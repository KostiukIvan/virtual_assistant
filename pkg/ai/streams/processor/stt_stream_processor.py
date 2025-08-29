import queue
import threading
import numpy as np
import time
import torch
from pkg.config import device, HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE
from pkg.ai.models.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.ai.streams.processor.aspd_stream_processor import AdvancedSpeechPauseDetectorStream

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
    FRAME_DURATION_MS = 30
    VAD_LEVEL=3
    SHORT_PAUSE_MS=300
    LONG_PAUSE_MS=1000
    STREAM_DETECTOR_INPUT_QUEUE = queue.Queue()  
    STT_INPUT_QUEUE = queue.Queue() 
    TTT_INPUT_QUEUE = queue.Queue()
    
    
    stream_detector = AdvancedSpeechPauseDetectorStream(
        input_queue=STREAM_DETECTOR_INPUT_QUEUE,
        output_queue=STT_INPUT_QUEUE,
        long_pause_callback=lambda: print("LONG CALLBACK"),
        short_pause_callback=lambda: print("SHORT CALLBACK"),
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        vad_level=VAD_LEVEL,
        short_pause_ms=SHORT_PAUSE_MS,
        long_pause_ms=LONG_PAUSE_MS
    )
    
    print(f"Loading STT model ({STT_MODE})...")
    STT_MODEL = LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device) if STT_MODE == "local" else RemoteSpeechToTextModel(STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    # 2. Initialize the updated SpeechToTextStreamProcessor
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=STT_INPUT_QUEUE,
        output_stream_queue=TTT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE
    )

    # 4. Start both components
    stt_processor.start()
    stream_detector.start()

    print("\n🎤 Microphone is active. Speak and then pause to trigger transcription.")
    print("Press Ctrl+C to stop.")

    try:
        # The main thread now listens for results from the TTT_INPUT_QUEUE
        while True:
            try:
                transcribed_text = TTT_INPUT_QUEUE.get(timeout=1.0)
                print(f"\n[Final Output] -> {transcribed_text}")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nStopping application.")
    finally:
        # 5. Stop the components gracefully
        stream_detector.stop()
        stt_processor.stop()