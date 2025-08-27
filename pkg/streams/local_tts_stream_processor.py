import queue
import threading
import sounddevice as sd
import time
import torch

class TextToSpeechStreamProcessor:
    """
    Consumes text from an input queue, converts it to speech using a TTS model,
    and plays the resulting audio.
    """
    def __init__(self, tts_model: object, input_stream_queue: queue.Queue):
        """
        Initializes the TextToSpeechStreamProcessor.

        Args:
            tts_model (object): An object with a `text_to_speech(text)` method.
            input_stream_queue (queue.Queue): The queue to get bot responses from.
        """
        self.tts_model = tts_model
        self.input_stream_queue = input_stream_queue
        
        self.is_running = False
        self.thread = None

    def start(self):
        """Starts the processor in a separate thread."""
        if self.is_running:
            print("TTS Processor is already running.")
            return
        
        print("Starting TTS Stream Processor...")
        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the processor thread."""
        print("Stopping TTS Stream Processor...")
        self.is_running = False
        if self.thread:
            # Stop any currently playing audio
            sd.stop()
            self.thread.join()
        print("TTS Processor stopped.")

    def _processing_loop(self):
        """The main loop for consuming text and playing audio."""
        while self.is_running:
            try:
                # Get the bot's response text from the input queue
                bot_response = self.input_stream_queue.get(timeout=1.0)
                
                print(f"🤖 Bot: {bot_response}")
                print("🔊 Generating speech...")

                # Generate the audio from the text
                audio_output = self.tts_model.text_to_speech(bot_response)
                
                # Play the generated audio
                sd.play(audio_output, samplerate=self.tts_model.sample_rate)
                sd.wait() # Wait for the audio to finish playing

            except queue.Empty:
                continue
            except Exception as e:
                print(f"An error occurred in the TTS processing loop: {e}")

# Assume your other classes are imported
from pkg.model_clients.vad_model import VAD
from pkg.model_clients.stt_model import LocalSpeechToTextModel
from pkg.model_clients.ttt_model import LocalTextToTextModel
from pkg.model_clients.tts_model import LocalTextToSpeechModel
from pkg.streams.local_voice_stream_ingestor import VoiceFrameIngestor
from pkg.streams.local_stt_stream_processor import SpeechToTextStreamProcessor
from pkg.streams.local_ttt_stream_processor import TextToTextStreamProcessor

if __name__ == '__main__':
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device==0 else 'CPU'}")
    # 1. Initialize all models and queues for the full pipeline
    SAMPLE_RATE = 16000
    AUDIO_QUEUE = queue.Queue()
    USER_TEXT_QUEUE = queue.Queue()
    BOT_RESPONSE_QUEUE = queue.Queue()

    print("Loading models...")
    VAD_MODEL = VAD()
    STT_MODEL = LocalSpeechToTextModel()
    TTT_MODEL = LocalTextToTextModel()
    TTS_MODEL = LocalTextToSpeechModel()
    print("Models loaded.")

    # 2. Initialize all stream processors
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=AUDIO_QUEUE,
        output_stream_queue=USER_TEXT_QUEUE
    )

    ttt_processor = TextToTextStreamProcessor(
        ttt_model=TTT_MODEL,
        input_stream_queue=USER_TEXT_QUEUE,
        output_stream_queue=BOT_RESPONSE_QUEUE
    )

    tts_processor = TextToSpeechStreamProcessor(
        tts_model=TTS_MODEL,
        input_stream_queue=BOT_RESPONSE_QUEUE
    )

    # 3. Initialize the Voice Ingestor
    ingestor = VoiceFrameIngestor(
        vad=VAD_MODEL,
        stream_queue=AUDIO_QUEUE,
        pause_callback=stt_processor.process_audio,
        sample_rate=SAMPLE_RATE,
        frame_ms=30,
        pause_threshold_ms=1000
    )

    # 4. Start all threaded components in order
    stt_processor.start()
    ttt_processor.start()
    tts_processor.start()
    ingestor.start()

    print("\n🎤 Voice Assistant is active. Speak, then pause for the bot to respond.")
    print("Press Ctrl+C to stop.")

    try:
        # The main thread can simply wait, as all work is done in the background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping application...")
    finally:
        # 5. Stop all components gracefully
        ingestor.stop()
        stt_processor.stop()
        ttt_processor.stop()
        tts_processor.stop()
        print("Application stopped.")