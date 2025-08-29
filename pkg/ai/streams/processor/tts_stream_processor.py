import queue
import threading
import sounddevice as sd
import time
import torch
from pkg.config import device, HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, TTT_MODE, TTT_MODEL_REMOTE, TTT_MODEL_LOCAL, TTS_MODE, TTS_MODEL_LOCAL, TTS_MODEL_REMOTE
from pkg.model_clients.spd_model import SpeechPauseDetector

class TextToSpeechStreamProcessor:
    """
    Consumes text from an input queue, converts it to speech using a TTS model,
    and plays the resulting audio.
    """
    def __init__(self, tts_model: object, input_stream_queue: queue.Queue, output_stream_queue: queue.Queue):
        """
        Initializes the TextToSpeechStreamProcessor.

        Args:
            tts_model (object): An object with a `text_to_speech(text)` method.
            input_stream_queue (queue.Queue): The queue to get bot responses from.
            output_stream_queue (queue.Queue): The queue to receive voice responses
        """
        self.tts_model = tts_model
        self.input_stream_queue = input_stream_queue
        self.output_stream_queue = output_stream_queue
        
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
                
                # self.output_stream_queue.put(audio_output)
                
                # Play the generated audio
                sd.play(audio_output, samplerate=self.tts_model.sample_rate)
                sd.wait() # Wait for the audio to finish playing

            except queue.Empty:
                continue
            except Exception as e:
                print(f"An error occurred in the TTS processing loop: {e}")

# Assume your other classes are imported
from pkg.model_clients.vad_model import VAD
from pkg.model_clients.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.model_clients.ttt_model import LocalTextToTextModel, RemoteTextToTextModel
from pkg.model_clients.tts_model import LocalTextToSpeechModel, RemoteTextToSpeechModel
from pkg.streams.local_voice_stream_ingestor import VoiceFrameIngestor
from pkg.streams.local_stt_stream_processor import SpeechToTextStreamProcessor
from pkg.streams.local_ttt_stream_processor import TextToTextStreamProcessor

if __name__ == '__main__':
    # 1. Initialize all models and queues for the full pipeline
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    AUDIO_QUEUE = queue.Queue()
    USER_TEXT_QUEUE = queue.Queue()
    BOT_RESPONSE_QUEUE = queue.Queue()

    print("Loading models...")
    VAD_MODEL = VAD()
    
    SPD_MODEL = SpeechPauseDetector(sample_rate=SAMPLE_RATE,
                            frame_duration_ms=FRAME_DURATION_MS,
                            silence_threshold_db=-40,
                            inhale_duration_ms=200,
                            sentence_end_duration_ms=450,
                            history_frames=5)
    
    print(f"Loading STT model ({STT_MODE})...")
    STT_MODEL = LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device) if STT_MODE == "local" else RemoteSpeechToTextModel(STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    print(f"Loading TTT model ({TTT_MODE})...")
    TTT_MODEL = LocalTextToTextModel(TTT_MODEL_LOCAL, device=device) if TTT_MODE == "local" else RemoteTextToTextModel(TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    print(f"Loading TTS model ({TTS_MODE})...")
    TTS_MODEL = LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device) if TTS_MODE == "local" else RemoteTextToSpeechModel(TTS_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    print("Models loaded.")

    # 2. Initialize the STT Processor
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=AUDIO_QUEUE,
        output_stream_queue=USER_TEXT_QUEUE, # Outputs to the user text queue
        sample_rate=SAMPLE_RATE,
    )

    # 3. Initialize the new TTT Processor
    ttt_processor = TextToTextStreamProcessor(
        ttt_model=TTT_MODEL,
        input_stream_queue=USER_TEXT_QUEUE, # Takes input from the user text queue
        output_stream_queue=BOT_RESPONSE_QUEUE # Outputs to the final response queue
    )

    tts_processor = TextToSpeechStreamProcessor(
        tts_model=TTS_MODEL,
        input_stream_queue=BOT_RESPONSE_QUEUE,
        output_stream_queue=None,
    )

    # 4. Initialize the Voice Ingestor
    ingestor = VoiceFrameIngestor(
        vad=VAD_MODEL,
        stream_queue=AUDIO_QUEUE, # Ingestor outputs to the AUDIO_QUEUE
        long_pause_callback=lambda: (stt_processor.process_audio(), ttt_processor.process_text()),
        short_pause_callback= stt_processor.process_audio,
        sample_rate=SAMPLE_RATE,
        frame_ms=FRAME_DURATION_MS,
        pause_threshold_ms=1000,
        spd=SPD_MODEL
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