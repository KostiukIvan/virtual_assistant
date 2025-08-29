import queue
import threading
import sounddevice as sd
import time
import torch
from pkg.config import device, HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, TTT_MODE, TTT_MODEL_REMOTE, TTT_MODEL_LOCAL, TTS_MODE, TTS_MODEL_LOCAL, TTS_MODEL_REMOTE


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
                
                if not bot_response:  # Handle empty responses
                    continue
                    
                print(f"🤖 Bot: {bot_response}")
                print("🔊 Generating speech...")

                # Generate the audio from the text
                audio_output = self.tts_model.text_to_speech(bot_response)
                
                self.output_stream_queue.put(audio_output)
                
                # # 2. Play the new audio. This call is non-blocking.
                # sd.play(audio_output, samplerate=self.tts_model.sample_rate)
                # sd.wait() # Wait for the audio to finish playing

            except queue.Empty:
                continue
            except Exception as e:
                print(f"An error occurred in the TTS processing loop: {e}")

# Assume your other classes are imported
from pkg.ai.models.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.ai.models.ttt_model import LocalTextToTextModel, RemoteTextToTextModel
from pkg.ai.models.tts_model import LocalTextToSpeechModel, RemoteTextToSpeechModel
from pkg.ai.streams.processor.stt_stream_processor import SpeechToTextStreamProcessor
from pkg.ai.streams.processor.ttt_stream_processor import TextToTextStreamProcessor
from pkg.ai.streams.processor.aspd_stream_processor import AdvancedSpeechPauseDetectorStream
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.output.local.audio_producer import LocalAudioProducer

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
    TTS_INPUT_QUEUE = queue.Queue()
    AUDIO_PRODUCER_INPUT_QUEUE = queue.Queue()
    
    
    audio_stream = LocalAudioStream(output_queue=STREAM_DETECTOR_INPUT_QUEUE)

    # 3. Start capturing audio

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
    
    print(f"Loading TTT model ({TTT_MODE})...")
    TTT_MODEL = LocalTextToTextModel(TTT_MODEL_LOCAL, device=device) if TTT_MODE == "local" else RemoteTextToTextModel(TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    print(f"Loading TTS model ({TTS_MODE})...")
    TTS_MODEL = LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device) if TTS_MODE == "local" else RemoteTextToSpeechModel(TTS_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    print("Models loaded.")

    # 2. Initialize the STT Processor
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=STT_INPUT_QUEUE,
        output_stream_queue=TTT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE
    )

    # 3. Initialize the new TTT Processor
    ttt_processor = TextToTextStreamProcessor(
        ttt_model=TTT_MODEL,
        input_stream_queue=TTT_INPUT_QUEUE, # Takes input from the user text queue
        output_stream_queue=TTS_INPUT_QUEUE # Outputs to the final response queue
    )

    tts_processor = TextToSpeechStreamProcessor(
        tts_model=TTS_MODEL,
        input_stream_queue=TTS_INPUT_QUEUE,
        output_stream_queue=AUDIO_PRODUCER_INPUT_QUEUE,
    )

    audio_producer = LocalAudioProducer(
        input_queue=AUDIO_PRODUCER_INPUT_QUEUE,
        speak_callback=lambda is_speaking: print(f"[Playback Status: {'SPEAKING' if is_speaking else 'IDLE'}]")
    )

    # 4. Start all threaded components in order
    audio_stream.start()
    stream_detector.start()
    stt_processor.start()
    ttt_processor.start()
    tts_processor.start()
    audio_producer.start()

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
        audio_stream.stop()
        stream_detector.stop()
        stt_processor.stop()
        ttt_processor.stop()
        tts_processor.stop()
        audio_producer.stop()
        print("Application stopped.")