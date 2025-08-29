import queue
import threading
import time
import torch
from pkg.config import device, HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, TTT_MODE, TTT_MODEL_REMOTE, TTT_MODEL_LOCAL
from pkg.model_clients.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.model_clients.tts_model import LocalTextToSpeechModel, RemoteTextToTextModel
from pkg.model_clients.spd_model import SpeechPauseDetector

class TextToTextStreamProcessor:
    """
    Consumes text from an input queue, processes it with a text-to-text model,
    and places the generated response into an output queue.
    """
    def __init__(self, ttt_model: object, input_stream_queue: queue.Queue, output_stream_queue: queue.Queue):
        """
        Initializes the TextToTextStreamProcessor.

        Args:
            ttt_model (object): An object with a `text_to_text(message)` method.
            input_stream_queue (queue.Queue): The queue to get user text from.
            output_stream_queue (queue.Queue): The queue to put bot responses into.
        """
        self.ttt_model = ttt_model
        self.input_stream_queue = input_stream_queue
        self.output_stream_queue = output_stream_queue
        
        self.is_running = False
        self.thread = None

    def start(self):
        # """Starts the processor in a separate thread."""
        # if self.is_running:
        #     print("TTT Processor is already running.")
        #     return
        
        # print("Starting TTT Stream Processor...")
        # self.is_running = True
        # self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        # self.thread.start()
        pass

    def stop(self):
        # """Stops the processor thread."""
        # print("Stopping TTT Stream Processor...")
        # self.is_running = False
        # if self.thread:
        #     self.thread.join()
        # print("TTT Processor stopped.")
        pass

    def _processing_loop(self):
        """The main loop for consuming text and generating responses."""
        while self.is_running:
            try:
                # Get transcribed text from the input queue
                user_text = self.input_stream_queue.get(timeout=1.0)
                
                print(f"\n🗣 You said: {user_text}")
                print("🧠 Thinking...")

                # Generate a response using the TTT model
                bot_response = self.ttt_model.text_to_text(user_text)
                
                # Put the final response into the output queue
                self.output_stream_queue.put(bot_response)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"An error occurred in the TTT processing loop: {e}")
    
    def process_text(self):
        input_message = ""
        while True:
            try:
                # Get transcribed text from the input queue
                user_text = self.input_stream_queue.get(timeout=1.0)
                input_message += user_text
             
            except queue.Empty:
                break
            except Exception as e:
                print(f"An error occurred in the TTT processing loop: {e}")
        
        print(f"\n🗣 You said: {input_message}")
        print("🧠 Thinking...")

        # Generate a response using the TTT model
        bot_response = self.ttt_model.text_to_text(input_message)

        # Put the final response into the output queue
        self.output_stream_queue.put(bot_response)



# Assume your other classes are imported
from pkg.model_clients.vad_model import VAD
from pkg.model_clients.stt_model import LocalSpeechToTextModel
from pkg.model_clients.ttt_model import LocalTextToTextModel
from pkg.streams.local_voice_stream_ingestor import VoiceFrameIngestor
from pkg.streams.local_stt_stream_processor import SpeechToTextStreamProcessor

if __name__ == '__main__':

    # 1. Initialize models and all three queues
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    AUDIO_QUEUE = queue.Queue()
    USER_TEXT_QUEUE = queue.Queue()
    BOT_RESPONSE_QUEUE = queue.Queue()

    VAD_MODEL = VAD(vad_level=3)
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

    # 5. Start all threaded components
    stt_processor.start()
    ttt_processor.start()
    ingestor.start()

    print("\n🎤 Microphone is active. Speak, then pause for the bot to respond.")
    print("Press Ctrl+C to stop.")

    try:
        # The main thread now listens for the final bot response
        while True:
            try:
                bot_reply = BOT_RESPONSE_QUEUE.get(timeout=1.0)
                print(f"🤖 Bot: {bot_reply}\n")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nStopping application.")
    finally:
        # 6. Stop all components gracefully
        ingestor.stop()
        stt_processor.stop()
        ttt_processor.stop()