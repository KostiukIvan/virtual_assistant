import queue
import time

from pkg.model_clients.vad_model import VAD
from pkg.model_clients.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.model_clients.ttt_model import LocalTextToTextModel, RemoteTextToTextModel
from pkg.model_clients.tts_model import LocalTextToSpeechModel, RemoteTextToSpeechModel
from pkg.streams.local_voice_stream_ingestor import VoiceFrameIngestor
from pkg.streams.local_stt_stream_processor import SpeechToTextStreamProcessor
from pkg.streams.local_ttt_stream_processor import TextToTextStreamProcessor
from pkg.streams.local_tts_stream_processor import TextToSpeechStreamProcessor
from pkg.config import device, HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, TTT_MODE, TTT_MODEL_REMOTE, TTT_MODEL_LOCAL, TTS_MODE, TTS_MODEL_LOCAL, TTS_MODEL_REMOTE

class VirtualAssistant:
    """
    Orchestrates the entire voice assistant pipeline, from voice ingestion
    to speech synthesis of the bot's response.
    """
    def __init__(self,
                 vad_model,
                 stt_model,
                 ttt_model,
                 tts_model,
                 voice_ingestor,
                 stt_processor,
                 ttt_processor,
                 tts_processor):
        """
        Initializes the VirtualAssistant with all necessary components.
        """
        self.vad_model = vad_model
        self.stt_model = stt_model
        self.ttt_model = ttt_model
        self.tts_model = tts_model
        self.voice_ingestor = voice_ingestor
        self.stt_processor = stt_processor
        self.ttt_processor = ttt_processor
        self.tts_processor = tts_processor

    def start(self):
        """
        Starts all the stream processing threads in the correct order
        to begin the conversation.
        """
        print("🚀 Starting Virtual Assistant...")
        # Start processors that wait for input first
        self.stt_processor.start()
        self.ttt_processor.start()
        self.tts_processor.start()
        
        # Start the voice ingestor last, as it begins the data flow
        self.voice_ingestor.start()
        
        print("\n🎤 Assistant is now active. Speak, then pause for a response.")

    def stop(self):
        """
        Stops all stream processing threads gracefully.
        """
        print("\n🛑 Stopping Virtual Assistant...")
        self.voice_ingestor.stop()
        self.stt_processor.stop()
        self.ttt_processor.stop()
        self.tts_processor.stop()
        print("✅ Assistant has been shut down.")

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Initialize all models and queues
    SAMPLE_RATE = 16000
    AUDIO_QUEUE = queue.Queue()
    USER_TEXT_QUEUE = queue.Queue()
    BOT_RESPONSE_QUEUE = queue.Queue()

    print("Loading models...")
    VAD_MODEL = VAD()
    print(f"Loading STT model ({STT_MODE})...")
    STT_MODEL = LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device) if STT_MODE == "local" else RemoteSpeechToTextModel(STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    print(f"Loading TTT model ({TTT_MODE})...")
    TTT_MODEL = LocalTextToTextModel(TTT_MODEL_LOCAL, device=device) if TTT_MODE == "local" else RemoteTextToTextModel(TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    print(f"Loading TTS model ({TTS_MODE})...")
    TTS_MODEL = LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device) if TTS_MODE == "local" else RemoteTextToSpeechModel(TTS_MODEL_REMOTE, hf_token=HF_API_TOKEN)
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

    # 4. Create the Virtual Assistant instance
    assistant = VirtualAssistant(
        vad_model=VAD_MODEL,
        stt_model=STT_MODEL,
        ttt_model=TTT_MODEL,
        tts_model=TTS_MODEL,
        voice_ingestor=ingestor,
        stt_processor=stt_processor,
        ttt_processor=ttt_processor,
        tts_processor=tts_processor
    )

    # 5. Start the conversation
    assistant.start()

    # The main thread waits for a KeyboardInterrupt to stop the assistant
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # 6. Stop the assistant gracefully
        assistant.stop()