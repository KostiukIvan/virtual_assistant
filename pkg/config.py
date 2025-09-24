import torch
from dotenv import load_dotenv

load_dotenv()

LOCAL = False

# Client config
if LOCAL:
    HF_WS_URL = "ws://127.0.0.1:8000/stream"
else:
    HF_WS_URL = "wss://ivankostiuk-virtual-voice-assistant.hf.space/stream"


# Server config
# ============================== STT ==============================
# Faster Whisper models: "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
# Other models: "facebook/wav2vec2-base-960h", "openai/whisper-small"
STT_MODEL = "small.en"
STT_CONFIDENCE_THRESHOLD = 0.3  # minimum confidence to accept transcription

# ============================== TTT ==============================
if LOCAL:
    TTT_MODEL = "facebook/blenderbot-400M-distill"
else:
    # TTT_MODEL = "facebook/blenderbot-3B"
    TTT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TTT_MAX_TOKENS = 256
TTT_NUM_RETURN_SEQUENCES = 1
TTT_MEMORY_SIZE = 10
TTT_MEMORY_MANAGER_WINDOW_SIZE = 10
TTT_MEMORY_MANAGER_SUMMARY_EVERY = 20


# ============================== TTS ==============================
# TTS_MODEL = "microsoft/speecht5_tts"
TTS_MODEL = "microsoft/speecht5_tts"

# ============================== AUDIO CONFIG ==============================
AUDIO_SAMPLE_RATE = 16000
AUDIO_FRAME_DURATION_MS = 30  # ms
AUDIO_FRAME_DURATION_NS = AUDIO_FRAME_DURATION_MS * 1_000_000  # ns
AUDIO_GRACE_MS = 100  # ms
AUDIO_GRACE_NS = AUDIO_GRACE_MS * 1_000_000  #
AUDIO_CHANNELS = 1
AUDIO_FRAME_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_FRAME_DURATION_MS / 1000)
AUDIO_DTYPE = "float32"  # float32 for models, int16 for pyaudio
AUDIO_VAD_LEVEL = 3  # 0-3, 3 is the most aggressive
AUDIO_SHORT_PAUSE_MS = 300  # ms
AUDIO_LONG_PAUSE_MS = 1000  # ms
AUDIO_HISTORY_FRAMES = 10  # number of frames to keep in history for VAD context

# ============================== QUEUE CONFIG ==============================
QUEUE_MAXSIZE = 200  # max size for all queues
QUEUE_TIMEOUT = 1.0  # seconds
QUEUE_SLEEP = 0.01  # seconds

# ============================== GPU CONFIG ==============================
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
DEVICE_CUDA_OR_CPU = "cuda" if torch.cuda.is_available() else "cpu"
TENSOR_DTYPE = torch.float32
