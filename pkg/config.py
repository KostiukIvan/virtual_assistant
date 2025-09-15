import os

import torch
from dotenv import load_dotenv

load_dotenv()

# SERVER CONFIG
# Hugging Face API Token for remote models
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# STT CONFIG
STT_MODE = os.environ.get("STT_MODE", "local")
#STT_MODEL_LOCAL = "openai/whisper-base"
STT_MODEL_LOCAL = "openai/whisper-base"
STT_MODEL_REMOTE = "https://hzhe10fml4qh6k4g.us-east-1.aws.endpoints.huggingface.cloud"


# TTT CONFIG
TTT_MODE = os.environ.get("TTT_MODE", "local")
#TTT_MODEL_LOCAL = "facebook/blenderbot-400M-distill"
TTT_MODEL_LOCAL = "facebook/blenderbot-400M-distill"
TTT_MODEL_REMOTE = "https://xzlqct3bgo2ke6fz.us-east-1.aws.endpoints.huggingface.cloud"
TTT_MAX_TOKENS = 256
TTT_NUM_RETURN_SEQUENCES = 1
TTT_MEMORY_SIZE = 10
TTT_MEMORY_MANAGER_WINDOW_SIZE = 10
TTT_MEMORY_MANAGER_SUMMARY_EVERY = 20


# TTS CONFIG
TTS_MODE = os.environ.get("TTS_MODE", "local")
#TTS_MODEL_LOCAL = "microsoft/speecht5_tts"
TTS_MODEL_LOCAL = "microsoft/speecht5_tts"
TTS_MODEL_REMOTE = "https://hjuxzb4rq4r0ujqm.us-east-1.aws.endpoints.huggingface.cloud"  # suno/bark

# AUDIO CONFIG
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

# QUEUE CONFIG
QUEUE_MAXSIZE = 200  # max size for all queues
QUEUE_TIMEOUT = 1.0  # seconds
QUEUE_SLEEP = 0.01  # seconds

# GPU CONFIG
DEVICE = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
DEVICE_CUDA_OR_CPU = "cuda" if torch.cuda.is_available() else "cpu"
