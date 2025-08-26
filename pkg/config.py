from dotenv import load_dotenv
import os

load_dotenv()

STT_MODE = os.environ.get("STT_MODE", "local")
TTS_MODE = os.environ.get("TTS_MODE", "local")
CONVO_MODE = os.environ.get("CONVO_MODE", "local")

STT_MODEL_LOCAL = "openai/whisper-base"
CONVO_MODEL_LOCAL = "facebook/blenderbot-400M-distill"

HUGGING_FACE_API_TOKEN = os.environ.get("HUGGING_FACE_API_TOKEN")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))