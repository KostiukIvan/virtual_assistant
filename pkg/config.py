from dotenv import load_dotenv
import os
import torch


load_dotenv()

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device==0 else 'CPU'}")


STT_MODE = os.environ.get("STT_MODE", "local")
TTT_MODE = os.environ.get("TTT_MODE", "local")
TTS_MODE = os.environ.get("TTS_MODE", "local")

STT_MODEL_LOCAL = "openai/whisper-base"
TTT_MODEL_LOCAL = "facebook/blenderbot-400M-distill"
TTS_MODEL_LOCAL = "microsoft/speecht5_tts"


STT_MODEL_REMOTE = "https://hzhe10fml4qh6k4g.us-east-1.aws.endpoints.huggingface.cloud"
TTT_MODEL_REMOTE = "https://xzlqct3bgo2ke6fz.us-east-1.aws.endpoints.huggingface.cloud"
TTS_MODEL_REMOTE = "https://hjuxzb4rq4r0ujqm.us-east-1.aws.endpoints.huggingface.cloud" # suno/bark

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
