import sounddevice as sd
import numpy as np
import webrtcvad
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from pkg.config import (HF_API_TOKEN, device, TTT_MODE, TTT_MODEL_LOCAL, TTT_MODEL_REMOTE, 
                        STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, TTS_MODE, 
                        TTS_MODEL_LOCAL, TTS_MODEL_REMOTE)
# New imports for remote models
import requests
import os
import io
from huggingface_hub import hf_hub_download
# New import for remote TTS audio processing
from scipy.io import wavfile

# Assume these classes are defined in their respective files as before
from pkg.model_clients.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.model_clients.ttt_model import LocalTextToTextModel, RemoteTextToTextModel
from pkg.model_clients.vad_model import VAD

# ===== Base TTS =====
class TextToSpeechModel:
    def __init__(self, model: str, sample_rate: int, device: int = 0):
        self.device = device
        self.model = model
        self.sample_rate = sample_rate
    
    def text_to_speech(self, text: str) -> np.ndarray:
        raise NotImplementedError


# ===== Local HuggingFace TTS (Corrected with Placeholder) =====
class LocalTextToSpeechModel(TextToSpeechModel):
    def __init__(self, model: str = TTS_MODEL_LOCAL, sample_rate: int = 16000, device: int = 0):
        super().__init__(model, sample_rate, device)
        self.processor = SpeechT5Processor.from_pretrained(model)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model) # .to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan") # .to(self.device)

        # --- FIX: Generate a generic speaker embedding to avoid network errors ---
        # This creates a placeholder tensor. The voice will be robotic but functional.
        print("WARN: Using a generic speaker embedding. For a high-quality voice, find and load a specific speaker embedding file.")
        self.speaker_embeddings = torch.randn(1, 512) # .to(self.device)
        # --------------------------------------------------------------------

    def text_to_speech(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt") # .to(self.device)
        speech_output = self.model.generate(
            **inputs,
            speaker_embeddings=self.speaker_embeddings,
            vocoder=self.vocoder 
        )
        audio = speech_output.cpu().numpy()
        return audio

# ===== Remote HuggingFace TTS (New Class) =====
class RemoteTextToSpeechModel(TextToSpeechModel):
    def __init__(self, model: str = TTS_MODEL_REMOTE, sample_rate: int = 22050, hf_token: str = None):
        super().__init__(model, sample_rate)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("Hugging Face API token not provided.")
        
        self.api_url = model 
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

    def text_to_speech(self, text: str) -> np.ndarray:
        """Sends text to the API and returns the audio as a NumPy float array."""
        payload = {"inputs": text}
        
        print(f"...sending text to remote TTS model: {self.model}...")
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
            error_info = response.json()
            raise ConnectionError(f"API request failed: {error_info.get('error', 'Unknown error')}")
        
        # The successful response is raw audio bytes (a .wav file in memory)
        audio_bytes = response.content
        
        # Use scipy to read the in-memory WAV file and get the sample rate and data
        rate, data = wavfile.read(io.BytesIO(audio_bytes))
        self.sample_rate = rate  # Update sample rate to the actual rate from the audio file
        
        # Convert audio data from integer to float format for playback
        if data.dtype == np.int16:
            audio_float = data.astype(np.float32) / 32768.0
        else:
            audio_float = data.astype(np.float32) # Assume it's already in a compatible format if not int16

        return audio_float

# ===== Main Conversational Loop =====
def main():
    sample_rate = 16000
    frame_duration = 30
    frame_samples = int(sample_rate * frame_duration / 1000)

    # --- Initialize Models Based on Config ---
    vad = VAD(vad_level=3)
    
    print(f"Loading STT model ({STT_MODE})...")
    stt = LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device) if STT_MODE == "local" else RemoteSpeechToTextModel(STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    print(f"Loading TTT model ({TTT_MODE})...")
    ttt = LocalTextToTextModel(TTT_MODEL_LOCAL, device=device) if TTT_MODE == "local" else RemoteTextToTextModel(TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    
    print(f"Loading TTS model ({TTS_MODE})...")
    tts = LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device) if TTS_MODE == "local" else RemoteTextToSpeechModel(TTS_MODEL_REMOTE, hf_token=HF_API_TOKEN)

    buffer = []
    recording = False
    print("\n🎤 Speak into the microphone... (Ctrl+C to stop)")

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_chunk, _ = stream.read(frame_samples)
            if vad.is_speech(audio_chunk.flatten(), sample_rate):
                if not recording: print("...listening...")
                buffer.extend(audio_chunk.flatten())
                recording = True
            elif recording and len(buffer) > 5000:
                print("🔎 Transcribing...")
                audio_np = np.array(buffer, dtype=np.float32)
                text = stt.audio_to_text(audio_np, sample_rate)
                print("🗣 You said:", text)
                
                if text and text.strip() and len(text.strip()) > 1:
                    reply = ttt.text_to_text(text)
                    print("🤖 Bot:", reply)
                    
                    audio_reply = tts.text_to_speech(reply)
                    print("🔊 Playing reply...")
                    sd.play(audio_reply, samplerate=tts.sample_rate)
                    sd.wait()
                else:
                    print("🎤 You said nothing, listening again...")
                
                buffer = []
                recording = False
            elif recording: # Reset if speech was too short
                buffer = []
                recording = False

if __name__ == "__main__":
    main()