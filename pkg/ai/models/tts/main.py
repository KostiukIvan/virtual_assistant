import io
import os

import numpy as np

# New imports for remote models
import requests
import sounddevice as sd
import torch

# New import for remote TTS audio processing
from scipy.io import wavfile
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from pkg.ai.models.aspd.aspd_detector import AdvancedSpeechPauseDetector

from pkg.ai.models.tts.tts_local import LocalTextToSpeechModel
from pkg.ai.models.tts.tts_remote import RemoteTextToSpeechModel
# Assume these classes are defined in their respective files as before
from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel
from pkg.ai.models.ttt.ttt_remote import RemoteTextToTextModel
from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel
from pkg.config import (
    HF_API_TOKEN,
    STT_MODE,
    STT_MODEL_LOCAL,
    STT_MODEL_REMOTE,
    TTS_MODE,
    TTS_MODEL_LOCAL,
    TTS_MODEL_REMOTE,
    TTT_MODE,
    TTT_MODEL_LOCAL,
    TTT_MODEL_REMOTE,
    device,
)

# ===== Main Conversational Loop =====
def main() -> None:
    sample_rate = 16000
    frame_duration = 30
    frame_samples = int(sample_rate * frame_duration / 1000)

    detector = AdvancedSpeechPauseDetector(
        sample_rate=sample_rate,
        frame_duration_ms=frame_duration,
        vad_level=3,
        short_pause_ms=250,
        long_pause_ms=1000,
    )

    stt = (
        LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device)
        if STT_MODE == "local"
        else RemoteSpeechToTextModel(STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    ttt = (
        LocalTextToTextModel(TTT_MODEL_LOCAL, device=device)
        if TTT_MODE == "local"
        else RemoteTextToTextModel(TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    tts = (
        LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device)
        if TTS_MODE == "local"
        else RemoteTextToSpeechModel(TTS_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    buffer = []
    recording = False

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
        while True:
            audio_chunk, _ = stream.read(frame_samples)
            if detector.is_speech(audio_chunk.flatten()):
                if not recording:
                    pass
                buffer.extend(audio_chunk.flatten())
                recording = True
            elif recording and len(buffer) > 5000:
                audio_np = np.array(buffer, dtype=np.float32)
                text = stt.audio_to_text(audio_np, sample_rate)

                if text and text.strip() and len(text.strip()) > 1:
                    reply = ttt.text_to_text(text)

                    audio_reply = tts.text_to_speech(reply)
                    sd.play(audio_reply, samplerate=tts.sample_rate)
                    sd.wait()
                else:
                    pass

                buffer = []
                recording = False
            elif recording:  # Reset if speech was too short
                buffer = []
                recording = False


if __name__ == "__main__":
    main()
