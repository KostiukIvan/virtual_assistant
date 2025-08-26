import asyncio
from typing import Optional
import numpy as np

class StreamingTTS:
    """
    Streaming Text-to-Speech.
    Converts text input into audio chunks and supports async streaming.
    """

    def __init__(self, model_name: str = "default-tts-model", voice: str = "default"):
        """
        :param model_name: TTS engine/model name.
        :param voice: Voice selection if supported.
        """
        self.model_name = model_name
        self.voice = voice
        self.audio_queue = asyncio.Queue()  # Queue for generated audio
        self.processing_task: Optional[asyncio.Task] = None

    async def speak(self, text: str):
        """
        Convert text to speech asynchronously and enqueue audio chunks.
        """
        # Here you would replace this with actual TTS API call
        # For demo, we simulate audio chunks
        audio_chunks = self._synthesize(text)

        for chunk in audio_chunks:
            await self.audio_queue.put(chunk)

    def _synthesize(self, text: str):
        """
        Returns a list of 16-bit PCM byte chunks (streaming simulation).
        """
        # Simulate a sine wave as a placeholder for audio
        duration = 0.1  # seconds per chunk
        freq = 440  # A4 note
        t = np.linspace(0, duration, int(16000 * duration), endpoint=False)

        # Create a sine wave array and convert to 16-bit PCM
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        audio_data = (wave * 32767).astype(np.int16)

        # Split the simulated audio into chunks
        chunk_size = 1024
        chunks = [audio_data[i:i + chunk_size].tobytes() for i in range(0, len(audio_data), chunk_size)]

        # Simulate processing delay
        for chunk in chunks:
            # A small sleep to simulate latency
            yield chunk # Use yield to return chunks one by one

    async def get_audio(self, timeout: float = 0.1) -> Optional[bytes]:
        """
        Async getter for audio chunks.
        Returns None if no audio is available within timeout.
        """
        try:
            return await asyncio.wait_for(self.audio_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def start(self):
        """Optional: Start background task if needed for streaming."""
        pass

    async def stop(self):
        """Clean up resources."""
        pass
