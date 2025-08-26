import asyncio
from typing import Optional, List

class VirtualAssistant:
    """
    High-level virtual assistant integrating StreamingSTT, ConversationalAI, and StreamingTTS.
    Automatically streams audio -> text -> AI -> speech.
    """

    def __init__(self, stt_class, ai_class, tts_class, stt_config=None, ai_config=None, tts_config=None):
        # Initialize STT, AI, TTS
        self.stt = stt_class(**(stt_config or {}))
        self.ai = ai_class(**(ai_config or {}))
        self.tts = tts_class(**(tts_config or {}))

        # Background tasks
        self._stt_task: Optional[asyncio.Task] = None
        self._ai_task: Optional[asyncio.Task] = None
        self._running = False

    async def _stt_loop(self):
        """
        Continuously get partials from STT and forward to AI.
        """
        while self._running:
            partial = await self.stt.get_partial(timeout=0.1)
            if partial is not None:
                await self.ai.enqueue_input(partial)
            await asyncio.sleep(0)  # yield control

    async def _ai_loop(self):
        """
        Continuously get AI responses and feed them to TTS.
        """
        while self._running:
            response = await self.ai.get_response(timeout=0.1)
            if response is not None:
                await self.tts.speak(response)
            await asyncio.sleep(0)  # yield control

    def start(self):
        """Start all background tasks."""
        self._running = True
        self.ai.start()  # start AI processing
        self._stt_task = asyncio.create_task(self._stt_loop())
        self._ai_task = asyncio.create_task(self._ai_loop())

    async def stop(self):
        """Stop all tasks gracefully."""
        self._running = False
        await self.ai.stop()
        if self._stt_task:
            await self._stt_task
        if self._ai_task:
            await self._ai_task

    async def feed_audio(self, audio_bytes: bytes):
        """Feed raw audio chunk to STT."""
        await self.stt.accept_chunk(audio_bytes)

    async def get_audio_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get next TTS audio chunk."""
        return await self.tts.get_audio(timeout)

    async def finalize(self):
        """
        Finalize STT and AI buffers at end of conversation.
        Returns (final_text, partial_texts).
        """
        final_text, parts = await self.stt.finalize()
        # Push any remaining partials to AI
        for part in parts:
            await self.ai.enqueue_input(part)
        await asyncio.sleep(0.05)  # give AI time to process
        return final_text, parts
