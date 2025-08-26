import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from pkg.virtual_assistant import VirtualAssistant


@pytest.mark.asyncio
async def test_virtual_assistant_end_to_end():
    """
    Full pipeline test: STT -> AI -> TTS.
    Ensures audio chunks are produced after feeding input.
    """

    # --- STT mock ---
    stt_mock = MagicMock()
    stt_mock.accept_chunk = AsyncMock()
    stt_mock.finalize = AsyncMock(return_value=("hello world", ["hello world"]))
    stt_mock.get_partial = AsyncMock(return_value=None)

    # --- AI mock ---
    responses = ["hi there", "how are you?"]

    async def ai_get_response_side_effect(timeout=0.1):
        if responses:
            return responses.pop(0)
        await asyncio.sleep(0)
        return None

    ai_mock = MagicMock()
    ai_mock.enqueue_input = AsyncMock()
    ai_mock.get_response = AsyncMock(side_effect=ai_get_response_side_effect)
    ai_mock.start = MagicMock()
    ai_mock.stop = AsyncMock()

    # --- TTS mock ---
    tts_chunks = [b"chunk1", b"chunk2", b"chunk3"]

    async def tts_get_audio_side_effect(timeout=0.1):
        if tts_chunks:
            return tts_chunks.pop(0)
        await asyncio.sleep(0)
        return None

    tts_mock = MagicMock()
    tts_mock.speak = AsyncMock()
    tts_mock.get_audio = AsyncMock(side_effect=tts_get_audio_side_effect)
    tts_mock.start = MagicMock()
    tts_mock.stop = AsyncMock()

    # --- VirtualAssistant with factories returning mocks ---
    assistant = VirtualAssistant(
        stt_class=lambda: stt_mock,
        ai_class=lambda: ai_mock,
        tts_class=lambda: tts_mock,
    )
    assistant.start()

    # --- Feed audio and finalize ---
    await assistant.feed_audio(b"dummy audio")
    await assistant.feed_audio(b"more audio")
    await assistant.feed_audio(b"end")
    await assistant.finalize()

    # --- Collect audio chunks ---
    collected = []
    for _ in range(5):  # give loops some time
        chunk = await assistant.get_audio_chunk(timeout=0.2)
        if chunk:
            collected.append(chunk)

    # --- Assertions ---
    stt_mock.accept_chunk.assert_any_call(b"dummy audio")
    stt_mock.finalize.assert_awaited()

    ai_mock.enqueue_input.assert_awaited_with("hello world")
    ai_mock.get_response.assert_called()

    tts_mock.speak.assert_any_await("hi there")
    tts_mock.speak.assert_any_await("how are you?")

    assert any(c == b"chunk1" for c in collected)
    assert any(c == b"chunk2" for c in collected)

    await assistant.stop()
