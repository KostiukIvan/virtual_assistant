import pytest
import asyncio
from pkg.streaming_tts import StreamingTTS  # adjust import if needed

DUMMY_TEXT = "Hello world"

@pytest.mark.asyncio
async def test_initialization():
    tts = StreamingTTS(model_name="dummy-model", voice="test-voice")
    assert tts.model_name == "dummy-model"
    assert tts.voice == "test-voice"
    assert isinstance(tts.audio_queue, asyncio.Queue)
    assert tts.processing_task is None

@pytest.mark.asyncio
async def test_speak_and_get_audio():
    tts = StreamingTTS()
    
    # Start speaking
    await tts.speak(DUMMY_TEXT)
    
    # Retrieve all audio chunks
    chunks = []
    for _ in range(3):  # _synthesize produces 3 chunks
        chunk = await tts.get_audio(timeout=0.2)
        assert chunk is not None
        chunks.append(chunk)
    
    # No more chunks should be available
    assert await tts.get_audio(timeout=0.05) is None

    # Check content of chunks
    expected_chunks = [f"{DUMMY_TEXT}-chunk-{i}".encode("utf-8") for i in range(3)]
    assert chunks == expected_chunks

@pytest.mark.asyncio
async def test_get_audio_timeout_returns_none():
    tts = StreamingTTS()
    result = await tts.get_audio(timeout=0.05)
    assert result is None

@pytest.mark.asyncio
async def test_multiple_speak_calls():
    tts = StreamingTTS()
    texts = ["Hello", "World"]
    
    for text in texts:
        await tts.speak(text)
    
    # Collect all chunks
    all_chunks = []
    for _ in range(len(texts) * 3):
        chunk = await tts.get_audio(timeout=0.2)
        assert chunk is not None
        all_chunks.append(chunk)
    
    expected_chunks = []
    for text in texts:
        expected_chunks.extend([f"{text}-chunk-{i}".encode("utf-8") for i in range(3)])
    
    assert all_chunks == expected_chunks

@pytest.mark.asyncio
async def test_start_and_stop_methods_do_nothing():
    tts = StreamingTTS()
    tts.start()  # should not fail
    await tts.stop()  # should not fail
