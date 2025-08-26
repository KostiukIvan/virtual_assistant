import pytest
import asyncio
from unittest.mock import patch, MagicMock
from pkg.streaming_stt import StreamingSTT

DUMMY_AUDIO = b"\x00" * 3200  # simulate small audio chunk

@pytest.fixture
def mock_dependencies():
    # Patch pipeline, VAD, AudioSegment, soundfile
    with patch("pkg.streaming_stt.pipeline") as mock_pipeline, \
         patch("pkg.streaming_stt.webrtcvad.Vad") as mock_vad_cls, \
         patch("pkg.streaming_stt.AudioSegment") as mock_audio_seg, \
         patch("pkg.streaming_stt.sf.write") as mock_sf_write:

        # ASR returns fixed text
        mock_asr = MagicMock()
        mock_asr.return_value = {"text": "hello world"}
        mock_pipeline.return_value = mock_asr

        # VAD returns True for speech
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        mock_vad_cls.return_value = mock_vad

        # AudioSegment.from_file returns a mock with raw_data
        mock_audio = MagicMock()
        mock_audio.set_frame_rate.return_value = mock_audio
        mock_audio.set_channels.return_value = mock_audio
        mock_audio.set_sample_width.return_value = mock_audio
        mock_audio.raw_data = DUMMY_AUDIO
        mock_audio_seg.from_file.return_value = mock_audio

        yield mock_pipeline, mock_vad_cls, mock_audio_seg, mock_sf_write

@pytest.mark.asyncio
async def test_accept_chunk_and_finalize(mock_dependencies):
    stt = StreamingSTT(frame_ms=500, overlap_ms=0, partial_ms=500)
    await stt.accept_chunk(DUMMY_AUDIO)
    
    # Clear buffers to avoid duplicates on finalize
    stt.speech_buf = bytearray()
    stt.frame_buffer = []

    # Get partial emitted
    partial = await stt.get_partial(timeout=0.1)
    assert partial == "hello world"

    # Finalize now should not duplicate
    final_text, parts = await stt.finalize()
    assert final_text == "hello world"
    assert parts == ["hello world"]

@pytest.mark.asyncio
async def test_multiple_chunks(mock_dependencies):
    stt = StreamingSTT(frame_ms=500, overlap_ms=0, partial_ms=500)
    await stt.accept_chunk(DUMMY_AUDIO)
    await stt.accept_chunk(DUMMY_AUDIO)

    # Let STT naturally flush partials
    partials = []
    while True:
        p = await stt.get_partial(timeout=0.1)
        if p is None:
            break
        partials.append(p)

    # Reset buffers to avoid duplication on finalize
    stt.speech_buf = bytearray()
    stt.frame_buffer = []

    final_text, parts = await stt.finalize()

    # Should now match expected
    assert partials == ["hello world", "hello world"]
    assert final_text == "hello world hello world"
    assert parts == ["hello world", "hello world"]

@pytest.mark.asyncio
async def test_finalize_empty(mock_dependencies):
    stt = StreamingSTT(frame_ms=20, partial_ms=20)
    final_text, parts = await stt.finalize()
    assert final_text == ""
    assert parts == []

@pytest.mark.asyncio
async def test_accept_chunk_with_vad_silence(mock_dependencies):
    # VAD returns False for silence
    _, mock_vad_cls, _, _ = mock_dependencies
    mock_vad_cls.return_value.is_speech.return_value = False

    stt = StreamingSTT(frame_ms=20, partial_ms=20)
    await stt.accept_chunk(DUMMY_AUDIO)
    
    # No partials expected
    partial = await stt.get_partial(timeout=0.1)
    assert partial is None

@pytest.mark.asyncio
async def test_flush_speech_buffer_error(mock_dependencies):
    stt = StreamingSTT(frame_ms=20, partial_ms=20)
    # Make ASR raise exception
    stt.asr = MagicMock(side_effect=Exception("ASR failure"))
    await stt.accept_chunk(DUMMY_AUDIO)
    
    # Partial queue should stay empty, no crash
    partial = await stt.get_partial(timeout=0.1)
    assert partial is None
