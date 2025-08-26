# import pytest
# import asyncio
# from unittest.mock import AsyncMock

# from virtual_assistant.streaming_stt import StreamingSTT
# from virtual_assistant.conversational_ai import ConversationalAI
# from virtual_assistant.streaming_tts import StreamingTTS

# DUMMY_AUDIO = b"\x00" * 3200  # simulated raw audio
# DUMMY_MODEL = "dummy-llm"

# @pytest.mark.asyncio
# async def test_full_pipeline_integration():
#     # --- Initialize components ---
#     stt = StreamingSTT(frame_ms=500, overlap_ms=0, partial_ms=500)
#     ai = ConversationalAI(model_name=DUMMY_MODEL, max_context=5)
#     tts = StreamingTTS()
    
#     ai._call_model = AsyncMock(side_effect=lambda prompt: f"AI reply to: {ai.conversation[-1]['content']}")
    
#     ai.start()
    
#     # --- Simulate audio streaming ---
#     await stt.accept_chunk(DUMMY_AUDIO)

#     # Flush buffer manually to ensure partials exist
#     await stt._flush_speech_buffer()
    
#     # Collect partials from STT
#     partials = []
#     while True:
#         p = await stt.get_partial(timeout=0.1)
#         if p is None:
#             break
#         partials.append(p)
#         # Send partial to AI
#         await ai.enqueue_input(p)
    
#     # Collect AI responses
#     responses = []
#     for _ in partials:
#         resp = await ai.get_response(timeout=0.3)
#         assert resp is not None
#         responses.append(resp)
    
#     # Send AI responses to TTS
#     for resp in responses:
#         await tts.speak(resp)
    
#     # Collect TTS audio chunks
#     audio_chunks = []
#     for _ in responses:
#         for _ in range(3):  # TTS simulation produces 3 chunks per text
#             chunk = await tts.get_audio(timeout=0.2)
#             assert chunk is not None
#             audio_chunks.append(chunk)
    
#     # --- Assertions ---
#     # STT partials should not be empty
#     assert len(partials) > 0
#     # AI responses should match STT partials
#     for part, resp in zip(partials, responses):
#         assert resp == f"AI reply to: {part}"
#     # TTS chunks should match AI responses
#     expected_chunks = []
#     for resp in responses:
#         expected_chunks.extend([f"{resp}-chunk-{i}".encode("utf-8") for i in range(3)])
#     assert audio_chunks == expected_chunks
    
#     # Stop AI gracefully
#     await ai.stop()
