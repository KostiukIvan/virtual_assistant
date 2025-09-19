# app.py (improved with graceful cancellation)

import asyncio
import contextlib
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# === Import your models ===
from pkg.ai.models.stt.stt_model_selector import STTModelSelector
from pkg.ai.models.tts.tts_local import LocalTextToSpeechModel
from pkg.ai.models.ttt.ttt_model_selector import TTTModelSelector
from pkg.ai.streams.processor.remote_audio_consumer import RemoteAudioStreamConsumer
from pkg.ai.streams.processor.remote_audio_producer import RemoteAudioStreamProducer
from pkg.ai.streams.processor.stt_stream_processor import SpeechToTextStreamProcessor
from pkg.ai.streams.processor.tts_stream_processor import TextToSpeechStreamProcessor
from pkg.ai.streams.processor.ttt_stream_processor import TextToTextStreamProcessor
from pkg.config import STT_MODEL, TTS_MODEL, TTT_MODEL

app = FastAPI()
# For local dev:
# uvicorn pkg.ai.app:app --host 0.0.0.0 --port 8000 --reload
# For Hugging Face Spaces:
# uvicorn app:app --host 0.0.0.0 --port $PORT


@app.get("/")
async def root():
    return {"status": "ok", "message": "Voice Assistant WebSocket is running. Connect to /stream"}



# ==== QUEUES ====
PLAYBACK_IN_QUEUE = queue.Queue(maxsize=200)
STT_INPUT_QUEUE = queue.Queue()
TTT_INPUT_QUEUE = queue.Queue()
TTS_INPUT_QUEUE = queue.Queue()

# ==== MODELS ====
STT = STTModelSelector.get_stt_model(STT_MODEL)
TTT = TTTModelSelector.get_stt_model(TTT_MODEL)
TTS = LocalTextToSpeechModel(TTS_MODEL)

# ==== PROCESSORS ====
stt_processor = SpeechToTextStreamProcessor(
    stt_model=STT,
    input_stream_queue=STT_INPUT_QUEUE,
    output_stream_queue=TTT_INPUT_QUEUE,
)
ttt_processor = TextToTextStreamProcessor(
    ttt_model=TTT,
    input_stream_queue=TTT_INPUT_QUEUE,
    output_stream_queue=TTS_INPUT_QUEUE,
)
tts_processor = TextToSpeechStreamProcessor(
    tts_model=TTS,
    input_stream_queue=TTS_INPUT_QUEUE,
    output_stream_queue=PLAYBACK_IN_QUEUE,
)

stt_processor.start()
ttt_processor.start()
tts_processor.start()


@app.websocket("/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected.")

    # ==== COMPONENTS ====
    audio_stream = RemoteAudioStreamConsumer(output_queue=STT_INPUT_QUEUE, ws=ws)
    audio_producer = RemoteAudioStreamProducer(
        input_queue=PLAYBACK_IN_QUEUE,
        ws=ws,
    )

    audio_stream.start()
    audio_producer.start()

    try:
        print("Assistant running. Speak into the mic...")
        while True:
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("Client disconnected.")

    except Exception as e:
        print("Stream error:", e)

    finally:
        print("Shutting down...")

        # async stops (with cancellation guard)
        with contextlib.suppress(asyncio.CancelledError):
            await audio_stream.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await audio_producer.stop()

        with contextlib.suppress(Exception):
            await ws.close()



# # threaded components
# stt_processor.stop()
# ttt_processor.stop()
# tts_processor.stop()