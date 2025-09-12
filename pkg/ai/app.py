# app.py (improved with graceful cancellation)

import asyncio
import contextlib
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# === Import your models ===
from pkg.ai.models.aec.mic_disabler_during_speech import AECWorker
from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.tts.tts_local import LocalTextToSpeechModel
from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel
from pkg.ai.streams.input.remote.remote_audio_consumer import RemoteAudioStreamConsumer
from pkg.ai.streams.output.remote.remote_audio_producer import RemoteAudioStreamProducer
from pkg.ai.streams.processor.aspd_stream_processor import AdvancedSpeechPauseDetectorStream
from pkg.ai.streams.processor.stt_stream_processor import SpeechToTextStreamProcessor
from pkg.ai.streams.processor.tts_stream_processor import TextToSpeechStreamProcessor
from pkg.ai.streams.processor.ttt_stream_processor import TextToTextStreamProcessor
from pkg.config import STT_MODEL_LOCAL, TTS_MODEL_LOCAL, TTT_MODEL_LOCAL, device

app = FastAPI()


@app.websocket("/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected.")

    # ==== SETTINGS ====
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    VAD_LEVEL = 3
    SHORT_PAUSE_MS = 300
    LONG_PAUSE_MS = 1000

    # ==== QUEUES ====
    mic_raw_queue = queue.Queue(maxsize=200)
    playback_ref_queue = queue.Queue(maxsize=200)
    mic_clean_queue = queue.Queue(maxsize=200)
    playback_in_queue = queue.Queue(maxsize=200)

    STT_INPUT_QUEUE = queue.Queue()
    TTT_INPUT_QUEUE = queue.Queue()
    TTS_INPUT_QUEUE = queue.Queue()

    # ==== MODELS ====
    STT_MODEL = LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device)
    TTT_MODEL = LocalTextToTextModel(TTT_MODEL_LOCAL, device=device)
    TTS_MODEL = LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device)

    # ==== COMPONENTS ====
    audio_stream = RemoteAudioStreamConsumer(output_queue=mic_raw_queue, ws=ws)
    audio_producer = RemoteAudioStreamProducer(
        input_queue=playback_in_queue,
        playback_ref_queue=playback_ref_queue,
        sample_rate=SAMPLE_RATE,
        channels=1,
        ws=ws,
        dtype="float32",
    )

    aec = AECWorker(
        mic_queue=mic_raw_queue,
        playback_ref_queue=playback_ref_queue,
        output_queue=mic_clean_queue,
        frame_size=FRAME_SAMPLES,
        sample_rate=SAMPLE_RATE,
    )

    stream_detector = AdvancedSpeechPauseDetectorStream(
        input_queue=mic_clean_queue,
        output_queue=STT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        vad_level=VAD_LEVEL,
        short_pause_ms=SHORT_PAUSE_MS,
        long_pause_ms=LONG_PAUSE_MS,
    )

    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=STT_INPUT_QUEUE,
        output_stream_queue=TTT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE,
    )
    ttt_processor = TextToTextStreamProcessor(
        ttt_model=TTT_MODEL,
        input_stream_queue=TTT_INPUT_QUEUE,
        output_stream_queue=TTS_INPUT_QUEUE,
    )
    tts_processor = TextToSpeechStreamProcessor(
        tts_model=TTS_MODEL,
        input_stream_queue=TTS_INPUT_QUEUE,
        output_stream_queue=playback_in_queue,
    )

    # ==== START ALL ====
    audio_stream.start()
    audio_producer.start()
    aec.start()
    stream_detector.start()
    stt_processor.start()
    ttt_processor.start()
    tts_processor.start()

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

        # threaded components
        aec.stop()
        stream_detector.stop()
        stt_processor.stop()
        ttt_processor.stop()
        tts_processor.stop()

        with contextlib.suppress(Exception):
            await ws.close()
