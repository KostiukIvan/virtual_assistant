import asyncio
import json
import time

import numpy as np
import websockets

from pkg.voice_app.aec_worker import AECWorker
from pkg.voice_app.aspd_worker import AdvancedSpeechPauseDetectorAsyncStream
from pkg.voice_app.input_audio_stream import LocalAudioStream
from pkg.voice_app.output_audio_stream import LocalAudioProducer

"""
[ Local Mic ] -----> [ LocalAudioStream ] ---<raw frame>--> [ AEC ] ---<cleaned frame>--> [ ASPD ] ---<frames, event>---> | WebSocket |-------------
                                                                ^                                                                                   |  
                                                                ------------<dup frame>----------                                        [ Remote STT/TTT/TTS ]            
                                                                                                |                                                    |                 
[ Local Speaker ] <------------------------<original frame>-------------------------- [ LocalAudioProducer ] <-------------| WebSocket |<-------------

"""


# HF_WS_URL = "wss://your-hf-space-url.hf.space/stream"
HF_WS_URL = "ws://127.0.0.1:8000/stream"


async def voice_client(
    stt_input_queue: asyncio.Queue, playback_queue: asyncio.Queue, playback_ref_queue: asyncio.Queue
):

    async with websockets.connect(HF_WS_URL) as ws:
        print("Connected to WebSocket Voice API")

        # Sender coroutine: sends mic audio to the server
        async def sender():
            try:
                while True:
                    # Wait for a speech event from the detector
                    event_data = await stt_input_queue.get()
                    stt_input_queue.task_done()

                    event = event_data.get("event")
                    data = event_data.get("data")
                    if event == "p" and data is not None:
                        await ws.send(data.tobytes())
                    if event == "s":
                        print("[SENT] Short pause detected, sending audio.")
                        await ws.send(json.dumps({"event": "s"}))
                    elif event == "L":
                        print("[SENT] Long pause detected, sending stop signal.")
                        await ws.send(json.dumps({"event": "L"}))
            except asyncio.CancelledError:
                print("Sender task was cancelled.")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server, sender exiting.")

        # Receiver coroutine: receives audio from the server and queues it for playback
        async def receiver():
            try:
                async for msg in ws:
                    # The server sends back audio data to be played
                    frame = np.frombuffer(msg, dtype="float32")
                    await playback_queue.put(frame)
                    await playback_ref_queue.put((time.monotonic_ns(), frame))  # for AEC

            except asyncio.CancelledError:
                print("Receiver task was cancelled.")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server, receiver exiting.")

        # Run both sender and receiver concurrently
        await asyncio.gather(sender(), receiver())


async def main():

    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    VAD_LEVEL = 3
    SHORT_PAUSE_MS = 300
    LONG_PAUSE_MS = 1000

    mic_raw_queue = asyncio.Queue()
    mic_cleaned_queue = asyncio.Queue()
    playback_ref_queue = asyncio.Queue()
    playback_queue = asyncio.Queue()
    stt_input_queue = asyncio.Queue()

    audio_stream = LocalAudioStream(output_queue=mic_raw_queue)
    audio_producer = LocalAudioProducer(input_queue=playback_queue)
    aec = AECWorker(
        mic_queue=mic_raw_queue,
        playback_ref_queue=playback_ref_queue,
        output_queue=mic_cleaned_queue,
        frame_size=FRAME_SAMPLES,
        sample_rate=SAMPLE_RATE,
    )
    stream_detector = AdvancedSpeechPauseDetectorAsyncStream(
        input_queue=mic_cleaned_queue,
        output_queue=stt_input_queue,
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        vad_level=VAD_LEVEL,
        short_pause_ms=SHORT_PAUSE_MS,
        long_pause_ms=LONG_PAUSE_MS,
    )

    audio_stream.start()
    audio_producer.start()
    aec.start()
    stream_detector.start()

    try:
        await voice_client(stt_input_queue, playback_queue, playback_ref_queue)
    except asyncio.CancelledError:
        pass
    finally:
        await audio_stream.stop()
        await audio_producer.stop()
        aec.stop()
        stream_detector.stop()


if __name__ == "__main__":
    asyncio.run(main())
