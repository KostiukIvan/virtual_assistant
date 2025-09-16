import asyncio
import json

import numpy as np
import websockets

import pkg.config as config
from pkg.voice_app.aec_worker import AECWorker
from pkg.voice_app.aspd_worker import AdvancedSpeechPauseDetectorAsyncStream
from pkg.voice_app.input_audio_stream import LocalAudioStream
from pkg.voice_app.output_audio_stream import LocalAudioProducer

"""
[ Local Mic ] -----> [ LocalAudioStream ] ---<raw frame>--> [ AEC ] ---<cleaned frame>--> [ ASPD ] ---<frame or event>----> | WebSocket |--------------
                                                                ^                                                                                    |  
                                                                |                                                                                    |   
                                                                ------------<dup frame>----------                                        [ Remote STT/TTT/TTS ]            
                                                                                                |                                                    |
                                                                                                |                                                    |                     
[ Local Speaker ] <------------------------<original frame>-------------------------- [ LocalAudioProducer ] <-------------| WebSocket |<-------------

"""

"""
    | WebSocket | -------> [ RemoteAudioStreamConsumer ] ---<List of frames, event>--> [ STT Stream Processor ] ---<text, event>--> [ Text Queue ]
    
    
    
    
    | WebSocket | <------- [ RemoteAudioStreamProducer ] <---<List of frames, event>--- [ TTS Stream Processor ] <---<text, event>--- [ Text Queue ]



"""


HF_WS_URL = "ws://127.0.0.1:8000/stream"
# HF_WS_URL = "wss://ivankostiuk-virtual-voice-assistant.hf.space/stream"



async def voice_client(
    stt_input_queue: asyncio.Queue,
    playback_queue: asyncio.Queue,
    playback_ref_queue: asyncio.Queue,
    start_workers_fn: callable = None,
):

    async with websockets.connect(HF_WS_URL) as ws:
        print("Connected to WebSocket Voice API")
        start_workers_fn()

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
                    frame = np.frombuffer(msg, dtype=config.AUDIO_DTYPE)
                    await playback_queue.put(frame)
                    await playback_ref_queue.put(frame)  # for AEC

            except asyncio.CancelledError:
                print("Receiver task was cancelled.")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server, receiver exiting.")

        # Run both sender and receiver concurrently
        await asyncio.gather(sender(), receiver())


def start_workers(
    audio_stream: LocalAudioStream,
    audio_producer: LocalAudioProducer,
    aec: AECWorker,
    stream_detector: AdvancedSpeechPauseDetectorAsyncStream,
):
    audio_stream.start()
    audio_producer.start()
    aec.start()
    stream_detector.start()


async def main():
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
    )
    stream_detector = AdvancedSpeechPauseDetectorAsyncStream(
        input_queue=mic_cleaned_queue,
        output_queue=stt_input_queue,
    )

    try:
        await voice_client(
            stt_input_queue,
            playback_queue,
            playback_ref_queue,
            start_workers_fn=lambda: start_workers(audio_stream, audio_producer, aec, stream_detector),
        )
    except asyncio.CancelledError:
        pass
    finally:
        await audio_stream.stop()
        await audio_producer.stop()
        aec.stop()
        stream_detector.stop()


if __name__ == "__main__":
    asyncio.run(main())
