import asyncio
import json
import queue

import numpy as np
import websockets

from pkg.voice_app.input_audio_stream import LocalAudioStream
from pkg.voice_app.output_audio_stream import LocalAudioProducer

# HF_WS_URL = "wss://your-hf-space-url.hf.space/stream"
HF_WS_URL = "ws://127.0.0.1:8000/stream"


async def voice_client(audio_in_queue: queue.Queue, audio_out_queue: queue.Queue):
    async with websockets.connect(HF_WS_URL) as ws:
        print("Connected to HF Bot API")

        async def sender():
            while True:
                frame = await asyncio.get_event_loop().run_in_executor(None, audio_in_queue.get)
                if frame is None:
                    break
                # frame: np.ndarray of float32 PCM
                await ws.send(json.dumps(frame.tolist()))

        async def receiver():
            async for msg in ws:
                frame = json.loads(msg)
                audio_chunk = np.array(frame, dtype="float32")
                audio_out_queue.put_nowait(audio_chunk)

        await asyncio.gather(sender(), receiver())


def main():
    mic_queue = queue.Queue()
    playback_queue = queue.Queue()

    audio_stream = LocalAudioStream(output_queue=mic_queue)
    audio_producer = LocalAudioProducer(input_queue=playback_queue)

    audio_stream.start()
    audio_producer.start()

    try:
        asyncio.run(voice_client(mic_queue, playback_queue))
    except KeyboardInterrupt:
        pass
    finally:
        audio_stream.stop()
        audio_producer.stop()


if __name__ == "__main__":
    main()
