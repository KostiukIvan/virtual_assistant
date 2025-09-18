import asyncio

import sounddevice as sd

import pkg.config as config


async def mic_producer(queue: asyncio.Queue):
    """Reads audio frames from microphone and puts them into the queue."""
    with sd.InputStream(
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=config.AUDIO_CHANNELS,
        dtype=config.AUDIO_DTYPE,
        blocksize=config.AUDIO_FRAME_SAMPLES,
    ) as stream:
        print("🎙️ Microphone stream started. Speak into the mic...")
        try:
            while True:
                audio_chunk, _ = stream.read(config.AUDIO_FRAME_SAMPLES)
                await queue.put(audio_chunk)
        except asyncio.CancelledError:
            print("Mic producer cancelled.")
