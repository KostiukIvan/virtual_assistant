import asyncio
import collections
import sys

import numpy as np
import sounddevice as sd
import webrtcvad

import pkg.config as config
from pkg.utils import float_to_pcm16


class AdvancedSpeechPauseDetector:
    """An advanced class to detect short and long pauses in a real-time audio stream
    using Google's WebRTC Voice Activity Detector (VAD).
    """

    def __init__(
        self,
    ) -> None:
        if config.AUDIO_SAMPLE_RATE not in [8000, 16000, 32000, 48000]:
            msg = "Unsupported sample rate. Must be 8k, 16k, 32k, or 48k."
            raise ValueError(msg)
        if config.AUDIO_FRAME_DURATION_MS not in [10, 20, 30]:
            msg = "Unsupported frame duration. Must be 10, 20, or 30 ms."
            raise ValueError(msg)

        self.vad = webrtcvad.Vad(config.AUDIO_VAD_LEVEL)
        self._min_short_pause_frames = config.AUDIO_SHORT_PAUSE_MS / config.AUDIO_FRAME_DURATION_MS
        self._min_long_pause_frames = config.AUDIO_LONG_PAUSE_MS / config.AUDIO_FRAME_DURATION_MS
        self.consecutive_silent_frames = 0
        self.pause_event_triggered = None
        self.at_least_one_speech = False
        self.history = collections.deque(maxlen=config.AUDIO_HISTORY_FRAMES)

    def process_chunk(self, audio_chunk: np.ndarray) -> str:
        """Processes a single chunk of audio and returns the current speech state.

        Args:
            audio_chunk (np.ndarray): A numpy array of audio samples (float32).

        Returns:
            str: The detected state, one of "SPEECH", "SILENCE",
                 "SHORT_PAUSE", or "LONG_PAUSE".

        """
        try:
            # Convert audio to required PCM16 format for VAD
            pcm16_chunk = float_to_pcm16(audio_chunk)
            is_speech = self.vad.is_speech(pcm16_chunk, config.AUDIO_SAMPLE_RATE)
            self.history.append(is_speech)
        except Exception:
            return "ERROR"

        # --- State Logic ---
        # Simple smoothing: consider it speech if a certain number of recent frames were speech
        is_speaking = sum(self.history) > (self.history.maxlen / 2)

        if is_speaking:
            # If we detect speech, reset the silence counter and any triggered events
            if not self.at_least_one_speech:
                self.at_least_one_speech = True
            self.consecutive_silent_frames = 0
            self.pause_event_triggered = None
            return "SPEECH"
        # If we detect silence, increment the counter
        self.consecutive_silent_frames += 1

        # This logic triggers the event only *once* when the threshold is first crossed.

        # Check for long pause first
        if self.at_least_one_speech and self.consecutive_silent_frames >= self._min_long_pause_frames:
            if self.pause_event_triggered != "long":
                self.pause_event_triggered = "long"
                return "LONG_PAUSE"

        # Check for short pause
        elif self.at_least_one_speech and self.consecutive_silent_frames >= self._min_short_pause_frames:
            if self.pause_event_triggered is None:
                self.pause_event_triggered = "short"
                return "SHORT_PAUSE"

        return "SILENCE"


class AdvancedSpeechPauseDetectorAsyncStream:
    """Consumes audio frames from an asyncio input queue, detects speech pauses,
    and puts event messages into an asyncio output queue.
    """

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.detector = AdvancedSpeechPauseDetector()

        self._processing_task: asyncio.Task = None
        self.current_buffer = []

    async def _processing_loop(self) -> None:
        try:
            while True:
                audio_chunk = await self.input_queue.get()

                if audio_chunk is None:
                    continue

                await self.output_queue.put({"data": audio_chunk.flatten(), "event": "p"})

                status = self.detector.process_chunk(audio_chunk)

                if status == "SILENCE":
                    sys.stdout.write(".")
                    sys.stdout.flush()
                if status == "SPEECH":
                    sys.stdout.write("^")
                    sys.stdout.flush()

                if status == "SHORT_PAUSE":
                    sys.stdout.write("s")
                    await self.output_queue.put({"data": None, "event": "s"})

                if status == "LONG_PAUSE":
                    sys.stdout.write("L")
                    await self.output_queue.put({"data": None, "event": "L"})

        except asyncio.CancelledError:
            print("Processing loop was cancelled.")
        except Exception as e:
            print(f"An error occurred in the processing loop: {e}", file=sys.stderr)

    def start(self) -> None:
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(self._processing_loop())

    async def stop(self) -> None:
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass


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


async def main():
    mic_queue = asyncio.Queue(maxsize=1)
    event_queue = asyncio.Queue(maxsize=1)

    detector = AdvancedSpeechPauseDetectorAsyncStream(mic_queue, event_queue)
    detector.start()

    mic_task = asyncio.create_task(mic_producer(mic_queue))

    try:
        while True:
            await event_queue.get()

            # else: ignore "p" (raw chunks) for now
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic_task.cancel()
        await detector.stop()


if __name__ == "__main__":
    asyncio.run(main())
