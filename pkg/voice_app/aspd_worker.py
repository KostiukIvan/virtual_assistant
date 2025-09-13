import asyncio
import collections
import sys

import numpy as np
import webrtcvad

from pkg.voice_app.aec_worker import AECWorker
from pkg.voice_app.input_audio_stream import LocalAudioStream
from pkg.voice_app.output_audio_stream import LocalAudioProducer


# Since the original external package 'pkg.utils' is not available,
# we'll provide a mock implementation of float_to_pcm16.
def float_to_pcm16(audio_chunk: np.ndarray) -> bytes:
    """Converts a numpy array of float32 audio samples to PCM 16-bit bytes."""
    # Convert from float32 to int16
    pcm_int16 = (audio_chunk * 32767).astype(np.int16)
    # Convert to bytes
    return pcm_int16.tobytes()


class AdvancedSpeechPauseDetector:
    """An advanced class to detect short and long pauses in a real-time audio stream
    using Google's WebRTC Voice Activity Detector (VAD).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_level: int = 3,
        short_pause_ms: int = 200,
        long_pause_ms: int = 500,
        history_frames: int = 10,
    ) -> None:
        if sample_rate not in [8000, 16000, 32000, 48000]:
            msg = "Unsupported sample rate. Must be 8k, 16k, 32k, or 48k."
            raise ValueError(msg)
        if frame_duration_ms not in [10, 20, 30]:
            msg = "Unsupported frame duration. Must be 10, 20, or 30 ms."
            raise ValueError(msg)

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.vad = webrtcvad.Vad(vad_level)
        self._min_short_pause_frames = short_pause_ms / frame_duration_ms
        self._min_long_pause_frames = long_pause_ms / frame_duration_ms
        self.consecutive_silent_frames = 0
        self.pause_event_triggered = None
        self.at_least_one_speech = False
        self.history = collections.deque(maxlen=history_frames)

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
            is_speech = self.vad.is_speech(pcm16_chunk, self.sample_rate)
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
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_level: int = 3,
        short_pause_ms: int = 250,
        long_pause_ms: int = 600,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.detector = AdvancedSpeechPauseDetector(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            vad_level=vad_level,
            short_pause_ms=short_pause_ms,
            long_pause_ms=long_pause_ms,
        )

        self._processing_task: asyncio.Task = None
        self.current_buffer = []

    async def _processing_loop(self) -> None:
        try:
            while True:
                audio_chunk = await self.input_queue.get()
                self.input_queue.task_done()

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


async def main() -> None:
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 60
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    VAD_LEVEL = 3
    SHORT_PAUSE_MS = 300
    LONG_PAUSE_MS = 1000

    # 1. Setup the queues to connect our components
    mic_raw_queue = asyncio.Queue()
    mic_cleaned_queue = asyncio.Queue()
    playback_ref_queue = asyncio.Queue()
    playback_queue = asyncio.Queue()
    stt_input_queue = asyncio.Queue()

    # 2. Initialize our components
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

    # 3. Start the pipelines
    audio_stream.start()
    audio_producer.start()
    aec.start()
    stream_detector.start()

    try:
        # 4. The main loop consumes from the final output queue
        print("Starting asyncio audio pipeline. Press Ctrl+C to stop.")
        while True:
            event_data = await stt_input_queue.get()
            stt_input_queue.task_done()

            event = event_data.get("event")
            data = event_data.get("data")

            if event == "s":
                print(f"\n[EVENT] Short pause detected with {len(data)} samples.")
            elif event == "L":
                print("\n[EVENT] Long pause detected.")

    except KeyboardInterrupt:
        print("\nStopping gracefully...")
    finally:
        # 5. Stop the pipelines
        audio_stream.stop()
        await audio_producer.stop()
        aec.stop()
        await stream_detector.stop()


if __name__ == "__main__":
    asyncio.run(main())
