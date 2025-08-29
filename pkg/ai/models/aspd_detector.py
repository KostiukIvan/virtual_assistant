import collections
import queue

import numpy as np
import sounddevice as sd
import webrtcvad

from pkg.utils import float_to_pcm16


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
        """Initializes the advanced pause detector.

        Args:
            sample_rate (int): The sample rate of the audio stream (8000, 16000, 32000, 48000).
            frame_duration_ms (int): The duration of each audio chunk in ms (10, 20, 30).
            vad_level (int): The aggressiveness of the VAD (0=least, 3=most aggressive).
            short_pause_ms (int): Min duration of silence for a "short pause".
            long_pause_ms (int): Min duration of silence for a "long pause".
            history_frames (int): Number of recent frames to keep for context.

        """
        if sample_rate not in [8000, 16000, 32000, 48000]:
            msg = "Unsupported sample rate. Must be 8k, 16k, 32k, or 48k."
            raise ValueError(msg)
        if frame_duration_ms not in [10, 20, 30]:
            msg = "Unsupported frame duration. Must be 10, 20, or 30 ms."
            raise ValueError(msg)

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)

        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_level)

        # Convert pause durations from milliseconds to frame counts
        self._min_short_pause_frames = short_pause_ms / frame_duration_ms
        self._min_long_pause_frames = long_pause_ms / frame_duration_ms

        # State tracking variables
        self.consecutive_silent_frames = 0
        self.pause_event_triggered = None  # Can be 'short' or 'long'

        # Buffer to hold recent history of speech/silence results
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
            self.consecutive_silent_frames = 0
            self.pause_event_triggered = None
            return "SPEECH"
        # If we detect silence, increment the counter
        self.consecutive_silent_frames += 1

        # This logic triggers the event only *once* when the threshold is first crossed.

        # Check for long pause first
        if self.consecutive_silent_frames >= self._min_long_pause_frames:
            if self.pause_event_triggered != "long":
                self.pause_event_triggered = "long"
                return "LONG_PAUSE"

        # Check for short pause
        elif self.consecutive_silent_frames >= self._min_short_pause_frames:
            if self.pause_event_triggered is None:
                self.pause_event_triggered = "short"
                return "SHORT_PAUSE"

        return "SILENCE"

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        return self.process_chunk(audio_chunk=audio_chunk) == "SPEECH"

    def is_short_pause(self, audio_chunk: np.ndarray) -> bool:
        return self.process_chunk(audio_chunk=audio_chunk) == "SHORT_PAUSE"

    def is_long_pause(self, audio_chunk: np.ndarray) -> bool:
        return self.process_chunk(audio_chunk=audio_chunk) == "LONG_PAUSE"

    def is_silence(self, audio_chunk: np.ndarray) -> bool:
        return self.process_chunk(audio_chunk=audio_chunk) == "SILENCE"


def main() -> None:
    """Example of using the detector with a live microphone stream."""
    # --- Configuration ---
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30

    detector = AdvancedSpeechPauseDetector(
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        vad_level=3,
        short_pause_ms=250,
        long_pause_ms=1000,
    )

    frame_samples = detector.frame_samples
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time, status) -> None:
        """This is called from a separate thread for each audio block."""
        if status:
            pass
        audio_queue.put(indata.copy())

    try:
        with sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=frame_samples,
            dtype="float32",
            callback=audio_callback,
        ):
            while True:
                audio_chunk = audio_queue.get()
                status = detector.process_chunk(audio_chunk)

                # Print only the "event" states to avoid cluttering the console
                if status not in ["SPEECH", "SILENCE"]:
                    detector.consecutive_silent_frames * FRAME_DURATION_MS

    except KeyboardInterrupt:
        pass
    except Exception:
        pass


if __name__ == "__main__":
    # Make sure you have the necessary libraries installed:
    # pip install sounddevice numpy webrtcvad
    main()
