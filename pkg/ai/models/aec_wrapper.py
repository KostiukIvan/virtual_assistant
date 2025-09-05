
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from typing import Optional, Callable

# Import your pause detector
from pkg.ai.models.aspd_detector import AdvancedSpeechPauseDetector  # your class path

# -------------------------
# AEC Adapter (plug your library here)
# -------------------------
class AECWrapper:
    """
    Adapter around an AEC library.
    Replace internals with actual binding calls (webRTC AEC / speexdsp / etc).
    API assumptions:
      - process_render(int16_block): feed the exact frames you play to speakers.
      - process_capture(int16_block) -> int16_block: returns cleaned capture.
      - optionally has_speech_from_capture() -> bool or similar; we'll use VAD separately.
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1, frame_ms: int = 10):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        # Example placeholders for internal objects:
        # self._aec = SomeAECBinding(sample_rate, channels)
        # self._lock = threading.Lock()  # if binding not thread-safe
        # self._vad = OptionalVAD()     # you will still use your AdvancedSpeechPauseDetector

    def process_render(self, int16_block: np.ndarray) -> None:
        """
        Feed the far-end (render) block into the AEC.
        int16_block: 1-D np.ndarray dtype=np.int16, length == frame_samples * channels
        """
        # Example:
        # with self._lock:
        #     self._aec.add_farend(int16_block.tobytes())
        # Replace above with actual binding call.
        return

    def process_capture(self, int16_block: np.ndarray) -> np.ndarray:
        """
        Process the captured microphone block and return cleaned int16 numpy array.
        This should run AEC (and optionally NS) inside it.
        """
        # Example:
        # with self._lock:
        #     cleaned_bytes = self._aec.process_capture(int16_block.tobytes())
        # cleaned = np.frombuffer(cleaned_bytes, dtype=np.int16)
        # return cleaned

        # Placeholder — pass-through (replace with real call)
        return int16_block

    # Optionally expose metrics such as a residual echo energy useful for barge-in
    def get_last_residual_energy(self) -> float:
        # binding dependent, return a float energy estimate, 0..1
        return 0.0
