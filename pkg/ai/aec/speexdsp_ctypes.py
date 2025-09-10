import ctypes
import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd


class SpeexAEC:
    """Low-level SpeexDSP AEC wrapper using ctypes"""

    def __init__(self, frame_size=160, filter_length=8000, sample_rate=16000):
        self.lib = ctypes.cdll.LoadLibrary("/opt/homebrew/lib/libspeexdsp.dylib")

        self.lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.speex_echo_state_init.restype = ctypes.c_void_p

        self.lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
        self.lib.speex_echo_state_destroy.restype = None

        self.lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.speex_echo_ctl.restype = ctypes.c_int

        self.lib.speex_echo_cancellation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_short),
            ctypes.POINTER(ctypes.c_short),
            ctypes.POINTER(ctypes.c_short),
        ]
        self.lib.speex_echo_cancellation.restype = None

        # Init AEC
        self.state = self.lib.speex_echo_state_init(frame_size, filter_length)
        if not self.state:
            raise RuntimeError("Failed to init Speex echo state")

        sr = ctypes.c_int(sample_rate)
        ret = self.lib.speex_echo_ctl(self.state, 24, ctypes.byref(sr))  # SPEEX_ECHO_SET_SAMPLING_RATE
        if ret != 0:
            raise RuntimeError("Failed to set sample rate")

        self.frame_size = frame_size

    def process(self, mic_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
        mic = mic_frame.astype(np.int16, copy=False)
        ref = ref_frame.astype(np.int16, copy=False)
        out = np.zeros(self.frame_size, dtype=np.int16)

        self.lib.speex_echo_cancellation(
            self.state,
            mic.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
            ref.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_short)),
        )
        return out

    def __del__(self):
        if hasattr(self, "state") and self.state:
            self.lib.speex_echo_state_destroy(self.state)
            self.state = None


def int16_to_float32(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 32768.0


class AECWorker(threading.Thread):
    """Threaded AEC worker with optimized params + float in/out + timestamp alignment"""

    def __init__(
        self,
        mic_queue: queue.Queue,
        playback_ref_queue: queue.Queue,
        output_queue: queue.Queue,
        frame_size=320,
        filter_length=8000,
        sample_rate=16000,
        max_buffer_ms=400,
    ):
        super().__init__(daemon=True)
        self.mic_queue = mic_queue
        self.playback_ref_queue = playback_ref_queue
        self.output_queue = output_queue

        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.max_buffer_ns = int(max_buffer_ms * 1e6)  # ns
        self.running = True
        self.delay_frames = 0  # calibrated delay

        # AEC
        self.aec = SpeexAEC(frame_size, filter_length, sample_rate)

        # Playback buffer (timestamp, frame)
        self.ref_buffer = deque()

        # Initial calibration
        duration = 1.0
        chirp = generate_chirp(duration, self.sample_rate)

        # Play & record (mono)
        recorded = sd.playrec(chirp, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        recorded = recorded.flatten()

        # Run calibration
        self.calibrate_delay(playback_signal=chirp, mic_signal=recorded)

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert float [-1,1] or int16 to int16, resize to frame_size"""
        if isinstance(frame, bytes):
            frame = np.frombuffer(frame, dtype=np.int16)

        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, -1.0, 1.0)
            frame = (frame * 32767).astype(np.int16)

        if frame.shape[0] != self.frame_size:
            if frame.shape[0] > self.frame_size:
                frame = frame[: self.frame_size]
            else:
                frame = np.pad(frame, (0, self.frame_size - frame.shape[0]), "constant")
        return frame.astype(np.int16, copy=False)

    def calibrate_delay(self, playback_signal: np.ndarray, mic_signal: np.ndarray):
        """Estimate fixed system delay (samples)"""
        p = playback_signal.astype(np.float32)
        m = mic_signal.astype(np.float32)
        corr = np.correlate(m, p, mode="full")
        lag = np.argmax(corr) - (len(p) - 1)
        self.delay_frames = max(0, int(lag))
        print(
            f"[AEC] Calibrated delay: {self.delay_frames} samples "
            f"(~{self.delay_frames / self.sample_rate * 1000:.1f} ms)"
        )

    def run(self):
        zero_ref = np.zeros(self.frame_size, dtype=np.int16)

        while self.running:
            try:
                ts_mic, mic_frame = self.mic_queue.get(timeout=1.0)
                mic_frame = self._normalize_frame(mic_frame)
                ts_mic_shifted = ts_mic - int(self.delay_frames / self.sample_rate * 1e9)

                # Drain playback queue into buffer
                try:
                    while True:
                        ts_ref, ref_frame = self.playback_ref_queue.get_nowait()
                        ref_frame = self._normalize_frame(ref_frame)
                        self.ref_buffer.append((ts_ref, ref_frame))
                except queue.Empty:
                    pass

                # Remove old playback frames
                cutoff = ts_mic_shifted - self.max_buffer_ns
                while self.ref_buffer and self.ref_buffer[0][0] < cutoff:
                    self.ref_buffer.popleft()

                # Find closest ref frame
                ref_to_use = zero_ref
                if self.ref_buffer:
                    closest = min(self.ref_buffer, key=lambda x: abs(x[0] - ts_mic_shifted))
                    ref_to_use = closest[1]

                # Run AEC
                clean_frame_int16 = self.aec.process(mic_frame, ref_to_use)
                if np.all(clean_frame_int16 == 0):
                    clean_frame_int16 = mic_frame

                # Convert back to float32 [-1,1]
                clean_frame_float = int16_to_float32(clean_frame_int16)

                print(sum(clean_frame_float) / len(clean_frame_float))
                self.output_queue.put(clean_frame_float)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"AECWorker error: {e}")
                break

    def stop(self):
        self.running = False


def generate_chirp(duration=1.0, sample_rate=16000, f0=500, f1=4000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    chirp = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
    return (chirp * 0.7).astype(np.float32)  # float32 in [-1, 1]
