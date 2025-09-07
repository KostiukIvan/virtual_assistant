# aec_enabled_audio_io.py
import math
import queue
import threading
import time

import numpy as np

from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.output.local.audio_producer import LocalAudioProducer


# ---------------------------
# Acoustic Echo Canceller (FDLMS-ish)
# ---------------------------
class AcousticEchoCanceller:
    """
    Frequency-domain block LMS (FDLMS-like) acoustic echo canceller.

    Design notes:
    - Works block-by-block with frame_size samples (must match capture/playback frame size).
    - Keeps a short history / ring buffer for the playback reference to allow alignment.
    - Uses FFT-based adaptation and overlap-add for efficiency.
    - Parameters exposed for tuning.
    - Thread-safe queue-based I/O, non-blocking where appropriate.
    """

    def __init__(
        self,
        mic_queue: queue.Queue,
        playback_ref_queue: queue.Queue,
        output_queue: queue.Queue,
        sample_rate: int = 16000,
        frame_size: int = 480,  # 30 ms @ 16kHz = 480 samples
        fft_len: int | None = None,
        mu: float = 0.1,  # adaptation step size
        leak: float = 1e-4,
        ref_history_ms: int = 500,
        max_ref_queue: int = 200,
    ):
        self.mic_queue = mic_queue
        self.playback_ref_queue = playback_ref_queue
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.fft_len = fft_len if fft_len is not None else 2 ** int(np.ceil(np.log2(frame_size * 2)))
        self.mu = mu
        self.leak = leak

        # reference ring buffer length in frames
        self.ref_history_frames = max(2, int((ref_history_ms / 1000.0) / (frame_size / sample_rate)))
        self.ref_ring = [np.zeros((frame_size, 1), dtype=np.float32) for _ in range(self.ref_history_frames)]
        self.ref_ring_idx = 0

        # For FFT-domain adaptive filter we keep H (complex) per bin
        self.H = np.zeros((self.fft_len // 2 + 1, 1), dtype=np.complex64)

        # Overlap buffers for OLA (mic and output)
        self.overlap = np.zeros((self.fft_len - self.frame_size, 1), dtype=np.float32)

        # control
        self.is_running = False
        self.thread = None

        # internal small queue to store recent refs (bounded)
        self.local_ref_queue = queue.Queue(maxsize=max_ref_queue)

        # Precompute window
        self.window = np.hanning(self.fft_len).astype(np.float32).reshape(-1, 1)

        # safety clamp for mu
        self.max_mu = 1.0
        self.min_mu = 1e-6

    def start(self) -> None:
        if self.is_running:
            return

        # spawn a small thread that consumes playback_ref_queue and fills a fast local queue (avoid blocking audio thread)
        self.is_running = True
        self._ref_thread = threading.Thread(target=self._ref_collector_loop, daemon=True)
        self._ref_thread.start()

        self.thread = threading.Thread(target=self._aec_loop, daemon=True)
        self.thread.start()
        print("AcousticEchoCanceller started")

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        if self.thread:
            self.thread.join()
        if hasattr(self, "_ref_thread"):
            self._ref_thread.join()
        print("AcousticEchoCanceller stopped")

    def _ref_collector_loop(self) -> None:
        # Collect playback reference frames into a local bounded queue quickly
        while self.is_running:
            try:
                ref = self.playback_ref_queue.get(timeout=0.1)
                try:
                    self.local_ref_queue.put_nowait(ref)
                except queue.Full:
                    # drop if too slow
                    pass
            except queue.Empty:
                continue

    def _aec_loop(self) -> None:
        fft_len = self.fft_len
        frame = self.frame_size
        window = self.window
        H = self.H
        mu = self.mu
        leak = self.leak

        # buffers for block processing
        x_block = np.zeros((fft_len, 1), dtype=np.float32)
        X = np.zeros((fft_len // 2 + 1, 1), dtype=np.complex64)

        while self.is_running:
            try:
                mic_frame = self.mic_queue.get(timeout=0.1)  # blocking briefly
            except queue.Empty:
                continue

            # normalize shape
            mic = np.asarray(mic_frame, dtype=np.float32)
            if mic.ndim > 1:
                mic = mic.reshape(-1, 1)

            # build x_block from available reference frames. We'll pop one ref if available,
            # otherwise zero pad. We keep a small ring to account for small latency jitter.
            ref = None
            try:
                ref = self.local_ref_queue.get_nowait()
            except queue.Empty:
                # no ref available; use zeros
                ref = np.zeros_like(mic)

            ref = np.asarray(ref, dtype=np.float32)

            # get reference frame
            try:
                ref = self.local_ref_queue.get_nowait()
            except queue.Empty:
                ref = np.zeros_like(mic)

            ref = np.asarray(ref, dtype=np.float32)
            if ref.ndim == 1:
                ref = ref.reshape(-1, 1)
            elif ref.ndim > 1:
                ref = ref[:, :1]

            # place ref into ring (for alignment jitter)
            self.ref_ring[self.ref_ring_idx] = ref
            self.ref_ring_idx = (self.ref_ring_idx + 1) % self.ref_history_frames

            # Create analysis frame for FFT
            # Build x_block: put the ref at the beginning and zero pad
            x_block[:frame, 0] = ref[:, 0] if ref.shape[0] >= frame else np.pad(ref[:, 0], (0, frame - ref.shape[0]))
            x_block[frame:, 0] = 0.0

            # FFT
            X = np.fft.rfft(x_block[:, 0] * window[:, 0])
            X = X.reshape(-1, 1).astype(np.complex64)

            # Estimate playback echo: Y_hat = inverseFFT(H * X)
            Y_hat_freq = H * X  # complex
            y_hat_time = np.fft.irfft(Y_hat_freq[:, 0], n=fft_len).reshape(-1, 1) * window

            # take first frame-sized part and apply overlap-add
            y_hat_frame = y_hat_time[:frame, 0].reshape(-1, 1)

            # error (near-end - estimated echo)
            # mic may be longer/shorter; align
            e = mic[:frame, 0].reshape(-1, 1) - y_hat_frame

            # Prepare E_freq for adaptation
            e_block = np.zeros((fft_len, 1), dtype=np.float32)
            e_block[:frame, 0] = e[:, 0]
            E = np.fft.rfft(e_block[:, 0] * window[:, 0]).reshape(-1, 1).astype(np.complex64)

            # Power normalization (avoid divide by small values)
            power = (np.abs(X) ** 2).reshape(-1, 1) + 1e-8

            # NLMS-like update in freq domain: H <- (1 - leak)*H + mu * conj(X) * E / power
            mu_clamped = max(self.min_mu, min(self.max_mu, mu))
            H = (1 - leak) * H + mu_clamped * (np.conj(X) * E) / power

            # store H back
            self.H = H

            # Output the error (near-end cleaned)
            # We also perform a simple DC / low-level clamp to avoid large pops
            e_out = e.copy()
            # soft clip to avoid extremes
            np.clip(e_out, -0.9999, 0.9999, out=e_out)

            # push to output_queue; if full, drop to preserve real-time
            try:
                self.output_queue.put_nowait(e_out)
            except queue.Full:
                # If downstream is blocked, drop the frame (prefer real-time)
                pass

        print("AEC loop exiting")


# ---------------------------
# Demo main() that tests the pipeline
# ---------------------------
def main(real_capture: bool = False) -> None:
    """
    Demo/test harness.
    - If real_capture == True: uses your physical microphone (LocalAudioStream) and plays to speakers.
      The AEC will attempt to remove speaker playback from the mic stream.
    - If real_capture == False: runs a synthetic test where we generate a remote signal that is played,
      then synthesize a near-end speech plus echo captured on the mic. This helps verify AEC performance
      without hardware.
    """

    # Queues:
    mic_raw_queue = queue.Queue(maxsize=200)  # raw microphone frames from LocalAudioStream
    playback_ref_queue = queue.Queue(maxsize=200)  # frames that are played to the speaker (from LocalAudioProducer)
    mic_clean_queue = queue.Queue(maxsize=200)  # cleaned mic frames after AEC (to be consumed by bot/transport)
    playback_in_queue = queue.Queue(maxsize=200)  # incoming audio to play (remote agent -> speaker)

    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

    # 1) instantiate components
    audio_stream = LocalAudioStream(output_queue=mic_raw_queue, sample_rate=SAMPLE_RATE, frame_duration_ms=FRAME_MS)
    audio_producer = LocalAudioProducer(
        input_queue=playback_in_queue,
        playback_ref_queue=playback_ref_queue,
        sample_rate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    # 2) AEC
    aec = AcousticEchoCanceller(
        mic_queue=mic_raw_queue,
        playback_ref_queue=playback_ref_queue,
        output_queue=mic_clean_queue,
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SAMPLES,
        mu=0.12,
        leak=1e-4,
    )

    # Start AEC and producer
    aec.start()
    audio_producer.start()

    # Start capture only if we want real mic
    if real_capture:
        audio_stream.start()

    # Test generator thread: if not real_capture, feed the producer and simulate mic raw queue with echoed signal
    stop_flag = threading.Event()

    def synth_playback_and_mic_simulator():
        """
        When real_capture == False, this thread:
        - generates a continuous sine or speech-like signal as 'remote' audio -> sent to playback_in_queue
        - constructs a synthetic mic capture = near_end_speech + alpha * delayed(playback)
        - pushes the synthetic mic frames to mic_raw_queue so AEC can process them
        """
        t = 0.0
        dt = FRAME_SAMPLES / SAMPLE_RATE
        # generate a 440 Hz tone as remote audio
        freq = 440.0
        playback_delay_frames = 2  # simulate echo path delay (in frames)
        # circular buffer for playback history to create echo
        playback_ring = [np.zeros((FRAME_SAMPLES, 1), dtype=np.float32) for _ in range(10)]
        idx = 0
        while not stop_flag.is_set():
            # remote played audio chunk
            times = np.arange(FRAME_SAMPLES) / SAMPLE_RATE + t
            remote = 0.3 * np.sin(2.0 * math.pi * freq * times).astype(np.float32).reshape(-1, 1)
            t += dt

            # feed remote into playback_in_queue (played to speaker)
            try:
                playback_in_queue.put_nowait({"data": remote, "event": "L"})
            except queue.Full:
                pass

            # emulate echo: echo = alpha * (remote delayed by some frames)
            playback_ring[idx] = remote
            echo_idx = (idx - playback_delay_frames) % len(playback_ring)
            echo = 0.6 * playback_ring[echo_idx]  # echo amplitude
            idx = (idx + 1) % len(playback_ring)

            # near-end speech: short burst randomly
            near = np.zeros_like(remote)
            if (int(time.time() * 5) % 20) < 3:
                # burst of speech-like component
                near[:, 0] = 0.15 * np.random.randn(FRAME_SAMPLES)

            # captured mic = near + echo + small noise
            mic_sim = near + echo + 0.005 * np.random.randn(FRAME_SAMPLES, 1).astype(np.float32)

            # push into mic_raw_queue so the AEC consumes it
            try:
                mic_raw_queue.put_nowait(mic_sim)
            except queue.Full:
                pass

            # small sleep to emulate real-time
            time.sleep(dt * 0.9)

    synth_thread = None
    if not real_capture:
        synth_thread = threading.Thread(target=synth_playback_and_mic_simulator, daemon=True)
        synth_thread.start()
        print("Started synthetic playback+mic simulator")

    # Consumer thread that reads cleaned mic frames and does simple logging (simulate sending to bot)
    def mic_consumer():
        count = 0
        time.time()
        while not stop_flag.is_set():
            try:
                cleaned = mic_clean_queue.get(timeout=0.5)
                # Here you'd normally send `cleaned` to your STT/agent/pipeline
                count += 1
                if count % 50 == 0:
                    print("Consumed %d cleaned mic frames (latest RMS %.5f)", count, np.sqrt(np.mean(cleaned**2)))
            except queue.Empty:
                continue

    consumer_thread = threading.Thread(target=mic_consumer, daemon=True)
    consumer_thread.start()

    # Run for N seconds, observing logs
    try:
        RUN_SECONDS = 20
        print("Running demo for %d seconds (real_capture=%s)...", RUN_SECONDS, real_capture)
        time.sleep(RUN_SECONDS)
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    finally:
        # stop everything
        stop_flag.set()
        if synth_thread:
            synth_thread.join(timeout=1.0)
        aec.stop()
        audio_producer.stop()
        if real_capture:
            audio_stream.stop()
        print("Demo finished")


if __name__ == "__main__":
    # Change to True to use real microphone input
    main(real_capture=False)
