import queue
import threading
import time


class AECWorker(threading.Thread):
    """Threaded AEC worker: outputs mic frames only when speaker is silent"""

    def __init__(
        self,
        mic_queue: queue.Queue,
        playback_ref_queue: queue.Queue,
        output_queue: queue.Queue,
        frame_size=320,
        sample_rate=16000,
        grace_ms=100,
    ):
        super().__init__(daemon=True)
        self.mic_queue = mic_queue
        self.playback_ref_queue = playback_ref_queue
        self.output_queue = output_queue

        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.grace = grace_ms * 1e6

        # Track how long speaker is active
        self.speaker_active_until = 0.0
        self.frame_duration = frame_size / float(sample_rate) * 1e9

        self.running = True

    def run(self):
        while self.running:
            now = time.monotonic_ns()

            # ---- Handle speaker frames ----
            try:
                while True:
                    speaker_timestamp, speaker_frame = self.playback_ref_queue.get_nowait()

                    # Compute when this speaker frame will finish in real time
                    frame_end = speaker_timestamp + self.frame_duration
                    self.speaker_active_until = max(self.speaker_active_until, frame_end + self.grace)
            except queue.Empty:
                pass

            # ---- Handle mic frames ----
            try:
                mic_timestamp, mic_frame = self.mic_queue.get_nowait()

                # Only forward mic if wall-clock time is past speaker_active_until
                if now >= self.speaker_active_until:
                    self.output_queue.put(mic_frame)
                else:
                    # Drop mic frame (speaker is still active in real time)
                    pass
            except queue.Empty:
                pass

            time.sleep(0.001)

    def stop(self):
        self.running = False
