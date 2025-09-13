import asyncio
import time


class AECWorker:
    """An Async AEC worker that outputs mic frames only when the speaker is silent."""

    def __init__(
        self,
        mic_queue: asyncio.Queue,
        playback_ref_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        frame_size=320,
        sample_rate=16000,
        grace_ms=100,
    ):
        self.mic_queue = mic_queue
        self.playback_ref_queue = playback_ref_queue
        self.output_queue = output_queue

        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.grace = grace_ms * 1e6

        self.speaker_active_until = 0.0
        self.frame_duration = frame_size / float(sample_rate) * 1e9

        self.running = True
        self._task: asyncio.Task | None = None

    async def run(self):
        try:
            while self.running:
                now = time.monotonic_ns()

                try:
                    while True:
                        speaker_timestamp, speaker_frame = self.playback_ref_queue.get_nowait()
                        frame_end = speaker_timestamp + self.frame_duration
                        self.speaker_active_until = max(self.speaker_active_until, frame_end + self.grace)
                        self.playback_ref_queue.task_done()
                except asyncio.QueueEmpty:
                    pass

                try:
                    mic_timestamp, mic_frame = self.mic_queue.get_nowait()

                    if now >= self.speaker_active_until:
                        await self.output_queue.put(mic_frame)
                    else:
                        pass
                    self.mic_queue.task_done()
                except asyncio.QueueEmpty:
                    pass

                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            print("AEC worker was cancelled.")

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
