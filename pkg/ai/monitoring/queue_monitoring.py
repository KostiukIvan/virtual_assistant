import queue
import threading
import time

from rich.live import Live
from rich.table import Table

"""
QUEUES = {
    "PLAYBACK_IN": PLAYBACK_IN_QUEUE,
    "STT_INPUT": STT_INPUT_QUEUE,
    "TTT_INPUT": TTT_INPUT_QUEUE,
    "TTS_INPUT": TTS_INPUT_QUEUE,
}
"""


class QueueMonitor:
    def __init__(self, queues: dict[str, queue.Queue], interval: float = 1.0):
        self.queues = queues
        self.interval = interval
        self._stop_event = threading.Event()

    def _make_table(self) -> Table:
        table = Table(title="Queue Monitor")
        table.add_column("Queue", justify="left")
        table.add_column("Size", justify="right")
        table.add_column("Maxsize", justify="right")

        for name, q in self.queues.items():
            size = q.qsize()
            maxsize = q.maxsize if q.maxsize > 0 else "∞"
            table.add_row(name, str(size), str(maxsize))
        return table

    def start(self):
        def loop():
            with Live(refresh_per_second=4) as live:
                while not self._stop_event.is_set():
                    live.update(self._make_table())
                    time.sleep(self.interval)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        if hasattr(self, "thread"):
            self.thread.join()
