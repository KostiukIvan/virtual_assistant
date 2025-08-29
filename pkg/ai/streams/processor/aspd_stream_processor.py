import queue
import sys
import threading

import numpy as np

from pkg.ai.models.aspd_detector import AdvancedSpeechPauseDetector
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.processor.helper import tts_finished_its_speech


class AdvancedSpeechPauseDetectorStream:
    """Consumes audio frames from an input queue, detects speech pauses,
    and puts event messages into an output queue.
    """

    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        long_pause_callback: callable,
        short_pause_callback: callable,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_level: int = 3,
        short_pause_ms: int = 250,
        long_pause_ms: int = 600,
    ) -> None:
        """Initializes the stream processor.

        Args:
            input_queue (queue.Queue): Queue to get audio frames from.
            output_queue (queue.Queue): Queue to put pause event strings into.
            All other args are passed to the AdvancedSpeechPauseDetector.

        """
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.long_pause_callback = long_pause_callback
        self.short_pause_callback = short_pause_callback

        self.detector = AdvancedSpeechPauseDetector(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            vad_level=vad_level,
            short_pause_ms=short_pause_ms,
            long_pause_ms=long_pause_ms,
        )

        self.is_running = False
        self.thread = None
        self.current_buffer = []

    def _processing_loop(self) -> None:
        """The main loop for processing audio from the input queue."""
        while self.is_running:
            try:
                # Get a chunk of audio from the input queue
                audio_chunk = self.input_queue.get(timeout=1.0)

                # Process the chunk to detect pauses
                status = self.detector.process_chunk(audio_chunk)

                if status == "SILENCE":
                    print(".", end="")
                    sys.stdout.flush()

                if status == "SPEECH" and not tts_finished_its_speech.is_set():
                    print("^", end="")
                    sys.stdout.flush()
                    self.current_buffer.extend(audio_chunk.flatten())
                    tts_finished_its_speech.clear()

                # In the _processing_loop method
                if status == "SHORT_PAUSE":
                    if self.current_buffer:  # Add this check
                        audio_np = np.array(self.current_buffer, dtype=np.float32)
                        self.output_queue.put(audio_np)
                        self.current_buffer = []
                        self.short_pause_callback()

                if status == "LONG_PAUSE":
                    self.long_pause_callback()

            except queue.Empty:
                # If the input queue is empty, just continue waiting
                continue
            except Exception:
                break

    def start(self) -> None:
        """Starts the processing in a separate thread."""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops the processing thread."""
        self.is_running = False
        if self.thread:
            self.thread.join()  # Wait for the thread to finish


def main() -> None:
    """Example of using LocalAudioStream to feed an AdvancedSpeechPauseDetectorStream."""
    # 1. Create the queues to connect the components
    mic_output_queue = queue.Queue()
    detector_output_queue = queue.Queue()

    # 2. Initialize the local audio stream to capture mic input
    audio_stream = LocalAudioStream(output_queue=mic_output_queue)

    # 3. Initialize the pause detector stream
    # It will consume from the mic's output queue and produce to its own output queue.
    stream_detector = AdvancedSpeechPauseDetectorStream(
        input_queue=mic_output_queue,
        output_queue=detector_output_queue,
        long_pause_callback=lambda: (print("L", end=""), sys.stdout.flush()),
        short_pause_callback=lambda: (print("s", end=""), sys.stdout.flush()),
        sample_rate=16000,
        frame_duration_ms=30,
        vad_level=3,
        short_pause_ms=200,
        long_pause_ms=700,
    )

    # 4. Start both processing threads
    audio_stream.start()
    stream_detector.start()

    try:
        # 5. The main thread can now consume the final processed audio from the detector
        while True:
            try:
                # Get the processed frame from the final queue
                detector_output_queue.get(timeout=1.0)
                # For this demo, we just print a dot to show that frames are flowing through.

            except queue.Empty:
                # This is normal if there's a brief gap in processing
                continue

    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        # 6. Stop the streams gracefully
        audio_stream.stop()
        stream_detector.stop()


if __name__ == "__main__":
    # To run this, you need sounddevice, numpy, and webrtcvad:
    # pip install sounddevice numpy webrtcvad-wheels
    main()
