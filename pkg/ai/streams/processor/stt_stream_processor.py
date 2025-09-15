import queue
import threading

import pkg.config as config


class SpeechToTextStreamProcessor:
    """Processes chunks of audio for speech-to-text transcription.
    It consumes audio chunks from an input queue and places transcribed text
    into an output queue.
    """

    def __init__(
        self,
        stt_model: object,
        input_stream_queue: queue.Queue,
        output_stream_queue: queue.Queue,
    ) -> None:
        """Initializes the SpeechToTextStreamProcessor.

        Args:
            stt_model (object): An object with an `audio_to_text(buffer, sample_rate)` method.
            input_stream_queue (queue.Queue): The queue to get audio chunks from.
            output_stream_queue (queue.Queue): The queue to put transcribed text into.

        """
        self.stt_model = stt_model
        self.input_stream_queue = input_stream_queue
        self.output_stream_queue = output_stream_queue

        self.is_running = False
        self.thread = None

    def start(self) -> None:
        """Starts the processor in a separate thread."""
        if self.is_running:
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops the processor thread."""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _processing_loop(self) -> None:
        """The main loop for consuming audio chunks, transcribing them,
        and putting the result in the output queue.
        """
        while self.is_running:
            try:
                data = self.input_stream_queue.get(timeout=1.0)
                audio_chunk = data["data"]
                event = data["event"]

                if event == "L" and audio_chunk is None:
                    self.output_stream_queue.put({"data": None, "event": "L"})

                if audio_chunk is None:
                    continue

                # Perform speech-to-text on the received chunk
                text = self.stt_model.audio_to_text(
                    audio_chunk.flatten(),
                    sample_rate=config.AUDIO_SAMPLE_RATE,
                )
                if text:
                    # Place the final text into the output queue
                    self.output_stream_queue.put({"data": text, "event": event})
                else:
                    self.output_stream_queue.put({"data": None, "event": event})

            except queue.Empty:
                # This is expected when there's no speech.
                continue
            except Exception as e:
                print(e)
