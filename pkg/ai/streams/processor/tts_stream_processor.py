import queue
import sys
import threading

import sounddevice as sd


class TextToSpeechStreamProcessor:
    """Consumes text from an input queue, converts it to speech using a TTS model,
    and plays the resulting audio.
    """

    def __init__(
        self,
        tts_model: object,
        input_stream_queue: queue.Queue,
        output_stream_queue: queue.Queue,
    ) -> None:
        """Initializes the TextToSpeechStreamProcessor.

        Args:
            tts_model (object): An object with a `text_to_speech(text)` method.
            input_stream_queue (queue.Queue): The queue to get bot responses from.
            output_stream_queue (queue.Queue): The queue to receive voice responses

        """
        self.tts_model = tts_model
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
            # Stop any currently playing audio
            sd.stop()
            self.thread.join()

    def _processing_loop(self) -> None:
        """Main loop for consuming text and playing audio on demand."""
        buffer = []
        events = set()
        while self.is_running:
            # Get bot response text if available
            try:
                data = self.input_stream_queue.get(timeout=0.2)
                bot_response = data["data"]
                events.add(data["event"])
                print("TTS E = ", data["event"], events)
                sys.stdout.flush()
                if bot_response:
                    buffer.append(bot_response)
            except queue.Empty:
                pass

            if "L" in events:
                for text in buffer:
                    try:
                        audio_output = self.tts_model.text_to_speech(text)

                        self.output_stream_queue.put({"data": audio_output, "event": "L"})
                    except Exception:
                        continue
                buffer.clear()
                events = set()
