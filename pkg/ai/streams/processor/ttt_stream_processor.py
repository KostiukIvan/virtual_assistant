import queue
import threading


class TextToTextStreamProcessor:
    """Consumes text from an input queue, processes it with a text-to-text model,
    and places the generated response into an output queue.
    """

    def __init__(
        self,
        ttt_model: object,
        input_stream_queue: queue.Queue,
        output_stream_queue: queue.Queue,
    ) -> None:
        """Initializes the TextToTextStreamProcessor.

        Args:
            ttt_model (object): An object with a `text_to_text(message)` method.
            input_stream_queue (queue.Queue): The queue to get user text from.
            output_stream_queue (queue.Queue): The queue to put bot responses into.

        """
        self.ttt_model = ttt_model
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
        """The main loop for consuming text and generating responses."""
        while self.is_running:
            try:
                # Get transcribed text from the input queue
                data = self.input_stream_queue.get(timeout=1.0)
                user_text = data["data"]
                event = data["event"]

                if event == "L" and user_text is None:
                    self.output_stream_queue.put({"data": None, "event": event})

                if user_text is None:
                    continue

                # Generate a response using the TTT model
                # print(user_text, end="")
                bot_response = self.ttt_model.text_to_text(user_text)
                # print(bot_response, end="")

                # Put the final response into the output queue
                self.output_stream_queue.put({"data": bot_response, "event": event})

            except queue.Empty:
                continue
            except Exception as e:
                print("Error in TTT processing loop", str(e))

    def process_text(self) -> None:
        input_message = ""
        while True:
            try:
                # Get transcribed text from the input queue
                user_text = self.input_stream_queue.get(timeout=1.0)
                input_message += user_text

            except queue.Empty:
                break
            except Exception:
                pass

        # Generate a response using the TTT model
        bot_response = self.ttt_model.text_to_text(input_message)

        # Put the final response into the output queue
        self.output_stream_queue.put(bot_response)
