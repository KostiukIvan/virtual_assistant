import queue
import threading

from pkg.ai.models.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.ai.models.ttt_model import LocalTextToTextModel, RemoteTextToTextModel
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.processor.aspd_stream_processor import (
    AdvancedSpeechPauseDetectorStream,
)
from pkg.ai.streams.processor.stt_stream_processor import SpeechToTextStreamProcessor
from pkg.config import (
    HF_API_TOKEN,
    STT_MODE,
    STT_MODEL_LOCAL,
    STT_MODEL_REMOTE,
    TTT_MODE,
    TTT_MODEL_LOCAL,
    TTT_MODEL_REMOTE,
    device,
)


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
                print(user_text, end="")
                bot_response = self.ttt_model.text_to_text(user_text)
                print(bot_response, end="")

                # Put the final response into the output queue
                self.output_stream_queue.put({"data": bot_response, "event": event})

            except queue.Empty:
                continue
            except Exception:
                pass

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


if __name__ == "__main__":
    # 1. Initialize the core components and both queues
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    VAD_LEVEL = 3
    SHORT_PAUSE_MS = 300
    LONG_PAUSE_MS = 1000
    STREAM_DETECTOR_INPUT_QUEUE = queue.Queue()
    STT_INPUT_QUEUE = queue.Queue()
    TTT_INPUT_QUEUE = queue.Queue()
    TTS_INPUT_QUEUE = queue.Queue()

    audio_stream = LocalAudioStream(output_queue=STREAM_DETECTOR_INPUT_QUEUE)
    # 3. Start capturing audio

    stream_detector = AdvancedSpeechPauseDetectorStream(
        input_queue=STREAM_DETECTOR_INPUT_QUEUE,
        output_queue=STT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        vad_level=VAD_LEVEL,
        short_pause_ms=SHORT_PAUSE_MS,
        long_pause_ms=LONG_PAUSE_MS,
    )

    STT_MODEL = (
        LocalSpeechToTextModel(STT_MODEL_LOCAL, device=device)
        if STT_MODE == "local"
        else RemoteSpeechToTextModel(STT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    TTT_MODEL = (
        LocalTextToTextModel(TTT_MODEL_LOCAL, device=device)
        if TTT_MODE == "local"
        else RemoteTextToTextModel(TTT_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    # 2. Initialize the STT Processor
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=STT_INPUT_QUEUE,
        output_stream_queue=TTT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE,
    )

    # 3. Initialize the new TTT Processor
    ttt_processor = TextToTextStreamProcessor(
        ttt_model=TTT_MODEL,
        input_stream_queue=TTT_INPUT_QUEUE,  # Takes input from the user text queue
        output_stream_queue=TTS_INPUT_QUEUE,  # Outputs to the final response queue
    )

    # 5. Start all threaded components
    audio_stream.start()
    stream_detector.start()
    stt_processor.start()
    ttt_processor.start()

    try:
        # The main thread now listens for the final bot response
        while True:
            try:
                data = TTS_INPUT_QUEUE.get(timeout=1.0)
                bot_reply = data["data"]
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        # 6. Stop all components gracefully
        audio_stream.stop()
        stream_detector.stop()
        stt_processor.stop()
        ttt_processor.stop()
