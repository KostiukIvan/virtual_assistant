import queue
import threading

from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.processor.aspd_stream_processor import (
    AdvancedSpeechPauseDetectorStream,
)
from pkg.config import HF_API_TOKEN, STT_MODE, STT_MODEL_LOCAL, STT_MODEL_REMOTE, device


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
        sample_rate: int = 16000,
    ) -> None:
        """Initializes the SpeechToTextStreamProcessor.

        Args:
            stt_model (object): An object with an `audio_to_text(buffer, sample_rate)` method.
            input_stream_queue (queue.Queue): The queue to get audio chunks from.
            output_stream_queue (queue.Queue): The queue to put transcribed text into.
            sample_rate (int): The sample rate of the audio.

        """
        self.stt_model = stt_model
        self.input_stream_queue = input_stream_queue
        self.output_stream_queue = output_stream_queue
        self.sample_rate = sample_rate

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
                    sample_rate=self.sample_rate,
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

    audio_stream = LocalAudioStream(output_queue=STREAM_DETECTOR_INPUT_QUEUE)

    # 3. Start capturing audio
    audio_stream.start()

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

    # 2. Initialize the updated SpeechToTextStreamProcessor
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=STT_INPUT_QUEUE,
        output_stream_queue=TTT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE,
    )

    # 4. Start both components
    stt_processor.start()
    stream_detector.start()

    try:
        # The main thread now listens for results from the TTT_INPUT_QUEUE
        while True:
            try:
                data = TTT_INPUT_QUEUE.get(timeout=1.0)
                transcribed_text = data["data"]
                event = data["event"]
                print(transcribed_text, end="")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        # 5. Stop the components gracefully
        stream_detector.stop()
        stt_processor.stop()
        audio_stream.stop()
