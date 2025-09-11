import queue
import sys
import threading
import time

import sounddevice as sd

from pkg.ai.models.aec.mic_disabler_during_speech import AECWorker
from pkg.ai.models.stt.stt_local import LocalSpeechToTextModel
from pkg.ai.models.stt.stt_remote import RemoteSpeechToTextModel
from pkg.ai.models.tts.main import LocalTextToSpeechModel, RemoteTextToSpeechModel
from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel
from pkg.ai.models.ttt.ttt_remote import RemoteTextToTextModel
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.output.local.audio_producer import LocalAudioProducer
from pkg.ai.streams.processor.aspd_stream_processor import (
    AdvancedSpeechPauseDetectorStream,
)
from pkg.ai.streams.processor.stt_stream_processor import SpeechToTextStreamProcessor
from pkg.ai.streams.processor.ttt_stream_processor import TextToTextStreamProcessor
from pkg.config import (
    HF_API_TOKEN,
    STT_MODE,
    STT_MODEL_LOCAL,
    STT_MODEL_REMOTE,
    TTS_MODE,
    TTS_MODEL_LOCAL,
    TTS_MODEL_REMOTE,
    TTT_MODE,
    TTT_MODEL_LOCAL,
    TTT_MODEL_REMOTE,
    device,
)


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


if __name__ == "__main__":
    # ==== SETTINGS ====
    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30
    FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
    VAD_LEVEL = 3
    SHORT_PAUSE_MS = 300
    LONG_PAUSE_MS = 1000

    # ==== QUEUES ====
    mic_raw_queue = queue.Queue(maxsize=200)  # raw mic frames
    playback_ref_queue = queue.Queue(maxsize=200)  # audio that went to speaker
    mic_clean_queue = queue.Queue(maxsize=200)  # mic after AEC
    playback_in_queue = queue.Queue(maxsize=200)  # audio to play (TTS responses)

    STT_INPUT_QUEUE = queue.Queue()
    TTT_INPUT_QUEUE = queue.Queue()
    TTS_INPUT_QUEUE = queue.Queue()

    # Speech processing models
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
    TTS_MODEL = (
        LocalTextToSpeechModel(TTS_MODEL_LOCAL, device=device)
        if TTS_MODE == "local"
        else RemoteTextToSpeechModel(TTS_MODEL_REMOTE, hf_token=HF_API_TOKEN)
    )

    # ==== COMPONENTS ====

    # Audio in/out
    audio_stream = LocalAudioStream(
        output_queue=mic_raw_queue,
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
    )

    audio_producer = LocalAudioProducer(
        input_queue=playback_in_queue,
        playback_ref_queue=playback_ref_queue,
        sample_rate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    # Acoustic Echo Canceller
    aec = AECWorker(
        mic_queue=mic_raw_queue,
        playback_ref_queue=playback_ref_queue,
        output_queue=mic_clean_queue,
        frame_size=FRAME_SAMPLES,
        # filter_length=FRAME_SAMPLES*20,
        sample_rate=SAMPLE_RATE,
        # max_buffer_ms=500,
    )

    # Pause detector (takes CLEAN mic frames after AEC)
    stream_detector = AdvancedSpeechPauseDetectorStream(
        input_queue=mic_clean_queue,
        output_queue=STT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE,
        frame_duration_ms=FRAME_DURATION_MS,
        vad_level=VAD_LEVEL,
        short_pause_ms=SHORT_PAUSE_MS,
        long_pause_ms=LONG_PAUSE_MS,
    )

    # STT → TTT → TTS processors
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=STT_INPUT_QUEUE,
        output_stream_queue=TTT_INPUT_QUEUE,
        sample_rate=SAMPLE_RATE,
    )
    ttt_processor = TextToTextStreamProcessor(
        ttt_model=TTT_MODEL,
        input_stream_queue=TTT_INPUT_QUEUE,
        output_stream_queue=TTS_INPUT_QUEUE,
    )
    tts_processor = TextToSpeechStreamProcessor(
        tts_model=TTS_MODEL,
        input_stream_queue=TTS_INPUT_QUEUE,
        output_stream_queue=playback_in_queue,  # goes directly to speaker
    )

    # ==== START THREADS ====
    audio_stream.start()
    audio_producer.start()
    aec.start()
    stream_detector.start()
    stt_processor.start()
    ttt_processor.start()
    tts_processor.start()

    try:
        print("Assistant running. Speak into the mic...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Stop components gracefully
        audio_stream.stop()
        audio_producer.stop()
        aec.stop()
        stream_detector.stop()
        stt_processor.stop()
        ttt_processor.stop()
        tts_processor.stop()
