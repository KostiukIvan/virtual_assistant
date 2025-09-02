import queue
import time
from typing import Any

from pkg.ai.models.stt_model import LocalSpeechToTextModel, RemoteSpeechToTextModel
from pkg.ai.models.tts_model import LocalTextToSpeechModel, RemoteTextToSpeechModel
from pkg.ai.models.ttt_model import LocalTextToTextModel, RemoteTextToTextModel
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.output.local.audio_producer import LocalAudioProducer
from pkg.ai.streams.processor.aspd_stream_processor import (
    AdvancedSpeechPauseDetectorStream,
)
from pkg.ai.streams.processor.stt_stream_processor import SpeechToTextStreamProcessor
from pkg.ai.streams.processor.tts_stream_processor import TextToSpeechStreamProcessor
from pkg.ai.streams.processor.ttt_stream_processor import TextToTextStreamProcessor

# Assuming these are your custom module imports
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


class VirtualAssistant:
    """An extensible and configurable Virtual Assistant component that manages the end-to-end
    conversation pipeline from audio input to audio output.

    This class orchestrates various components:
    1.  Input Stream (e.g., local microphone).
    2.  Speech Activity Detection (VAD).
    3.  Speech-to-Text (STT) processing.
    4.  Text-to-Text (LLM) processing.
    5.  Text-to-Speech (TTS) processing.
    6.  Output Stream (e.g., local speakers).
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_stream_class: type,
        output_stream_class: type,
    ) -> None:
        """Initializes the Virtual Assistant and its components based on the provided configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing all necessary configurations
                                     for models, streams, and processors.
            input_stream_class (Type): The class to be used for audio input (e.g., LocalAudioStream).
                                       Must accept an 'output_queue' argument in its constructor.
            output_stream_class (Type): The class to be used for audio output (e.g., LocalAudioProducer).
                                        Must accept an 'input_queue' argument in its constructor.

        """
        self.config = config
        self.input_stream_class = input_stream_class
        self.output_stream_class = output_stream_class

        self.components = {}
        self.queues = self._initialize_queues()
        self.models = self._initialize_models()
        self._initialize_pipeline_components()

    ## Core Methods

    def start(self) -> None:
        """Starts all processing threads in the correct pipeline order."""
        for component in self.components.values():
            if hasattr(component, "start"):
                component.start()

    def stop(self) -> None:
        """Stops all processing threads gracefully in reverse order."""
        # Stop components in reverse order to allow pipelines to clear
        for _name, component in reversed(list(self.components.items())):
            if hasattr(component, "stop"):
                component.stop()

    def run(self) -> None:
        """A convenience method to start the assistant and wait for a shutdown signal."""
        self.start()
        try:
            # Keep the main thread alive while background threads do the work
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # User pressed Ctrl+C
            pass
        finally:
            self.stop()

    ## Initialization Methods

    def _initialize_queues(self) -> dict[str, queue.Queue]:
        """Creates all the necessary queues for data flow between components."""
        return {
            "detector_input": queue.Queue(),
            "stt_input": queue.Queue(),
            "ttt_input": queue.Queue(),
            "tts_input": queue.Queue(),
            "audio_producer_input": queue.Queue(),
        }

    def _initialize_models(self) -> dict[str, Any]:
        """Loads STT, TTT, and TTS models based on the configuration (local or remote)."""
        models = {}

        # Speech-to-Text Model
        stt_mode = self.config.get("stt_mode", "local")
        if stt_mode == "local":
            models["stt"] = LocalSpeechToTextModel(
                self.config["stt_model_local"],
                device=device,
            )
        else:
            models["stt"] = RemoteSpeechToTextModel(
                self.config["stt_model_remote"],
                hf_token=HF_API_TOKEN,
            )

        # Text-to-Text Model
        ttt_mode = self.config.get("ttt_mode", "local")
        if ttt_mode == "local":
            models["ttt"] = LocalTextToTextModel(
                self.config["ttt_model_local"],
                device=device,
            )
        else:
            models["ttt"] = RemoteTextToTextModel(
                self.config["ttt_model_remote"],
                hf_token=HF_API_TOKEN,
            )

        # Text-to-Speech Model
        tts_mode = self.config.get("tts_mode", "local")
        if tts_mode == "local":
            models["tts"] = LocalTextToSpeechModel(
                self.config["tts_model_local"],
                device=device,
            )
        else:
            models["tts"] = RemoteTextToSpeechModel(
                self.config["tts_model_remote"],
                hf_token=HF_API_TOKEN,
            )

        return models

    def _initialize_pipeline_components(self) -> None:
        """Instantiates and connects all pipeline components."""
        # 1. Input Stream (configurable)
        self.components["audio_input"] = self.input_stream_class(
            output_queue=self.queues["detector_input"],
        )

        # 2. Speech Activity Detector
        self.components["speech_detector"] = AdvancedSpeechPauseDetectorStream(
            input_queue=self.queues["detector_input"],
            output_queue=self.queues["stt_input"],
            long_pause_callback=lambda: (print("L"), self.components["tts_processor"].speak()),
            short_pause_callback=self.config.get("short_pause_callback", lambda: None),
            sample_rate=self.config.get("sample_rate", 16000),
            frame_duration_ms=self.config.get("frame_duration_ms", 30),
            vad_level=self.config.get("vad_level", 3),
            short_pause_ms=self.config.get("short_pause_ms", 300),
            long_pause_ms=self.config.get("long_pause_ms", 1000),
        )

        # 3. STT Processor
        self.components["stt_processor"] = SpeechToTextStreamProcessor(
            stt_model=self.models["stt"],
            input_stream_queue=self.queues["stt_input"],
            output_stream_queue=self.queues["ttt_input"],
            sample_rate=self.config.get("sample_rate", 16000),
        )

        # 4. TTT Processor
        self.components["ttt_processor"] = TextToTextStreamProcessor(
            ttt_model=self.models["ttt"],
            input_stream_queue=self.queues["ttt_input"],
            output_stream_queue=self.queues["tts_input"],
        )

        # 5. TTS Processor
        self.components["tts_processor"] = TextToSpeechStreamProcessor(
            tts_model=self.models["tts"],
            input_stream_queue=self.queues["tts_input"],
            output_stream_queue=self.queues["audio_producer_input"],
        )

        # 6. Output Stream (configurable)
        self.components["audio_output"] = self.output_stream_class(
            input_queue=self.queues["audio_producer_input"],
            speak_callback=self.config.get("speak_callback", lambda is_speaking: None),
        )


if __name__ == "__main__":
    # --- Configuration ---
    # Centralize all settings here for easy modification.
    ASSISTANT_CONFIG = {
        # Stream Settings
        "sample_rate": 16000,
        "frame_duration_ms": 30,
        # VAD (Voice Activity Detection) Settings
        "vad_level": 3,
        "short_pause_ms": 300,
        "long_pause_ms": 1000,
        # Model Selection ('local' or 'remote')
        "stt_mode": STT_MODE,
        "ttt_mode": TTT_MODE,
        "tts_mode": TTS_MODE,
        # Model Identifiers
        "stt_model_local": STT_MODEL_LOCAL,
        "stt_model_remote": STT_MODEL_REMOTE,
        "ttt_model_local": TTT_MODEL_LOCAL,
        "ttt_model_remote": TTT_MODEL_REMOTE,
        "tts_model_local": TTS_MODEL_LOCAL,
        "tts_model_remote": TTS_MODEL_REMOTE,
        # Callbacks for advanced control (optional)
        "long_pause_callback": lambda: print("L", end=""),
        "short_pause_callback": lambda: print("S", end=""),
        "speak_callback": lambda is_speaking: print(
            f"[Playback Status: {'SPEAKING' if is_speaking else 'IDLE'}]",
        ),
    }

    # --- Initialization & Execution ---
    # Instantiate the assistant with the config and desired I/O classes.
    # To use remote streams, you would simply pass different classes here,
    # e.g., RemoteAudioStream, RemoteAudioProducer.
    assistant = VirtualAssistant(
        config=ASSISTANT_CONFIG,
        input_stream_class=LocalAudioStream,
        output_stream_class=LocalAudioProducer,
    )

    # The run() method handles starting, waiting for KeyboardInterrupt, and stopping.
    assistant.run()
