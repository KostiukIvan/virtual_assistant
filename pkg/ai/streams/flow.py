import queue
import threading
import argparse
from typing import Dict, Any


# --- Virtual Assistant Class ---

class VirtualAssistant:
    """
    A configurable virtual assistant that processes audio streams using various components.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the virtual assistant with a given configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing the configuration.
        """
        self.config = config
        self.threads = []
        self.queues = {
            "audio_input": queue.Queue(),
            "pause_detection": queue.Queue(),
            "stt_output": queue.Queue(),
            "ttt_output": queue.Queue(),
            "tts_output": queue.Queue(),
        }
        self._create_components()

    def _create_components(self):
        """
        Factory method to create and instantiate components based on the configuration.
        """
        # --- Input Stream ---
        if self.config["input"]["type"] == "local":
            self.input_stream = LocalAudioStream(
                output_queue=self.queues["audio_input"],
                **self.config["input"]["local"]
            )
        else:
            raise ValueError(f"Unsupported input type: {self.config['input']['type']}")

        # --- Output Stream ---
        if self.config["output"]["type"] == "local":
            self.output_stream = LocalAudioProducer(
                input_queue=self.queues["tts_output"],
                speak_callback=lambda: print("Speaking..."),
                **self.config["output"]["local"]
            )
        else:
            raise ValueError(f"Unsupported output type: {self.config['output']['type']}")

        # --- Models ---
        self.stt_model = self._create_model("stt")
        self.ttt_model = self._create_model("ttt")
        self.tts_model = self._create_model("tts")

        # --- Stream Processors ---
        self.aspd_processor = AdvancedSpeechPauseDetectorStream(
            input_queue=self.queues["audio_input"],
            output_queue=self.queues["pause_detection"],
            long_pause_callback=lambda: print("Long pause detected"),
            short_pause_callback=lambda: print("Short pause detected"),
            **self.config["processor"]["aspd"]
        )
        self.stt_processor = SpeechToTextStreamProcessor(
            stt_model=self.stt_model,
            input_stream_queue=self.queues["pause_detection"],
            output_stream_queue=self.queues["stt_output"],
            **self.config["processor"]["stt"]
        )
        self.ttt_processor = TextToTextStreamProcessor(
            ttt_model=self.ttt_model,
            input_stream_queue=self.queues["stt_output"],
            output_stream_queue=self.queues["ttt_output"],
        )
        self.tts_processor = TextToSpeechStreamProcessor(
            tts_model=self.tts_model,
            input_stream_queue=self.queues["ttt_output"],
            output_stream_queue=self.queues["tts_output"],
        )

    def _create_model(self, model_type: str):
        """
        Creates a model instance based on the configuration.

        Args:
            model_type (str): The type of model to create (e.g., "stt", "ttt", "tts").

        Returns:
            An instance of the specified model.
        """
        model_config = self.config["models"][model_type]
        if model_config["type"] == "local":
            if model_type == "stt":
                return LocalSpeechToTextModel(**model_config["local"])
            elif model_type == "ttt":
                return LocalTextToTextModel(**model_config["local"])
            elif model_type == "tts":
                return LocalTextToSpeechModel(**model_config["local"])
        elif model_config["type"] == "remote":
            if model_type == "stt":
                return RemoteSpeechToTextModel(**model_config["remote"])
            elif model_type == "ttt":
                return RemoteTextToTextModel(**model_config["remote"])
            elif model_type == "tts":
                return RemoteTextToSpeechModel(**model_config["remote"])
        raise ValueError(f"Unsupported model type: {model_config['type']} for {model_type}")

    def start(self):
        """
        Starts all the components in separate threads.
        """
        components = [
            self.input_stream, self.output_stream, self.aspd_processor,
            self.stt_processor, self.ttt_processor, self.tts_processor
        ]
        for component in components:
            thread = threading.Thread(target=component.start)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        print("Virtual Assistant started.")

    def stop(self):
        """
        Stops all the components.
        """
        components = [
            self.input_stream, self.output_stream, self.aspd_processor,
            self.stt_processor, self.ttt_processor, self.tts_processor
        ]
        for component in components:
            component.stop()
        print("Virtual Assistant stopped.")


# --- Configuration and Execution ---

def get_default_config():
    """
    Provides a default configuration for the virtual assistant.
    """
    return {
        "input": {
            "type": "local",
            "local": {"sample_rate": 16000, "frame_duration_ms": 30}
        },
        "output": {
            "type": "local",
            "local": {"sample_rate": 16000}
        },
        "models": {
            "stt": {
                "type": "local",
                "local": {"model": "openai/whisper-tiny.en"},
                "remote": {"model": "openai/whisper-large-v2"}
            },
            "ttt": {
                "type": "local",
                "local": {"model": "google/flan-t5-small"},
                "remote": {"model": "google/flan-t5-large"}
            },
            "tts": {
                "type": "local",
                "local": {"model": "microsoft/speecht5_tts"},
                "remote": {"model": "facebook/mms-tts-eng"}
            }
        },
        "processor": {
            "aspd": {"short_pause_ms": 250, "long_pause_ms": 600},
            "stt": {"sample_rate": 16000}
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Virtual Assistant.")
    parser.add_argument("--input", type=str, default="local", choices=["local"], help="Input stream type.")
    parser.add_argument("--output", type=str, default="local", choices=["local"], help="Output stream type.")
    parser.add_argument("--stt", type=str, default="local", choices=["local", "remote"], help="STT model type.")
    parser.add_argument("--ttt", type=str, default="local", choices=["local", "remote"], help="TTT model type.")
    parser.add_argument("--tts", type=str, default="local", choices=["local", "remote"], help="TTS model type.")
    args = parser.parse_args()

    # Create configuration from arguments
    config = get_default_config()
    config["input"]["type"] = args.input
    config["output"]["type"] = args.output
    config["models"]["stt"]["type"] = args.stt
    config["models"]["ttt"]["type"] = args.ttt
    config["models"]["tts"]["type"] = args.tts

    # Initialize and run the assistant
    assistant = VirtualAssistant(config)
    assistant.start()

    try:
        # Keep the main thread alive
        while True:
            pass
    except KeyboardInterrupt:
        assistant.stop()