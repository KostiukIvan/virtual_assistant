import queue
import time

from pkg.ai.models.aec.mic_disabler_during_speech import AECWorker
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream
from pkg.ai.streams.output.local.audio_producer import LocalAudioProducer
from pkg.ai.streams.processor.aspd_stream_processor import (
    AdvancedSpeechPauseDetectorStream,
)

from pkg.ai.models.stt.whisper import LocalSpeechToTextModel
from pkg.ai.models.tts.main import LocalTextToSpeechModel
from pkg.ai.models.ttt.ttt_local import LocalTextToTextModel
from pkg.ai.streams.processor.stt_stream_processor import SpeechToTextStreamProcessor
from pkg.ai.streams.processor.tts_stream_processor import TextToSpeechStreamProcessor
from pkg.ai.streams.processor.ttt_stream_processor import TextToTextStreamProcessor
from pkg.config import (
    STT_MODEL,
    TTS_MODEL,
    TTT_MODEL,
    device,
)

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
    STT_MODEL = LocalSpeechToTextModel(STT_MODEL, device=device)
    TTT_MODEL = LocalTextToTextModel(TTT_MODEL, device=device)
    TTS_MODEL = LocalTextToSpeechModel(TTS_MODEL, device=device)

    # ==== COMPONENTS ====

    # Audio in/out
    audio_stream = LocalAudioStream(
        output_queue=mic_raw_queue,
    )

    audio_producer = LocalAudioProducer(
        input_queue=playback_in_queue,
        playback_ref_queue=playback_ref_queue,
    )

    # Acoustic Echo Canceller
    aec = AECWorker(
        mic_queue=mic_raw_queue,
        playback_ref_queue=playback_ref_queue,
        output_queue=mic_clean_queue,
    )

    # Pause detector (takes CLEAN mic frames after AEC)
    stream_detector = AdvancedSpeechPauseDetectorStream(
        input_queue=mic_clean_queue,
        output_queue=STT_INPUT_QUEUE,
    )

    # STT → TTT → TTS processors
    stt_processor = SpeechToTextStreamProcessor(
        stt_model=STT_MODEL,
        input_stream_queue=STT_INPUT_QUEUE,
        output_stream_queue=TTT_INPUT_QUEUE,
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
