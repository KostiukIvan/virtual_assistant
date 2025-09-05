import queue
import threading
import time

import numpy as np
import sounddevice as sd

from pkg.ai.call_state_machines import BotOrchestrator, EventBus, UserFSM, BotFSM
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream


class LocalAudioProducer:
    """
    Play audio chunks via sounddevice and feed the exact played frames to the AEC.
    input_queue: expects np.ndarray float32 frames or int16; chunks may be any multiple of frame size.
    """

    def __init__(self,
                 input_queue: queue.Queue,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 dtype: str = "float32",
                 botx: Optional[BotOrchestrator] = None,
                 aec: Optional[AECWrapper] = None,
                 frame_ms: int = 10):
        self.input_queue = input_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.botx = botx
        self.aec = aec
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000) * channels

        self.is_running = False
        self.thread = None
        self.stream = None

    def _play_and_feed_aec(self, int16_chunk: np.ndarray):
        # Write to device
        self.stream.write(int16_chunk)
        # Also feed to AEC render in frame-sized pieces
        n = len(int16_chunk)
        step = self.frame_samples
        for i in range(0, n, step):
            block = int16_chunk[i:i+step]
            if len(block) < step:
                # pad with zeros to frame size (best to send exact frames)
                block = np.pad(block, (0, step - len(block)), mode="constant")
            # Feed render to AEC
            try:
                if self.aec:
                    self.aec.process_render(block)
            except Exception as e:
                # non-fatal: log and continue
                print(f"[LocalAudioProducer] AEC render error: {e}")

    def _production_loop(self):
        try:
            # Use int16 output stream for consistent AEC expectations
            self.stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                blocksize=self.frame_samples,
            )
            self.stream.start()

            while self.is_running:
                try:
                    audio_chunk = self.input_queue.get(timeout=1)
                except queue.Empty:
                    # Could optionally do botx.ensure_listening()
                    continue

                if audio_chunk is None:
                    break

                # Convert to int16 if needed
                if audio_chunk.dtype == np.float32:
                    int16_chunk = float32_to_int16(audio_chunk.flatten())
                elif audio_chunk.dtype == np.int16:
                    int16_chunk = audio_chunk.flatten()
                else:
                    int16_chunk = audio_chunk.astype(np.int16).flatten()

                # Wait for permission to speak (policy)
                while self.botx and not self.botx.allowed_to_speak():
                    time.sleep(0.01)

                # Play and feed AEC
                self._play_and_feed_aec(int16_chunk)

        except Exception as e:
            print(f"[LocalAudioProducer] error: {e}")
        finally:
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._production_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()


def main() -> None:
    """Directly tests the LocalAudioProducer with a single mock audio chunk."""
    audio_queue = queue.Queue()
    
    bus = EventBus()
    user = UserFSM(bus)
    bot = BotFSM(bus)
    botx = BotOrchestrator(bot)
    botx.speak_pipeline()

    audio_producer = LocalAudioProducer(
        input_queue=audio_queue,
        botx=botx
    )

    # 2. Initialize the audio stream component
    audio_stream = LocalAudioStream(output_queue=audio_queue)
    
    # 3. Start capturing audio
    audio_stream.start()
    audio_producer.start()

    try:
        # 4. Main loop to consume frames from the queue
        while True:
            pass

    except KeyboardInterrupt:
        pass
    finally:

        audio_producer.stop()
        audio_stream.stop()



if __name__ == "__main__":
    main()


# _R_LISTENING_...........................................^
# _C_SPEAKING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^.........
# _C_SHORT_PAUSE_.^
# _C_SPEAKING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Tell me at least when was the last time you spoke^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^.........
# _C_SHORT_PAUSE_.............^
# _C_SPEAKING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^. I spoke to her about a week ago. I was so happy to hear from her. with someone who is incredibly interested to you......^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ That's true, but it's hard to find someone who's interested in you. I guess I'll just have to keep trying.^^^^^^^^^^^^^^^^^^^^........^^^^^^^^
# _R_PROCESSING_
# _R_SPEAKING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^.........
# _C_SHORT_PAUSE_.....^
# _C_SPEAKING_
# _R_LISTENING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^.........
# _C_SHORT_PAUSE_.........._R_PROCESSING_
# _R_SPEAKING_..... Maybe it was your friend, maybe it was your teacher or someone else. I spoke to her about a week ago......^
# _C_SPEAKING_
# _R_LISTENING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ I don't think it was my friend. It was a stranger. I was so scared. I was so happy to hear from her.^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ I bet you were.  What did she do to make you so happy?  Was it a surprise?^^^^^^^^^^^^^^.........
# _C_SHORT_PAUSE_..................^
# _C_SPEAKING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ That's true, but it's hard to find someone who's interested in you.^^^^^^^
# _R_PROCESSING_
# _R_SPEAKING_^^^^.......^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ I know, but I'm trying to be optimistic about it. It's just hard when you know you're good at what you do.^^^^^^^^^^^^^^^^^^^^^^^^^^.........
# _C_SHORT_PAUSE_.^
# _C_SPEAKING_
# _R_LISTENING_^^^^^^^^^^^^^^^^^^^^^^^^^
# _R_PROCESSING_
# _R_SPEAKING_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^.........^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ I guess I'll just have to keep trying. I don't think it was my friend. It was a stranger.^^^^^^^^^^^^^^^^^^^^^^^.. I'm sorry to hear that. I hope you can find a way to get over it.......^^^^^^^^^^^^^^^^^^^^^^^^^^.......^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^C^