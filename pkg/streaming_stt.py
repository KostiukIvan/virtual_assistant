import asyncio
from io import BytesIO
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from transformers import pipeline
import webrtcvad
import sounddevice as sd

# Assuming 'config' is a file with STT_MODEL_LOCAL defined.
# For this example, let's mock it:
class Config:
    STT_MODEL_LOCAL = "openai/whisper-tiny.en"
config = Config()

class StreamingSTT:
    """
    Streaming STT with low-latency partials, overlapping frames,
    and early emission for conversational AI.
    """

    def __init__(
        self,
        model_name=config.STT_MODEL_LOCAL,  # Name or path of the ASR model to load
        vad_level=2,                        # Sensitivity of Voice Activity Detection (0=low, 3=high)
        frame_ms=20,                        # Duration of each audio frame in milliseconds
        overlap_ms=10,                      # Overlap between consecutive frames in milliseconds
        partial_ms=400,                     # Emit partial transcripts every ~partial_ms of speech
        sample_rate=16000                   # Audio sample rate expected by VAD and ASR (16kHz mono)
    ):
        print(f"Loading STT model: {model_name}")

        # ASR pipeline from HuggingFace Transformers
        # The model is loaded here, which can take some time
        self.asr = pipeline("automatic-speech-recognition", model=model_name)

        # WebRTC VAD instance for detecting speech vs silence
        self.vad = webrtcvad.Vad(vad_level)

        # Audio sample rate expected by VAD and ASR (16kHz mono)
        self.sample_rate = sample_rate

        # Duration of each frame in milliseconds
        self.frame_ms = frame_ms

        # Overlap between consecutive frames in milliseconds
        self.overlap_ms = overlap_ms

        # Number of bytes per frame (16-bit PCM mono)
        self.frame_bytes = int(self.sample_rate * self.frame_ms / 1000 * 2)

        # Number of frames after which we emit a partial transcript
        # partial_ms / frame_ms gives approximate number of frames per partial
        self.partial_frames_threshold = int(partial_ms / frame_ms)

        # Initialize internal buffers for speech accumulation and partials
        self._reset_buffers()

        # Async queue to emit partial transcripts for real-time consumption
        self.partial_queue = asyncio.Queue()


    def _reset_buffers(self):
        self.speech_buf = bytearray()
        self.frame_buffer = []
        self.partials = []

    def _decode_to_pcm16(self, media_bytes: bytes) -> bytes:
        audio = AudioSegment.from_file(BytesIO(media_bytes))
        audio = audio.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
        return audio.raw_data

    def _frames(self, pcm: bytes):
        step = self.frame_bytes - int(self.sample_rate * self.overlap_ms / 1000 * 2)
        for i in range(0, len(pcm), step):
            frame = pcm[i:i+self.frame_bytes]
            if len(frame) < self.frame_bytes:
                frame += b"\x00" * (self.frame_bytes - len(frame))
            yield frame

    async def accept_chunk(self, media_bytes: bytes):
        """
        Process a chunk immediately and generate partials asynchronously.
        """
        try:
            # The bytes from sounddevice are already 16-bit PCM, no decoding needed.
            pcm = media_bytes
        except Exception as e:
            print("Decode error:", e)
            return
        
        for frame in self._frames(pcm):
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            self.frame_buffer.append((frame, is_speech))
            
            if is_speech:
                self.speech_buf.extend(frame)
            # Emit partials every N frames or after silence
            if len(self.frame_buffer) >= self.partial_frames_threshold or not is_speech:
                await self._flush_speech_buffer()

    def accept_chunk_sync(self, audio_bytes: bytes, loop: asyncio.AbstractEventLoop):
        """
        Synchronous wrapper to be called from a different thread.
        Schedules the async accept_chunk on the main event loop.
        """
        asyncio.run_coroutine_threadsafe(self.accept_chunk(audio_bytes), loop)

    async def _flush_speech_buffer(self):
        if len(self.speech_buf) == 0:
            self.frame_buffer = []
            return

        # Use a thread-safe way to get the data
        speech_data = self.speech_buf[:]
        self.speech_buf = bytearray()
        self.frame_buffer = []

        try:
            # Move the blocking ASR call to a separate thread
            text = await asyncio.to_thread(self._run_asr, speech_data)

            if text:
                print(f"Partial transcript: {text}")
                self.partials.append(text)
                await self.partial_queue.put(text)
        except Exception as e:
            print("ASR error:", e)

    def _run_asr(self, speech_data: bytes) -> str:
        """Helper function to run the blocking ASR model call."""
        wav_buf = BytesIO()
        sf.write(wav_buf, np.frombuffer(speech_data, dtype=np.int16), self.sample_rate, format="WAV")
        wav_buf.seek(0)
        res = self.asr(wav_buf.read())
        return res.get("text", "").strip()

    async def get_partial(self, timeout=0.1):
        """
        Async getter for partial transcripts. Returns None if timeout.
        """
        try:
            return await asyncio.wait_for(self.partial_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def finalize(self):
        """
        Finalize all buffers and return full text and partials.
        """
        await self._flush_speech_buffer()
        final_text = " ".join(self.partials).strip()
        parts = self.partials[:]
        self._reset_buffers()
        return final_text, parts

# =========================================================================
# === MAIN FUNCTION FOR LIVE TESTING ===
# =========================================================================

async def main():
    """
    Sets up a live microphone stream to test the StreamingSTT class.
    """
    # Initialize the StreamingSTT object
    stt = StreamingSTT(model_name="openai/whisper-tiny.en", vad_level=2, frame_ms=20, overlap_ms=10, partial_ms=5000)
    
    # Create an asyncio event loop
    loop = asyncio.get_event_loop()
    
    # This callback will be triggered by sounddevice for each audio block
    def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            asyncio.run_coroutine_threadsafe(
                stt.accept_chunk(indata.tobytes()), loop
            )

        
    print("🎤 Starting microphone stream. Speak now...")
    
    try:
        with sd.InputStream(
            samplerate=stt.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=1024,
            callback=audio_callback,
        ):
            # This loop can now run freely without blocking
            while True:
                partial_text = await stt.get_partial()
                if partial_text:
                    print(f"Heard: {partial_text}")
                await asyncio.sleep(0.01) # Yield to the event loop

    except KeyboardInterrupt:
        print("\nStopping stream...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        final_text, _ = await stt.finalize()
        print(f"\nFinal Transcript: {final_text}")

if __name__ == "__main__":
    asyncio.run(main())