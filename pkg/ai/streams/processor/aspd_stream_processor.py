import numpy as np
import sounddevice as sd
import webrtcvad
import queue
import collections
import threading
import time
from pkg.ai.models.aspd_detector import AdvancedSpeechPauseDetector
from pkg.ai.streams.input.local.audio_input_stream import LocalAudioStream


class AdvancedSpeechPauseDetectorStream:
    """
    Consumes audio frames from an input queue, detects speech pauses,
    and puts event messages into an output queue.
    """
    def __init__(self,
                 input_queue: queue.Queue,
                 output_queue: queue.Queue,
                 long_pause_callback: callable,
                 short_pause_callback: callable,
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30,
                 vad_level: int = 3,
                 short_pause_ms: int = 250,
                 long_pause_ms: int = 600):
        """
        Initializes the stream processor.

        Args:
            input_queue (queue.Queue): Queue to get audio frames from.
            output_queue (queue.Queue): Queue to put pause event strings into.
            All other args are passed to the AdvancedSpeechPauseDetector.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        self.long_pause_callback = long_pause_callback
        self.short_pause_callback = short_pause_callback
        
        self.detector = AdvancedSpeechPauseDetector(
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
            vad_level=vad_level,
            short_pause_ms=short_pause_ms,
            long_pause_ms=long_pause_ms,
        )

        self.is_running = False
        self.thread = None
        
    def _processing_loop(self):
        """The main loop for processing audio from the input queue."""
        while self.is_running:
            try:
                # Get a chunk of audio from the input queue
                audio_chunk = self.input_queue.get(timeout=1.0)
                
                # Process the chunk to detect pauses
                status = self.detector.process_chunk(audio_chunk)
                
                if status == "SHORT_PAUSE": 
                    self.short_pause_callback()
                    
                if status == "LONG_PAUSE":
                    self.long_pause_callback()

                self.output_queue.put(audio_chunk)

            except queue.Empty:
                # If the input queue is empty, just continue waiting
                continue
            except Exception as e:
                print(f"An error occurred in the processing loop: {e}")
                break
    
    def start(self):
        """Starts the processing in a separate thread."""
        if self.is_running:
            print("Stream processor is already running.")
            return

        print("Starting speech pause detector stream processor...")
        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the processing thread."""
        print("Stopping speech pause detector stream processor...")
        self.is_running = False
        if self.thread:
            self.thread.join() # Wait for the thread to finish
        print("Stream processor stopped.")



def main():
    """
    Example of using LocalAudioStream to feed an AdvancedSpeechPauseDetectorStream.
    """
    # 1. Create the queues to connect the components
    mic_output_queue = queue.Queue()
    detector_output_queue = queue.Queue()

    # 2. Initialize the local audio stream to capture mic input
    audio_stream = LocalAudioStream(output_queue=mic_output_queue)

    # 3. Initialize the pause detector stream
    # It will consume from the mic's output queue and produce to its own output queue.
    stream_detector = AdvancedSpeechPauseDetectorStream(
        input_queue=mic_output_queue,
        output_queue=detector_output_queue,
        long_pause_callback=lambda: print("\n--- LONG PAUSE DETECTED ---"),
        short_pause_callback=lambda: print("\n--- SHORT PAUSE DETECTED ---"),
        sample_rate=16000,
        frame_duration_ms=30,
        vad_level=3,
        short_pause_ms=300,
        long_pause_ms=1500
    )
    
    print("🎤 Microphone is active. Speak or pause to trigger callbacks.")
    print("   Press Ctrl+C to stop the application.")
    
    # 4. Start both processing threads
    audio_stream.start()
    stream_detector.start()
    
    try:
        # 5. The main thread can now consume the final processed audio from the detector
        while True:
            try:
                # Get the processed frame from the final queue
                processed_frame = detector_output_queue.get(timeout=1.0)
                # For this demo, we just print a dot to show that frames are flowing through.
                print(".", end="", flush=True)
                
            except queue.Empty:
                # This is normal if there's a brief gap in processing
                continue

    except KeyboardInterrupt:
        print("\nInterruption detected. Stopping application...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 6. Stop the streams gracefully
        audio_stream.stop()
        stream_detector.stop()
        print("Application finished.")


if __name__ == "__main__":
    # To run this, you need sounddevice, numpy, and webrtcvad:
    # pip install sounddevice numpy webrtcvad-wheels
    main()