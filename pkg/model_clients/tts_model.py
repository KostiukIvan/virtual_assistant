import sounddevice as sd
import numpy as np
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch

# ===== Base TTS =====
class TextToSpeechModel:
    def __init__(self, model_name: str, sample_rate: int):
        self.model_name = model_name
        self.sample_rate = sample_rate
    
    def text_to_speech(self, text: str) -> np.ndarray:
        raise NotImplementedError


# ===== Local HuggingFace TTS =====
class LocalTextToSpeechModel(TextToSpeechModel):
    def __init__(self, model_name: str = "microsoft/speecht5_tts", sample_rate: int = 16000):
        """
        model_name examples:
        - "microsoft/speecht5_tts" (High quality, uses a separate vocoder)
        - "espnet/kan-bayashi_ljspeech_vits" (CPU-friendly, good English voice)
        """
        super().__init__(model_name, sample_rate)
        
        # Determine the device for model inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the processor, model, and vocoder
        self.processor = SpeechT5Processor.from_pretrained(model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # A speaker embedding is required for SpeechT5
        self.embeddings_dataset = "Matthijs/cmu-arctic-xvectors"
        self.speaker_embeddings = torch.randn(1, 512) # A placeholder, you should load a real one
        
        try:
            # For demonstration, we'll load a pre-trained speaker embedding
            from datasets import load_dataset
            embeddings = load_dataset(self.embeddings_dataset, split="validation")
            # Pick a speaker, for example, 'cmu_us_slt_arctic'
            self.speaker_embeddings = torch.tensor(embeddings[7306]["xvector"]).unsqueeze(0).to(self.device)
        except ImportError:
            print("To use a real speaker embedding, please install the `datasets` library: `pip install datasets`")
            print("Using a dummy speaker embedding, which may result in lower quality audio.")

    def text_to_speech(self, text: str, sample_rate: int = 16000) -> np.ndarray:
        # Process the text and get the audio output
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        
        # Generate the speech, using the speaker embedding and vocoder
        speech_output = self.model.generate(
            **inputs,
            speaker_embeddings=self.speaker_embeddings,
            vocoder=self.vocoder
        )
        
        # Convert the tensor to a numpy array
        audio = speech_output.cpu().numpy()
        return audio


# ===== TEST MAIN =====
def main():
    # Use the corrected class with the SpeechT5 model
    tts = LocalTextToSpeechModel("microsoft/speecht5_tts")

    while True:
        try:
            user_text = input("\n📝 Enter text to speak (or 'exit'): ")
            if user_text.lower() == "exit":
                break

            print("🤖 Generating speech...")
            audio = tts.text_to_speech(user_text)

            # The SpeechT5 model's native sample rate is 16000 Hz
            sample_rate = 16000

            print("🔊 Playing...")
            sd.play(audio, samplerate=sample_rate)
            sd.wait()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()