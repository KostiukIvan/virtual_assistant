import os

import requests

import pkg.config as config
from pkg.ai.models.ttt.ttt_interface import TextToTextModel


# ===== Remote HuggingFace TTT (New Class) =====
class RemoteTextToTextModel(TextToTextModel):
    def __init__(
        self,
        model: str = config.TTT_MODEL_REMOTE,
        hf_token: str | None = None,
    ) -> None:
        # We don't need the 'device' parameter for remote models
        super().__init__(model)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            msg = "Hugging Face API token not found. " "Import it from config or set the HF_TOKEN environment variable."
            raise ValueError(
                msg,
            )

        self.api_url = model
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }

    def text_to_text(self, message: str) -> str:
        """Sends text to the Hugging Face Inference API for a response."""
        payload = {
            "inputs": message,
            "parameters": {
                "max_new_tokens": 128,  # Limit the length of the reply
                "return_full_text": False,  # Only get the model's reply
            },
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code != 200:
            return f"Error: API returned status {response.status_code} - {response.text}"

        result = response.json()

        # Handle potential errors from the API
        if "error" in result:
            if "is currently loading" in result["error"]:
                estimated_time = result.get("estimated_time", 0)
                return f"Model is loading, please try again in {estimated_time:.0f} seconds."
            return f"API Error: {result['error']}"

        # Parse the successful response
        if isinstance(result, list) and result and "generated_text" in result[0]:
            return result[0]["generated_text"].strip()
        return "Error: Could not parse the API response."
