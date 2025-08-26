import asyncio
from typing import List, Optional
from abc import ABC, abstractmethod

# You'll need to install these libraries:
# pip install transformers accelerate torch
# pip install huggingface-hub # For cloud-based inference
from transformers import pipeline
from huggingface_hub import InferenceClient

# === Strategy Pattern: Abstract Model Caller ===
class _ModelCaller(ABC):
    """Abstract class for calling a conversational model."""
    @abstractmethod
    async def generate_response(self, conversation: List[dict], system_prompt: str) -> str:
        pass

# === Concrete Strategy 1: Local Hugging Face Model ===
class _LocalModelCaller:
    def __init__(self, model_name: str):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto"
        )
        print(f"Loaded local model: {model_name}")

    async def generate_response(self, conversation: List[dict], system_prompt: str) -> str:
        # Combine system prompt with user messages for the pipeline
        messages = [{"role": "system", "content": system_prompt}] + conversation
        
        # Call the pipeline with the messages list
        output = await asyncio.to_thread(self.pipe, messages, return_full_text=False, max_new_tokens=256)
        
        # The output is a list containing the generated text
        response_text = output[0]['generated_text']
        
        # Clean up any remaining role headers from the output
        if response_text.startswith("assistant"):
            return response_text.replace("assistant", "").strip()
        return response_text.strip()

# === Concrete Strategy 2: Cloud-based Hugging Face Inference API ===
class _CloudModelCaller(_ModelCaller):
    """Calls a conversational model via Hugging Face's Inference Endpoints."""
    def __init__(self, model_endpoint: str):
        self.client = InferenceClient(model=model_endpoint)
        print(f"Connected to cloud endpoint: {model_endpoint}")

    async def generate_response(self, conversation: List[dict], system_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}] + conversation
        try:
            # The client uses async internally to make the API call
            response = await self.client.chat_completion(messages=messages, stream=False)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling cloud endpoint: {e}")
            return "Sorry, I am unable to connect to the cloud model right now."

# =========================================================================
# === Core ConversationalAI Component (remains mostly unchanged) ===
# =========================================================================

class ConversationalAI:
    """
    Conversational AI that consumes streaming text input (partials),
    maintains context, and produces AI-generated responses.
    """
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are a helpful assistant.",
        max_context: int = 10,
        use_cloud: bool = False
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_context = max_context
        self.conversation: List[dict] = []
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None

        # Determine which model caller to use based on 'use_cloud'
        if use_cloud:
            self.model_caller = _CloudModelCaller(model_endpoint=model_name)
        else:
            self.model_caller = _LocalModelCaller(model_name=model_name)

    async def enqueue_input(self, text: str):
        await self.input_queue.put(text)

    async def _process_input(self):
        while True:
            user_text = await self.input_queue.get()
            if user_text is None:
                break
            
            # Use a copy of the conversation to avoid race conditions
            conversation_copy = self.conversation[:]
            conversation_copy.append({"role": "user", "content": user_text})

            # Pass the conversation history and system prompt to the model caller
            ai_response = await self.model_caller.generate_response(conversation_copy, self.system_prompt)

            self.conversation.append({"role": "user", "content": user_text})
            self.conversation.append({"role": "assistant", "content": ai_response})
            self.conversation = self.conversation[-self.max_context:]

            await self.output_queue.put(ai_response)

    def _build_prompt(self) -> str:
        # This is no longer needed since we're passing the message list directly
        # to the new caller classes. We can keep it for legacy or remove it.
        pass

    async def get_response(self, timeout: float = 0.1) -> Optional[str]:
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def start(self):
        self.processing_task = asyncio.create_task(self._process_input())

    async def stop(self):
        await self.input_queue.put(None)
        if self.processing_task:
            await self.processing_task

# =========================================================================
# === MAIN FUNCTION FOR TESTING ===
# =========================================================================
async def test_conversation(ai: ConversationalAI):
    """
    Simulates a live conversation by feeding user inputs and printing AI responses.
    """
    print("🤖 Conversation started. Type 'quit' to exit.")

    input_task = asyncio.create_task(process_user_input(ai))
    output_task = asyncio.create_task(monitor_ai_output(ai))
    
    await asyncio.gather(input_task, output_task)

async def process_user_input(ai: ConversationalAI):
    while True:
        user_text = await asyncio.get_event_loop().run_in_executor(
            None, lambda: input("You: ")
        )
        if user_text.lower() == 'quit':
            await ai.stop()
            break
        await ai.enqueue_input(user_text)

async def monitor_ai_output(ai: ConversationalAI):
    while True:
        response = await ai.get_response(timeout=0.1)
        if response is not None:
            print(f"AI: {response}")
        if not ai.processing_task and ai.input_queue.empty():
            break
        await asyncio.sleep(0.01)

async def main():
    """Main function to run the test scenario."""
    
    # --- Local Test Configuration ---
    # To test locally, you need a model with a conversational/chat-specific format.
    # The smallest model is often the best for local testing.
    # We will use "HuggingFaceH4/zephyr-7b-beta" as a good open-source option.
    # It requires ~15GB VRAM. If you have less, consider a smaller model
    # or use quantization with `device_map="auto"` to offload to CPU.
    local_model_name = "HuggingFaceH4/zephyr-7b-beta"
    ai_local = ConversationalAI(
        model_name=local_model_name,
        system_prompt="You are a helpful and friendly assistant.",
        use_cloud=False
    )
    
    # --- Cloud Test Configuration ---
    # The `model_endpoint` can be a Hugging Face Inference Endpoint URL
    # or a model ID if you have an API key configured.
    # For this to work, you MUST have your HF_TOKEN environment variable set.
    # `export HF_TOKEN="your_huggingface_token"`
    # cloud_model_endpoint = "HuggingFaceH4/zephyr-7b-beta"
    # ai_cloud = ConversationalAI(
    #     model_name=cloud_model_endpoint,
    #     system_prompt="You are a helpful and friendly assistant.",
    #     use_cloud=True
    # )

    # Choose which instance to run
    ai_assistant = ai_local # or ai_cloud

    ai_assistant.start()

    try:
        print(f"--- Testing ConversationalAI with {'local' if not ai_assistant.use_cloud else 'cloud'} model ---")
        await test_conversation(ai_assistant)

    except KeyboardInterrupt:
        print("\nTest interrupted.")
    finally:
        await ai_assistant.stop()

if __name__ == "__main__":
    asyncio.run(main())