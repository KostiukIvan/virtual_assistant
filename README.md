# Voice Assistant Space

A FastAPI WebSocket server that supports:
- Speech-to-Text (STT)
- Text-to-Text (LLM response)
- Text-to-Speech (TTS)

### Run locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
