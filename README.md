---
title: Virtual Voice Assistant
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: "0.0.1"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Voice Assistant Space

A FastAPI WebSocket server that supports:
- Speech-to-Text (STT)
- Text-to-Text (LLM response)
- Text-to-Speech (TTS)

### Run locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
