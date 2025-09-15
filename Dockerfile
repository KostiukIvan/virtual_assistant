FROM python:3.12-slim

ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache


# Install system deps (runtime + build)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

# Expose port for HF Space
ENV PORT=7860
CMD uvicorn pkg.ai.app:app --host 0.0.0.0 --port $PORT
