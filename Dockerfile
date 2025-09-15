FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

# Expose port for HF Space
ENV PORT=7860
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
