FROM python:3.12-slim

# Create cache directory with proper permissions
RUN mkdir -p /app/cache && chmod -R 777 /app/cache
ENV HF_HOME=/app/cache
ENV OMP_NUM_THREADS=1
# This forces CUDA to throw the error exactly where it happens, so the traceback is correct.
ENV CUDA_LAUNCH_BLOCKING=1 

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
CMD ["bash", "-c", "export OMP_NUM_THREADS=1 && uvicorn pkg.ai.app:app --host 0.0.0.0 --port 7860"]
