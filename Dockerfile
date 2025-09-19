FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

# Create cache directory
RUN mkdir -p /app/cache && chmod -R 777 /app/cache
ENV HF_HOME=/app/cache
ENV OMP_NUM_THREADS=1
ENV CUDA_LAUNCH_BLOCKING=1 

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    python3-dev \
    python3-pip \
    python3.12-venv \ 
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3.12 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install pip deps into the virtual environment
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app
WORKDIR /app

# Expose port for HF Space
ENV PORT=7860
CMD ["bash", "-c", "export OMP_NUM_THREADS=1 && uvicorn pkg.ai.app:app --host 0.0.0.0 --port 7860"]