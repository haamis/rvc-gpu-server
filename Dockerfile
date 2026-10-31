# ifrit-side RVC/TTS server image.
#
# The big runtime assets stay on the host and are mounted by compose.yaml:
#   rvc_infer/, models/, rvc_models/, ttsbot-diarize/, HF/torch caches.
# This image only contains the Python stack from requirements-gpu.txt.
#
# Build:  docker build -t ttsbot-rvc-server:latest ~/rvc-gpu-server
# Run:    docker compose up -d
# GPU:    docker compose -f compose.yaml -f compose.gpu.yaml up -d
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        ffmpeg libsndfile1 libgomp1 libgl1 libglib2.0-0 \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip

# Torch must come from the CUDA 12.6 wheel index; the rest of the freeze is
# installed with --no-deps so pip cannot downgrade torch for chatterbox etc.
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126

COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install --no-deps -r /tmp/requirements-gpu.txt
# ORT 1.30 is built for CUDA 13; the CUDA 12.6 base needs a 12.x build or
# Kokoro silently falls back to CPUExecutionProvider (libcublasLt.so.13).
RUN pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0

# The CUDA base already has UID/GID 1000 (ubuntu); reuse it and give the
# container a writable $HOME for HF/torch/matplotlib caches.
RUN mkdir -p /home/haama/.cache && chown -R 1000:1000 /home/haama

WORKDIR /app
COPY server.py worker.py worker_owner.py kokoro_engine.py chatterbox_engine.py whisperx_engine.py ./
COPY ttsbot/ ./ttsbot/

ENV HOME=/home/haama \
    PYTHONPATH=/app:/ttsbot-diarize \
    RVC_WORKER_ROOT=/rvc_infer \
    RVC_MODEL_ROOT=/home/haama/TTS-bot/rvc_models \
    KOKORO_MODEL_DIR=/models/kokoro \
    HF_HOME=/home/haama/.cache/huggingface \
    TORCH_HOME=/home/haama/.cache/torch \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    RVC_CUDA_GRAPH=0 \
    RVC_GPU_SERVER_PORT=8001

USER 1000:1000
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=5)"

CMD ["python", "server.py"]
