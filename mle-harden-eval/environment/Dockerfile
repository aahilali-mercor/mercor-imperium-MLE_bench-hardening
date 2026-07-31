# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.6 AS uv

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    PATH=/opt/venv/bin:/opt/uv-python/cpython-3.11.15-linux-x86_64-gnu/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      ffmpeg \
      fonts-dejavu-core \
      ghostscript \
      libglib2.0-0 \
      libgl1 \
      libgomp1 \
      libmagic1 \
      libsm6 \
      libtesseract-dev \
      libvips42 \
      libxext6 \
      libxrender1 \
      openslide-tools \
      tesseract-ocr \
      tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/imperium-environment
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv UV_NO_CACHE=0 uv python install 3.11.15 \
    && uv sync --frozen --no-dev --no-install-project

COPY smoke_runtime.py /opt/imperium/smoke_runtime.py
COPY weights-manifest.json /opt/imperium/weights-manifest.json

WORKDIR /workspace
ENTRYPOINT []
CMD ["python", "-c", "import torch; print(torch.__version__)" ]
