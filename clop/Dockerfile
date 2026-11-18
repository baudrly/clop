# syntax=docker/dockerfile:1.7
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/baudrly/clop"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# CPU wheels keep the image small; for GPUs, build FROM a CUDA PyTorch image instead
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1"
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install ".[all]" && useradd -m -u 10001 clop
USER clop
WORKDIR /work
ENTRYPOINT ["clop"]
CMD ["--help"]
