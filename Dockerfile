# syntax=docker/dockerfile:1.7-labs

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
LABEL org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY}"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade --no-cache-dir pip \
 && python -m pip install --no-cache-dir \
      numpy \
      pandas \
      scikit-learn \
      matplotlib \
      seaborn \
      fpdf2 \
      onnx \
      onnxruntime \
      polars

WORKDIR /app

# so we can run `-m clop.clop`
ENV PYTHONPATH=/app/src

COPY src/ /app/src/

RUN useradd -m -u 10001 appuser \
 && chown -R appuser:appuser /app
USER appuser

# docker run ... -- <args>
ENTRYPOINT ["python", "-m", "clop.clop"]
CMD ["--help"]
