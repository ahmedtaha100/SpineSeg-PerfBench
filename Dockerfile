# Optional GPU image for real benchmark runs. Docker is not required for CPU smoke tests.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/spineseg-perfbench
COPY . .
RUN python3.11 -m pip install --upgrade pip && python3.11 -m pip install -e .
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /workspace/spineseg-perfbench
USER appuser

CMD ["bash"]
