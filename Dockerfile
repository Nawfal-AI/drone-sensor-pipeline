FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install numpy first and pin it — must happen before torch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.26.4" && \
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY src/       ./src/
COPY scripts/   ./scripts/
COPY tests/     ./tests/

RUN mkdir -p data/raw/visdrone \
             data/processed \
             data/sharded

CMD ["bash", "-c", \
    "python scripts/run_ingestion.py --split val && \
     python scripts/run_alignment.py && \
     python scripts/run_curation.py && \
     python scripts/run_sharding.py --shard-size 200 && \
     python scripts/run_dataloader_benchmark.py --num-workers 2 --num-batches 10"]