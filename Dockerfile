FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY emokit/ ./emokit/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e ".[dev,docs]"

# Smoke-test the package with synthetic data so reviewers can verify the image.
RUN python -m emokit.run configs/quick_demo.yaml --dry-run

ENV EMOKIT_DATA_ROOT=/data
VOLUME ["/data", "/app/results"]

CMD ["python", "-m", "emokit.run", "configs/quick_demo.yaml", "--dry-run"]
