FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY uv.lock pyproject.toml README.md ./
RUN uv sync --locked --no-cache --no-install-project --dev

COPY src/ src/
COPY tests/ tests/
COPY configs/ configs/

ENTRYPOINT ["uv", "run", "pytest", "tests/"]
