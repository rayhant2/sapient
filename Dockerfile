FROM ghcr.io/astral-sh/uv:0.10.4-python3.12-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY . .
RUN uv sync --locked --no-dev && \
    useradd --create-home --uid 10001 sapient && \
    chown -R sapient:sapient /app

USER sapient

EXPOSE 8080 8501

CMD ["uv", "run", "python", "main.py"]
