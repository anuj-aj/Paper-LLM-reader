# -------- Base: resolve & install deps with uv --------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml ./
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

COPY src /app/src

# -------- Final: slim runtime --------
FROM python:3.12-slim AS final

WORKDIR /app
EXPOSE 8000
ENV PYTHONUNBUFFERED=1

COPY --from=base /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Adjust module path if your entry is different
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
