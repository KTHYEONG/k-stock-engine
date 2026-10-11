# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Seoul \
    UV_NO_SYNC=1 \
    OMP_NUM_THREADS=2 \
    POLARS_MAX_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 rclone \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked uv sync --frozen --no-dev

ARG KSE_CODE_COMMIT=UNKNOWN
ENV KSE_CODE_COMMIT=${KSE_CODE_COMMIT}
