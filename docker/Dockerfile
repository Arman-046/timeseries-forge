# Multi-stage build: keep the final image lean by not shipping build tools
# or training-only dependencies (tensorboard, matplotlib, pytest) into prod.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --target=/install ".[serve]"

# ---- runtime stage ----
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/deps \
    FORGE_MODEL_PATH=/app/artifacts/model.pt \
    FORGE_SCALER_PATH=/app/artifacts/scaler.json

COPY --from=builder /install /app/deps
COPY src/timeseries_forge /app/deps/timeseries_forge
COPY artifacts /app/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "timeseries_forge.deployment.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]
