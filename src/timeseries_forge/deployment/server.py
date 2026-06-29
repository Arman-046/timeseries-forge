"""
FastAPI inference server for a deployed ForgeNet TorchScript artifact.

Run with:
    uvicorn timeseries_forge.deployment.server:app --host 0.0.0.0 --port 8000

Expects the following environment variables (see deployment/config.py):
    FORGE_MODEL_PATH   path to the traced .pt TorchScript artifact
    FORGE_SCALER_PATH  path to the saved scaler JSON (mean/std per channel)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from timeseries_forge.data.datasets import ChannelScaler
from timeseries_forge.deployment.export import load_torchscript

logger = logging.getLogger("timeseries_forge.server")

MODEL_PATH = os.environ.get("FORGE_MODEL_PATH", "artifacts/model.pt")
SCALER_PATH = os.environ.get("FORGE_SCALER_PATH", "artifacts/scaler.json")
EXPECTED_SEQ_LEN = int(os.environ.get("FORGE_SEQ_LEN", "168"))
EXPECTED_NUM_FEATURES = int(os.environ.get("FORGE_NUM_FEATURES", "6"))

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading model from %s", MODEL_PATH)
    _state["model"] = load_torchscript(MODEL_PATH)

    import json

    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH) as f:
            _state["scaler"] = ChannelScaler.from_state_dict(json.load(f))
    else:
        logger.warning("no scaler found at %s; inference will use raw, unscaled inputs", SCALER_PATH)
        _state["scaler"] = None

    yield
    _state.clear()


app = FastAPI(
    title="TimeSeries Forge Inference API",
    description="Multi-task forecasting + anomaly detection serving endpoint",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    """A single window of multivariate time series observations.

    `series` must be shaped (seq_len, num_features) as nested lists,
    in raw (unscaled) units -- the server applies the saved scaler
    internally so clients never need to know about normalization.
    """

    series: list[list[float]] = Field(..., description="(seq_len, num_features) raw values")

    @field_validator("series")
    @classmethod
    def check_shape(cls, v):
        if len(v) != EXPECTED_SEQ_LEN:
            raise ValueError(f"expected seq_len={EXPECTED_SEQ_LEN}, got {len(v)}")
        if any(len(row) != EXPECTED_NUM_FEATURES for row in v):
            raise ValueError(f"expected {EXPECTED_NUM_FEATURES} features per timestep")
        return v


class PredictResponse(BaseModel):
    forecast_quantiles: list[list[list[float]]]  # (horizon, targets, quantiles)
    anomaly_scores: list[float]  # (seq_len,)
    inference_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = np.array(req.series, dtype=np.float32)
    scaler: ChannelScaler | None = _state["scaler"]
    scaled = scaler.transform(raw) if scaler else raw

    x = torch.from_numpy(scaled).unsqueeze(0)  # (1, seq_len, num_features)

    start = time.perf_counter()
    with torch.no_grad():
        forecast, reconstruction = _state["model"](x)
    elapsed_ms = (time.perf_counter() - start) * 1000

    anomaly_scores = (reconstruction - x).pow(2).mean(dim=-1).squeeze(0)  # (seq_len,)

    return PredictResponse(
        forecast_quantiles=forecast.squeeze(0).tolist(),
        anomaly_scores=anomaly_scores.tolist(),
        inference_ms=round(elapsed_ms, 3),
    )
