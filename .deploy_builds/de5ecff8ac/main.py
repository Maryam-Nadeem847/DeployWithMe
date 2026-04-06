from __future__ import annotations

import os
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import time

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/siamese_bilstm_final.keras")

app = FastAPI(title="DL Inference (TensorFlow)", version="1.0.0")
load_error = ""
model = None

try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    load_error = str(e)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    if load_error:
        return {"status": "load_failed", "framework": "tensorflow", "error": load_error}
    return {"status": "ok", "framework": "tensorflow"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if load_error:
        raise HTTPException(status_code=503, detail=load_error)
    try:
        started = time.time()
        x = np.asarray([req.features], dtype=np.float32)
        y = model.predict(x, verbose=0)
        return {"prediction": y.tolist() if hasattr(y, "tolist") else y, "latency_ms": round((time.time() - started) * 1000, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
