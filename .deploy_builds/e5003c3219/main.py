from __future__ import annotations

import os
import threading
import time
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/siamese_bilstm_final.keras")

load_error = ""
model = None


def _load_model():
    global model, load_error
    try:
        model = keras.models.load_model(
            MODEL_PATH, compile=False, safe_mode=False,
        )
    except Exception as e:
        load_error = str(e)


threading.Thread(target=_load_model, daemon=True).start()

app = FastAPI(title="DL Inference (TensorFlow)", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    if load_error:
        return {"status": "load_failed", "framework": "tensorflow", "error": load_error}
    if model is None:
        return {"status": "loading", "framework": "tensorflow"}
    return {"status": "ok", "framework": "tensorflow"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if load_error:
        raise HTTPException(status_code=503, detail=load_error)
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    try:
        started = time.time()
        x = np.asarray([req.features], dtype=np.float32)
        y = model.predict(x, verbose=0)
        return {"prediction": y.tolist() if hasattr(y, "tolist") else y, "latency_ms": round((time.time() - started) * 1000, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
