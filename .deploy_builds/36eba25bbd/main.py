from __future__ import annotations

import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/iris_tf.h5")

app = FastAPI(title="DL Inference (TensorFlow)", version="1.0.0")
load_error = ""
model = None

try:
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
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
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    try:
        x = np.asarray([req.features], dtype=np.float32)
        y = model(x, training=False)
        return {"prediction": y.numpy().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
