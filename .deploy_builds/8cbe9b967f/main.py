from __future__ import annotations

import os
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/iris_torch.onnx")

app = FastAPI(title="DL Inference (ONNX)", version="1.0.0")
load_error = ""
session = None
input_name = ""

try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
except Exception as e:
    load_error = str(e)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    if load_error:
        return {"status": "load_failed", "framework": "onnx", "error": load_error}
    return {"status": "ok", "framework": "onnx"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    try:
        x = np.asarray([req.features], dtype=np.float32)
        outputs = session.run(None, {input_name: x})
        values = [o.tolist() if hasattr(o, "tolist") else o for o in outputs]
        return {"prediction": values}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
