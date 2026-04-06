from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/iris_torch_full.pt")

app = FastAPI(title="DL Inference (PyTorch)", version="1.0.0")
load_error = ""
model = None
is_state_dict = False

try:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict):
        is_state_dict = True
    else:
        model = checkpoint.eval()
except Exception as e:
    load_error = str(e)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    if load_error:
        return {"status": "load_failed", "framework": "pytorch", "error": load_error}
    if is_state_dict:
        return {
            "status": "state_dict_only",
            "framework": "pytorch",
            "message": "Checkpoint is a state_dict. Provide architecture or full serialized model.",
        }
    return {"status": "ok", "framework": "pytorch"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {load_error}")
    if is_state_dict:
        raise HTTPException(
            status_code=400,
            detail="Model file is a state_dict without architecture; cannot run inference.",
        )
    try:
        x = torch.tensor([req.features], dtype=torch.float32)
        with torch.no_grad():
            y = model(x)
        arr = y.detach().cpu().numpy()
        return {"prediction": arr.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
