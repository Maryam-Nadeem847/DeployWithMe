"""
Auto-generated inference API for classical ML (sklearn).
Estimator: sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/digits_histgbc.joblib")

app = FastAPI(title="ML Inference", version="1.0.0")
model = joblib.load(MODEL_PATH)
N_FEATURES_IN = 64


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "framework": "sklearn"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    feats = req.features
    if N_FEATURES_IN is not None and len(feats) != N_FEATURES_IN:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES_IN} features, got {len(feats)}",
        )
    X = np.asarray([feats], dtype=np.float32)
    try:
        preds = model.predict(X)
        pred = preds[0] if hasattr(preds, "__len__") and len(preds) == 1 else preds
        out = {"prediction": pred.tolist() if hasattr(pred, "tolist") else pred}
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            out["proba"] = proba.tolist() if hasattr(proba, "tolist") else proba
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
