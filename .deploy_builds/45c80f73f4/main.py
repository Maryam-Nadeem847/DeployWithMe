from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import keras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/handwriting_model.h5")

load_error = ""
model = None

# ── Common custom layers (Siamese / metric-learning models) ──────────
@keras.saving.register_keras_serializable()
class AbsLayer(keras.layers.Layer):
    def call(self, inputs):
        return keras.ops.abs(inputs)

@keras.saving.register_keras_serializable()
class AbsDiff(keras.layers.Layer):
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return keras.ops.abs(inputs[0] - inputs[1])
        return keras.ops.abs(inputs)

@keras.saving.register_keras_serializable()
class L1Distance(keras.layers.Layer):
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return keras.ops.sum(keras.ops.abs(inputs[0] - inputs[1]), axis=-1, keepdims=True)
        return inputs

@keras.saving.register_keras_serializable()
class L2Distance(keras.layers.Layer):
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return keras.ops.sqrt(keras.ops.sum(keras.ops.square(inputs[0] - inputs[1]), axis=-1, keepdims=True) + 1e-7)
        return inputs

@keras.saving.register_keras_serializable()
class EuclideanDistance(L2Distance):
    pass

@keras.saving.register_keras_serializable()
class ManhattenDistance(L1Distance):
    pass

@keras.saving.register_keras_serializable()
class ManhattanDistance(L1Distance):
    pass

@keras.saving.register_keras_serializable()
class CosineSimilarity(keras.layers.Layer):
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            a, b = inputs
            a = a / (keras.ops.sqrt(keras.ops.sum(keras.ops.square(a), axis=-1, keepdims=True)) + 1e-7)
            b = b / (keras.ops.sqrt(keras.ops.sum(keras.ops.square(b), axis=-1, keepdims=True)) + 1e-7)
            return keras.ops.sum(a * b, axis=-1, keepdims=True)
        return inputs

KNOWN_CUSTOM = {
    "AbsLayer": AbsLayer,
    "AbsDiff": AbsDiff,
    "L1Distance": L1Distance,
    "L2Distance": L2Distance,
    "EuclideanDistance": EuclideanDistance,
    "ManhattenDistance": ManhattenDistance,
    "ManhattanDistance": ManhattanDistance,
    "CosineSimilarity": CosineSimilarity,
}

# ── Robust loader with auto-stub for unknown custom layers ───────────
_MISSING_CLS_RE = re.compile(r"Could not locate class '(\w+)'")

def _make_stub(cls_name):
    """Create a generic pass-through Keras layer for an unknown custom class."""
    stub = type(cls_name, (keras.layers.Layer,), {
        "call": lambda self, inputs, **kw: inputs,
    })
    keras.saving.register_keras_serializable()(stub)
    return stub


def _load_model():
    global model, load_error
    custom_objects = dict(KNOWN_CUSTOM)
    for _ in range(10):
        try:
            model = keras.models.load_model(
                MODEL_PATH,
                compile=False,
                safe_mode=False,
                custom_objects=custom_objects,
            )
            return
        except Exception as e:
            msg = str(e)
            m = _MISSING_CLS_RE.search(msg)
            if m and m.group(1) not in custom_objects:
                cls_name = m.group(1)
                custom_objects[cls_name] = _make_stub(cls_name)
                continue
            load_error = msg
            return
    load_error = "Failed to load model after multiple retries (too many unknown custom classes)"


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
