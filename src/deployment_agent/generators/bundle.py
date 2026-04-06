from __future__ import annotations

import shutil
import re
import textwrap
from pathlib import Path
from typing import Any

from deployment_agent import config


def _base_requirements(framework: str) -> list[str]:
    lines = [
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
    ]
    extra = {
        "sklearn": ["scikit-learn>=1.3.0"],
        "xgboost": ["scikit-learn>=1.3.0", "xgboost>=2.0.0"],
        "lightgbm": ["scikit-learn>=1.3.0", "lightgbm>=4.0.0"],
        "catboost": ["scikit-learn>=1.3.0", "catboost>=1.2.0"],
        "pytorch": ["numpy>=1.24.0", "torch==2.1.0+cpu", "--extra-index-url https://download.pytorch.org/whl/cpu"],
        "onnx": ["onnxruntime>=1.18.0"],
        # Keras 3 is required to load models exported by newer TF/Keras.
        "tensorflow": ["numpy==1.26.4", "tensorflow-cpu==2.21.0", "keras==3.13.2"],
        "unknown": ["scikit-learn>=1.3.0"],
    }
    lines.extend(extra.get(framework, extra["unknown"]))
    return lines


def merge_requirements(
    user_requirements: str | None, framework: str, python_tag: str
) -> tuple[str, list[str]]:
    """
    Merge user-provided requirements with generated pins.
    Returns (full requirements.txt content, decision log lines).
    """
    decisions: list[str] = []
    base = _base_requirements(framework)
    user_lines: list[str] = []
    if user_requirements:
        for line in user_requirements.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            user_lines.append(s)
        decisions.append(f"Merged {len(user_lines)} non-empty lines from user requirements.txt.")
    else:
        decisions.append("No user requirements.txt; using agent defaults for inference stack.")

    def pkg_name(line: str) -> str:
        return line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()

    seen: dict[str, str] = {}
    for line in base + user_lines:
        seen[pkg_name(line)] = line

    merged = list(seen.values())
    merged.sort(key=lambda x: pkg_name(x))
    header = (
        f"# Auto-generated for framework={framework}, target CPython {python_tag}\n"
        "# Pin versions in production as needed.\n"
    )
    return header + "\n".join(merged) + "\n", decisions


def _extract_exact_pin(requirements_content: str, package_names: list[str]) -> str | None:
    names = "|".join(re.escape(n) for n in package_names)
    pat = re.compile(rf"(?im)^\s*(?:{names})\s*==\s*([A-Za-z0-9_.+-]+)\s*$")
    for line in requirements_content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = pat.match(s)
        if m:
            return m.group(1)
    return None


def build_main_py(model_filename: str, meta: dict[str, Any]) -> str:
    framework = meta.get("framework", "sklearn")
    has_proba = meta.get("has_predict_proba", False)
    n_features = meta.get("n_features_in")
    estimator = meta.get("estimator_class") or "unknown"
    nf_repr = repr(n_features)

    is_xgb_booster = framework == "xgboost" and estimator.endswith("Booster")

    proba_lines = ""
    if has_proba and not is_xgb_booster:
        proba_lines = textwrap.dedent(
            """
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                out["proba"] = proba.tolist() if hasattr(proba, "tolist") else proba
            """
        ).strip("\n")

    if is_xgb_booster:
        predict_try_body = textwrap.dedent(
            """
            import xgboost as xgb
            dm = xgb.DMatrix(X)
            pred = model.predict(dm)
            out = {"prediction": pred.tolist() if hasattr(pred, "tolist") else float(pred)}
            return out
            """
        ).strip("\n")
    else:
        core = textwrap.dedent(
            """
            preds = model.predict(X)
            pred = preds[0] if hasattr(preds, "__len__") and len(preds) == 1 else preds
            out = {"prediction": pred.tolist() if hasattr(pred, "tolist") else pred}
            """
        ).strip("\n")
        if proba_lines:
            core += "\n" + proba_lines
        predict_try_body = core + "\nreturn out"

    main = f'''"""
Auto-generated inference API for classical ML ({framework}).
Estimator: {estimator}
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/{model_filename}")

app = FastAPI(title="ML Inference", version="1.0.0")
model = joblib.load(MODEL_PATH)
N_FEATURES_IN = {nf_repr}


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    return {{"status": "ok", "framework": "{framework}"}}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    feats = req.features
    if N_FEATURES_IN is not None and len(feats) != N_FEATURES_IN:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {{N_FEATURES_IN}} features, got {{len(feats)}}",
        )
    X = np.asarray([feats], dtype=np.float32)
    try:
{textwrap.indent(predict_try_body, "        ")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
'''
    return main


def build_main_pytorch(model_filename: str) -> str:
    return f'''from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/{model_filename}")

load_error = ""
model = None
is_state_dict = False


def _load_model():
    global model, load_error, is_state_dict
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            is_state_dict = True
        else:
            model = checkpoint.eval()
    except Exception as e:
        load_error = str(e)


threading.Thread(target=_load_model, daemon=True).start()

app = FastAPI(title="DL Inference (PyTorch)", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    if load_error:
        return {{"status": "load_failed", "framework": "pytorch", "error": load_error}}
    if is_state_dict:
        return {{
            "status": "state_dict_only",
            "framework": "pytorch",
            "message": "Checkpoint is a state_dict. Provide architecture or full serialized model.",
        }}
    if model is None:
        return {{"status": "loading", "framework": "pytorch"}}
    return {{"status": "ok", "framework": "pytorch"}}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {{load_error}}")
    if is_state_dict:
        raise HTTPException(
            status_code=400,
            detail="Model file is a state_dict without architecture; cannot run inference.",
        )
    if model is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    try:
        x = torch.tensor([req.features], dtype=torch.float32)
        with torch.no_grad():
            y = model(x)
        arr = y.detach().cpu().numpy()
        return {{"prediction": arr.tolist()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
'''


def build_main_onnx(model_filename: str) -> str:
    return f'''from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/{model_filename}")

load_error = ""
session = None
input_name = ""


def _load_model():
    global session, input_name, load_error
    try:
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
    except Exception as e:
        load_error = str(e)


threading.Thread(target=_load_model, daemon=True).start()

app = FastAPI(title="DL Inference (ONNX)", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="1D feature vector")


@app.get("/health")
def health() -> dict[str, Any]:
    if load_error:
        return {{"status": "load_failed", "framework": "onnx", "error": load_error}}
    if session is None:
        return {{"status": "loading", "framework": "onnx"}}
    return {{"status": "ok", "framework": "onnx"}}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {{load_error}}")
    if session is None:
        raise HTTPException(status_code=503, detail="Model still loading")
    try:
        x = np.asarray([req.features], dtype=np.float32)
        outputs = session.run(None, {{input_name: x}})
        values = [o.tolist() if hasattr(o, "tolist") else o for o in outputs]
        return {{"prediction": values}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
'''


def build_main_tensorflow(model_filename: str) -> str:
    return f'''from __future__ import annotations

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

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/{model_filename}")

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

KNOWN_CUSTOM = {{
    "AbsLayer": AbsLayer,
    "AbsDiff": AbsDiff,
    "L1Distance": L1Distance,
    "L2Distance": L2Distance,
    "EuclideanDistance": EuclideanDistance,
    "ManhattenDistance": ManhattenDistance,
    "ManhattanDistance": ManhattanDistance,
    "CosineSimilarity": CosineSimilarity,
}}

# ── Robust loader with auto-stub for unknown custom layers ───────────
_MISSING_CLS_RE = re.compile(r"Could not locate class '(\w+)'")

def _make_stub(cls_name):
    """Create a generic pass-through Keras layer for an unknown custom class."""
    stub = type(cls_name, (keras.layers.Layer,), {{
        "call": lambda self, inputs, **kw: inputs,
    }})
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
        return {{"status": "load_failed", "framework": "tensorflow", "error": load_error}}
    if model is None:
        return {{"status": "loading", "framework": "tensorflow"}}
    return {{"status": "ok", "framework": "tensorflow"}}


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
        return {{"prediction": y.tolist() if hasattr(y, "tolist") else y, "latency_ms": round((time.time() - started) * 1000, 2)}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
'''


def build_main_by_framework(model_filename: str, meta: dict[str, Any]) -> str:
    framework = meta.get("framework", "unknown")
    if framework in {"sklearn", "xgboost", "lightgbm", "catboost", "unknown"}:
        return build_main_py(model_filename, meta)
    if framework == "pytorch":
        return build_main_pytorch(model_filename)
    if framework == "onnx":
        return build_main_onnx(model_filename)
    if framework == "tensorflow":
        return build_main_tensorflow(model_filename)
    return build_main_py(model_filename, meta)


def build_dockerfile(
    python_tag: str,
    model_filename: str,
    framework: str,
    requirements_content: str,
) -> str:
    tf_install_block = ""
    if framework == "tensorflow":
        numpy_pin = _extract_exact_pin(requirements_content, ["numpy"]) or "1.26.4"
        tf_pin = _extract_exact_pin(requirements_content, ["tensorflow", "tensorflow-cpu"]) or "2.21.0"
        keras_pin = _extract_exact_pin(requirements_content, ["keras"]) or "3.13.2"
        # numpy install must be isolated before TensorFlow.
        tf_install_block = (
            "ENV TF_CPP_MIN_LOG_LEVEL=3\n"
            "ENV CUDA_VISIBLE_DEVICES=-1\n"
            f'RUN pip install --no-cache-dir "numpy=={numpy_pin}"\n'
            f'RUN pip install --no-cache-dir "tensorflow-cpu=={tf_pin}"\n'
            f'RUN pip install --no-cache-dir "keras=={keras_pin}"\n'
            'RUN python -c "import numpy as np; import keras; print(f\'numpy={np.__version__}\'); print(f\'keras={keras.__version__}\'); print(\'OK\')"\n'
        )
    return f"""FROM python:{python_tag}-slim
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/{model_filename}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
{tf_install_block}RUN python -c "import fastapi, uvicorn; print('deps_ok')"
COPY {model_filename} ./{model_filename}
COPY main.py .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


def write_deployment_bundle(
    build_dir: Path,
    model_src: Path,
    meta: dict[str, Any],
    user_requirements: str | None,
    python_tag: str | None = None,
) -> tuple[Path, list[str]]:
    """
    Copies model, writes main.py, Dockerfile, requirements.txt under build_dir.
    Returns (build_dir, decision_log).
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    decisions: list[str] = []
    tag = python_tag or config.DEFAULT_CONTAINER_PYTHON
    decisions.append(f"Container base image: python:{tag}-slim")

    model_filename = model_src.name
    dest_model = build_dir / model_filename
    shutil.copy2(model_src, dest_model)
    decisions.append(f"Staged model as {dest_model.name}")

    req_text, req_decisions = merge_requirements(user_requirements, meta.get("framework", "sklearn"), tag)
    decisions.extend(req_decisions)
    (build_dir / "requirements.txt").write_text(req_text, encoding="utf-8")

    main_py = build_main_by_framework(model_filename, meta)
    (build_dir / "main.py").write_text(main_py, encoding="utf-8")
    decisions.append("Generated FastAPI app with /health and /predict.")

    dockerfile = build_dockerfile(
        tag,
        model_filename,
        meta.get("framework", "unknown"),
        req_text,
    )
    (build_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    decisions.append("Wrote Dockerfile (uvicorn on port 8000).")

    return build_dir, decisions
