from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

ML_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".sav"}
DL_EXTENSIONS = {".pt", ".pth", ".onnx", ".h5", ".keras"}
SUPPORTED_EXTENSIONS = ML_EXTENSIONS | DL_EXTENSIONS


def inspect_model_file(model_path: Path) -> dict[str, Any]:
    """
    Load estimator with joblib (same as typical sklearn workflows) and infer framework.
    Returns: framework, estimator_class, has_predict_proba, n_features_in (if known).
    """
    path = model_path.resolve()
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported model extension: {path.suffix}")
    if not path.is_file():
        raise FileNotFoundError(str(path))

    # Extension-first detection for DL artifacts to avoid heavyweight host imports.
    if suffix in {".pt", ".pth"}:
        return {
            "framework": "pytorch",
            "estimator_class": "torch.serialized",
            "has_predict_proba": False,
            "n_features_in": None,
            "task_type": "dl",
            "detail": "Detected by extension (.pt/.pth)",
        }
    if suffix == ".onnx":
        return {
            "framework": "onnx",
            "estimator_class": "onnx.model",
            "has_predict_proba": False,
            "n_features_in": None,
            "task_type": "dl",
            "detail": "Detected by extension (.onnx)",
        }
    if suffix in {".h5", ".keras"}:
        return {
            "framework": "tensorflow",
            "estimator_class": "tf.keras.model",
            "has_predict_proba": False,
            "n_features_in": None,
            "task_type": "dl",
            "detail": "Detected by extension (.h5/.keras)",
        }

    try:
        obj = joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Could not load model with joblib: {e}") from e

    framework, detail = _classify_object(obj)
    has_proba = hasattr(obj, "predict_proba")
    n_features = getattr(obj, "n_features_in_", None)

    return {
        "framework": framework,
        "estimator_class": f"{type(obj).__module__}.{type(obj).__name__}",
        "has_predict_proba": bool(has_proba),
        "n_features_in": int(n_features) if n_features is not None else None,
        "task_type": "ml",
        "detail": detail,
    }


def _classify_object(obj: Any) -> tuple[str, str]:
    mod = getattr(type(obj), "__module__", "") or ""
    name = type(obj).__name__

    if mod.startswith("sklearn.") or "sklearn" in mod:
        return "sklearn", f"sklearn object: {name}"

    if mod.startswith("xgboost.") or "xgboost" in mod:
        return "xgboost", f"xgboost object: {name}"

    if mod.startswith("lightgbm.") or "lightgbm" in mod:
        return "lightgbm", f"lightgbm object: {name}"

    if mod.startswith("catboost.") or "catboost" in mod:
        return "catboost", f"catboost object: {name}"

    # Pipeline / ColumnTransformer wrap sklearn steps
    if hasattr(obj, "steps"):
        return "sklearn", f"pipeline-like ({name}); steps detected"

    return "unknown", f"module={mod!r} name={name}"


def detect_with_llm(model_path: Path, inspection_error: str | None) -> dict[str, Any]:
    """Use LLM when load fails or framework is unknown. Returns same keys as inspect_model_file where possible."""
    from deployment_agent.llm_client import complete_json

    system = (
        "You classify a machine learning artifact for deployment. "
        "Reply with JSON only: "
        '{"framework":"sklearn"|"xgboost"|"lightgbm"|"catboost"|"pytorch"|"tensorflow"|"onnx"|"unknown",'
        '"estimator_class":"string or null","has_predict_proba":bool,"n_features_in":int|null,'
        '"task_type":"ml"|"dl"|"unknown","detail":"short reason"}'
    )
    user = json.dumps(
        {
            "filename": model_path.name,
            "suffix": model_path.suffix,
            "inspection_error": inspection_error,
        }
    )
    data = complete_json(system, user)
    fw = data.get("framework", "unknown")
    if fw not in {"sklearn", "xgboost", "lightgbm", "catboost", "pytorch", "tensorflow", "onnx", "unknown"}:
        fw = "unknown"
    return {
        "framework": fw,
        "estimator_class": data.get("estimator_class"),
        "has_predict_proba": bool(data.get("has_predict_proba", False)),
        "n_features_in": data.get("n_features_in"),
        "task_type": data.get("task_type") if data.get("task_type") in {"ml", "dl"} else ("dl" if fw in {"pytorch", "tensorflow", "onnx"} else "ml"),
        "detail": data.get("detail", "llm"),
    }
