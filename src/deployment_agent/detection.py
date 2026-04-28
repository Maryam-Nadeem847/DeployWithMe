from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

ML_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".sav"}
DL_EXTENSIONS = {".pt", ".pth", ".onnx", ".h5", ".keras"}
SUPPORTED_EXTENSIONS = ML_EXTENSIONS | DL_EXTENSIONS

DEFAULT_INPUT_SPEC = {
    "height": 224,
    "width": 224,
    "channels": 3,
    "dtype": "float32",
    "channel_order": "NHWC",
    "auto_detected": False,
    "dynamic": True,
    "warning": "Auto-detect couldn't determine these — please confirm",
}


def _coerce_dim(value: Any) -> int | None:
    """Return an int >= 1 or None if value is None/symbolic/non-positive."""
    try:
        if value is None:
            return None
        v = int(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _spec_from_shape(shape: tuple, channel_order: str) -> dict[str, Any]:
    """
    Build an input spec from a shape tuple.

    `shape` is expected to include batch (e.g. (None, H, W, C) for NHWC,
    (None, C, H, W) for NCHW). Dimensions that are None / symbolic become
    None and trigger a dynamic-fallback spec.
    """
    if channel_order == "NHWC":
        # (N, H, W, C)
        if len(shape) != 4:
            return _dynamic_spec(reason=f"Unexpected NHWC rank: shape={shape!r}")
        _, h, w, c = shape
    else:
        # (N, C, H, W)
        if len(shape) != 4:
            return _dynamic_spec(reason=f"Unexpected NCHW rank: shape={shape!r}")
        _, c, h, w = shape

    height = _coerce_dim(h)
    width = _coerce_dim(w)
    channels = _coerce_dim(c)

    if height is None or width is None or channels is None:
        logger.warning(
            "Dynamic input shape detected (h=%s w=%s c=%s); falling back to defaults.",
            h,
            w,
            c,
        )
        spec = dict(DEFAULT_INPUT_SPEC)
        spec["channel_order"] = channel_order
        spec["channels"] = channels or DEFAULT_INPUT_SPEC["channels"]
        spec["height"] = height or DEFAULT_INPUT_SPEC["height"]
        spec["width"] = width or DEFAULT_INPUT_SPEC["width"]
        return spec

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": "float32",
        "channel_order": channel_order,
        "auto_detected": True,
        "dynamic": False,
        "warning": None,
    }


def _dynamic_spec(reason: str) -> dict[str, Any]:
    logger.warning("Input spec extraction fell back to defaults: %s", reason)
    return dict(DEFAULT_INPUT_SPEC)


def _extract_tf_spec(path: Path) -> dict[str, Any] | None:
    try:
        import keras  # type: ignore
    except Exception as e:
        logger.warning("Could not import keras for spec extraction: %s", e)
        return None
    try:
        model = keras.models.load_model(str(path), compile=False, safe_mode=False)
    except Exception as e:
        logger.warning("Could not load Keras model for spec extraction: %s", e)
        return None
    shape = getattr(model, "input_shape", None)
    if not isinstance(shape, tuple):
        return _dynamic_spec(reason=f"keras input_shape not tuple: {shape!r}")
    return _spec_from_shape(shape, channel_order="NHWC")


def _extract_onnx_spec(path: Path) -> dict[str, Any] | None:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        logger.warning("Could not import onnxruntime for spec extraction: %s", e)
        return None
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        shape = tuple(inp.shape)
    except Exception as e:
        logger.warning("Could not read ONNX input shape: %s", e)
        return None
    if len(shape) != 4:
        return _dynamic_spec(reason=f"ONNX rank != 4: shape={shape!r}")
    # NCHW heuristic: dim at index 1 is small (1 or 3 channels).
    second = _coerce_dim(shape[1])
    channel_order = "NCHW" if second in (1, 3) else "NHWC"
    return _spec_from_shape(shape, channel_order=channel_order)


def _extract_pytorch_spec(path: Path) -> dict[str, Any] | None:
    try:
        import torch  # type: ignore
    except Exception as e:
        logger.warning("Could not import torch for spec extraction: %s", e)
        return None
    try:
        model = torch.jit.load(str(path), map_location="cpu")
    except Exception as e:
        logger.warning("torch.jit.load failed for spec extraction: %s", e)
        return None
    try:
        graph = model.graph
        # Skip the implicit `self` input on graph; first real input is at idx 1.
        inputs = list(graph.inputs())
        if len(inputs) < 2:
            return _dynamic_spec(reason="TorchScript graph has no user inputs")
        first = inputs[1]
        ttype = first.type()
        sizes = None
        if hasattr(ttype, "sizes"):
            try:
                sizes = ttype.sizes()
            except Exception:
                sizes = None
        if not sizes or len(sizes) != 4:
            return _dynamic_spec(
                reason=f"TorchScript first input shape unavailable or rank != 4: {sizes!r}"
            )
        return _spec_from_shape(tuple(sizes), channel_order="NCHW")
    except Exception as e:
        return _dynamic_spec(reason=f"TorchScript graph parse failed: {e}")


def extract_input_spec(model_path: Path, framework: str) -> dict[str, Any]:
    """
    Best-effort image-input-spec extraction for DL frameworks.

    Returns a dict with keys: height, width, channels, dtype, channel_order,
    auto_detected, dynamic, warning. On any failure (missing host package,
    load error, dynamic shape), returns sensible defaults plus a yellow
    "please confirm" warning so the UI auto-expands the editor.

    Classical-ML frameworks return None here — the caller already has
    `n_features_in_` and no image spec is needed.
    """
    fw = (framework or "").lower()
    if fw in {"sklearn", "xgboost", "lightgbm", "catboost", "joblib", "unknown"}:
        return None  # type: ignore[return-value]

    spec: dict[str, Any] | None = None
    try:
        if fw in ("tensorflow", "keras"):
            spec = _extract_tf_spec(model_path)
        elif fw == "onnx":
            spec = _extract_onnx_spec(model_path)
        elif fw == "pytorch":
            spec = _extract_pytorch_spec(model_path)
    except Exception as e:
        logger.warning("Unexpected error during spec extraction (%s): %s", fw, e)
        spec = None

    return spec if spec is not None else _dynamic_spec(reason=f"no extractor for fw={fw}")


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

    # Extension-first detection for ML artifacts — avoids version-mismatch crashes.
    ext_framework_map = {
        ".pkl": "sklearn", ".pickle": "sklearn", ".joblib": "sklearn", ".sav": "sklearn",
        ".json": "xgboost", ".ubj": "xgboost",
    }
    ext_framework = ext_framework_map.get(suffix, "unknown")

    # Try to extract sklearn version metadata without requiring a compatible install.
    sklearn_version: str | None = None
    if ext_framework == "sklearn":
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                if hasattr(data, "__sklearn_version__"):
                    sklearn_version = data.__sklearn_version__
                elif hasattr(data, "_sklearn_version"):
                    sklearn_version = data._sklearn_version
        except Exception:
            sklearn_version = None

    # Try full load for richer metadata, but never crash.
    try:
        obj = joblib.load(path)
        framework, detail = _classify_object(obj)
        has_proba = hasattr(obj, "predict_proba")
        n_features = getattr(obj, "n_features_in_", None)

        # Extract sklearn version from loaded object if not found via pickle.
        if sklearn_version is None and framework == "sklearn":
            if hasattr(obj, "__sklearn_version__"):
                sklearn_version = obj.__sklearn_version__
            elif hasattr(obj, "_sklearn_version"):
                sklearn_version = obj._sklearn_version

        return {
            "framework": framework,
            "estimator_class": f"{type(obj).__module__}.{type(obj).__name__}",
            "has_predict_proba": bool(has_proba),
            "n_features_in": int(n_features) if n_features is not None else None,
            "task_type": "ml",
            "detail": detail,
            "sklearn_version": sklearn_version,
        }
    except Exception as e:
        logger.warning("Could not load model with joblib (version mismatch?): %s", e)
        return {
            "framework": ext_framework,
            "estimator_class": None,
            "has_predict_proba": False,
            "n_features_in": None,
            "task_type": "ml",
            "detail": f"Detected by extension ({suffix}); load failed: {e}",
            "sklearn_version": sklearn_version,
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
