from __future__ import annotations

import io
import json
import logging
import pickle
import pickletools
import zipfile
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

ML_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".sav"}
DL_EXTENSIONS = {".pt", ".pth", ".onnx", ".h5", ".keras"}
XGB_EXTENSIONS = {".json", ".ubj"}
SUPPORTED_EXTENSIONS = ML_EXTENSIONS | DL_EXTENSIONS | XGB_EXTENSIONS

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
ZIP_MAGIC = b"PK\x03\x04"

# Modules whose presence in a pickle's GLOBAL refs tells us the framework.
_FRAMEWORK_KEYWORDS = {
    "ultralytics": "yolo",
    "sklearn": "sklearn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}

# Module prefixes considered "neutral" — their presence alone doesn't identify
# a framework. A pickle that *only* references these is likely a state-dict.
_NEUTRAL_MODULES = {
    "torch",
    "collections",
    "_codecs",
    "numpy",
    "numpy.core.multiarray",
    "builtins",
    "copyreg",
    "__builtin__",
}

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


def _make_result(
    framework: str,
    *,
    variant: str | None = None,
    confidence: str = "high",
    deployable: bool = True,
    reason: str = "",
    estimator_class: str | None = None,
    has_predict_proba: bool = False,
    n_features_in: int | None = None,
    task_type: str = "ml",
    detail: str = "",
    sklearn_version: str | None = None,
) -> dict[str, Any]:
    """Build a detection-result dict carrying both legacy keys (consumed by
    graph nodes / api_server) and the new content-inspection keys."""
    return {
        "framework": framework,
        "estimator_class": estimator_class,
        "has_predict_proba": bool(has_predict_proba),
        "n_features_in": int(n_features_in) if n_features_in is not None else None,
        "task_type": task_type,
        "detail": detail or reason,
        "sklearn_version": sklearn_version,
        "variant": variant,
        "confidence": confidence,
        "deployable": deployable,
        "reason": reason,
    }


# ── pickle / zip content inspection ───────────────────────────────────


def _scan_pickle_modules(pickle_bytes: bytes) -> set[str]:
    """Return module-name top-levels referenced by a pickle stream.

    Combines two methods so we don't depend on any single one:
      1. Walk pickle opcodes via ``pickletools.genops`` to capture both
         legacy ``GLOBAL`` and protocol-4+ ``STACK_GLOBAL`` references.
      2. As a backstop, substring-scan the raw bytes for known framework
         module names (pickle stores module names as ASCII strings).

    The opcode walker never executes pickle code, so it is safe to run on
    untrusted data.
    """
    found: set[str] = set()

    # Method 1: opcode walk.
    try:
        last_strings: list[str] = []
        for op, arg, _pos in pickletools.genops(io.BytesIO(pickle_bytes)):
            name = op.name
            if name == "GLOBAL" and isinstance(arg, str):
                module = arg.split("\n", 1)[0].split(" ", 1)[0]
                if module:
                    found.add(module)
                    found.add(module.split(".", 1)[0])
            elif name == "STACK_GLOBAL":
                if len(last_strings) >= 2:
                    module = last_strings[-2]
                    if module:
                        found.add(module)
                        found.add(module.split(".", 1)[0])
            elif name in (
                "SHORT_BINUNICODE",
                "BINUNICODE",
                "BINUNICODE8",
                "UNICODE",
                "SHORT_BINSTRING",
                "BINSTRING",
                "STRING",
            ):
                if isinstance(arg, str):
                    last_strings.append(arg)
                    if len(last_strings) > 16:
                        last_strings.pop(0)
    except Exception as e:
        logger.debug("pickletools.genops scan failed: %s", e)

    # Method 2: substring backstop for known frameworks.
    try:
        text_view = pickle_bytes.decode("latin-1", errors="replace")
        for kw in _FRAMEWORK_KEYWORDS:
            if kw in text_view:
                found.add(kw)
    except Exception:
        pass

    return found


def _classify_modules(modules: set[str]) -> tuple[str | None, str]:
    """Pick the most specific framework keyword present in ``modules``.

    Returns (framework_key, matched_module). ``framework_key`` is one of the
    values in ``_FRAMEWORK_KEYWORDS`` (e.g. "yolo", "sklearn"), or None if
    no framework keyword is present.
    """
    for kw, fw in _FRAMEWORK_KEYWORDS.items():
        if kw in modules:
            return fw, kw
        for m in modules:
            if m.startswith(kw + ".") or m == kw:
                return fw, m
    return None, ""


def _pt_pickle_bytes(path: Path) -> bytes | None:
    """PyTorch saves are zip archives. Pull the main pickle (data.pkl) out
    without unpickling. Returns None if the file isn't a torch zip save
    (e.g. legacy pickle-only saves)."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                fname = info.filename
                if fname == "data.pkl" or fname.endswith("/data.pkl"):
                    with zf.open(info) as f:
                        return f.read()
    except (zipfile.BadZipFile, OSError) as e:
        logger.debug("Not a torch-zip save: %s", e)
    except Exception as e:
        logger.debug("Reading torch zip pickle failed: %s", e)
    return None


def _pt_zip_member_names(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.namelist()
    except Exception:
        return []


def _is_torchscript_zip(path: Path) -> bool:
    """TorchScript bundles always include constants.pkl plus a code/ dir."""
    names = _pt_zip_member_names(path)
    has_constants = any(n.endswith("/constants.pkl") or n == "constants.pkl" for n in names)
    has_code = any("/code/" in n or n.startswith("code/") for n in names)
    return has_constants and has_code


def _try_torch_jit_load(path: Path) -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    try:
        torch.jit.load(str(path), map_location="cpu")
        return True
    except Exception:
        return False


# ── per-format inspectors ─────────────────────────────────────────────


def _inspect_pytorch(path: Path) -> dict[str, Any]:
    """.pt / .pth — distinguish TorchScript / YOLO / state-dict / unknown."""
    suffix = path.suffix.lower()

    # 1. TorchScript: structure of zip is unambiguous.
    if _is_torchscript_zip(path) or _try_torch_jit_load(path):
        return _make_result(
            framework="pytorch",
            variant="torchscript",
            confidence="high",
            deployable=True,
            estimator_class="torch.jit.ScriptModule",
            task_type="dl",
            detail="TorchScript archive (constants.pkl + code/ present)",
            reason="",
        )

    # 2. Inspect the pickle inside the torch zip without unpickling.
    pkl_bytes = _pt_pickle_bytes(path)
    if pkl_bytes is None:
        # Legacy pickle-only save. Try a raw pickle scan of the whole file.
        try:
            pkl_bytes = path.read_bytes()
        except Exception as e:
            return _make_result(
                framework="pytorch",
                variant=None,
                confidence="low",
                deployable=False,
                task_type="dl",
                estimator_class=None,
                detail=f"Could not read .{suffix.lstrip('.')} file: {e}",
                reason=f"File could not be read for inspection: {e}",
            )

    modules = _scan_pickle_modules(pkl_bytes)
    fw_key, matched = _classify_modules(modules)

    if fw_key == "yolo":
        return _make_result(
            framework="yolo",
            variant="yolo",
            confidence="high",
            deployable=True,
            estimator_class=f"ultralytics ({matched})",
            task_type="dl",
            detail=f"Ultralytics YOLO checkpoint detected via pickle module ref: {matched}",
            reason="",
        )

    # 3. Plain state-dict — only torch / collections / numpy refs.
    non_neutral = {
        m for m in modules
        if m not in _NEUTRAL_MODULES and not any(m.startswith(prefix + ".") for prefix in _NEUTRAL_MODULES)
    }
    if not non_neutral:
        return _make_result(
            framework="pytorch",
            variant="state_dict",
            confidence="high",
            deployable=False,
            estimator_class="torch.state_dict (OrderedDict of tensors)",
            task_type="dl",
            detail="Pickle references only torch / collections / numpy — looks like a plain state dict.",
            reason=(
                "This .pt file appears to be a state-dict (raw weights), not a deployable model. "
                "Re-save with `torch.jit.save(model, ...)` (TorchScript) before deploying."
            ),
        )

    # 4. Pickle references some external class but not one we know how to deploy.
    return _make_result(
        framework="pytorch",
        variant=None,
        confidence="low",
        deployable=False,
        estimator_class=f"pickled python object ({sorted(non_neutral)[:3]!r})",
        task_type="dl",
        detail=f"Pickle references unknown classes: {sorted(non_neutral)[:5]!r}",
        reason=(
            "This .pt file references custom Python classes that aren't bundled in the deployment. "
            "Re-export your model as TorchScript (`torch.jit.save`) so it can load standalone."
        ),
    )


def _inspect_pickle(path: Path) -> dict[str, Any]:
    """.pkl / .pickle / .joblib / .sav — classify by pickle module refs first,
    then fall back to joblib.load() for richer metadata."""
    sklearn_version: str | None = None
    pkl_modules: set[str] = set()

    # Content inspection without execution.
    try:
        raw = path.read_bytes()
        pkl_modules = _scan_pickle_modules(raw)
    except Exception as e:
        logger.debug("Could not read pickle bytes for opcode scan: %s", e)

    fw_key, matched = _classify_modules(pkl_modules)

    # Try a real load for n_features_in / has_predict_proba / sklearn_version.
    obj = None
    load_err: str | None = None
    try:
        obj = joblib.load(path)
    except Exception as e:
        load_err = str(e)
        logger.warning("joblib.load failed (likely version mismatch): %s", e)

    # Best-effort sklearn version regardless of which path succeeded.
    try:
        if obj is not None:
            sklearn_version = (
                getattr(obj, "__sklearn_version__", None)
                or getattr(obj, "_sklearn_version", None)
            )
        if sklearn_version is None:
            with open(path, "rb") as f:
                head = pickle.load(f)
            sklearn_version = (
                getattr(head, "__sklearn_version__", None)
                or getattr(head, "_sklearn_version", None)
            )
    except Exception:
        pass

    if obj is not None:
        framework, detail = _classify_object(obj)
        if fw_key and framework == "unknown":
            # Pickle scan said something useful even though _classify_object
            # didn't — trust the scan.
            framework = fw_key
        return _make_result(
            framework=framework,
            variant=None,
            confidence="high",
            deployable=framework != "unknown",
            estimator_class=f"{type(obj).__module__}.{type(obj).__name__}",
            has_predict_proba=hasattr(obj, "predict_proba"),
            n_features_in=getattr(obj, "n_features_in_", None),
            task_type="ml",
            detail=detail,
            sklearn_version=sklearn_version,
            reason="" if framework != "unknown" else "Loaded object did not match any supported framework.",
        )

    # Joblib load failed — rely on the pickle scan alone.
    if fw_key in {"sklearn", "xgboost", "lightgbm", "catboost"}:
        return _make_result(
            framework=fw_key,
            variant=None,
            confidence="low",
            deployable=True,
            estimator_class=None,
            task_type="ml",
            detail=(
                f"Pickle module scan matched '{matched}'. Full load failed "
                f"({load_err}); proceeding on scan match alone."
            ),
            sklearn_version=sklearn_version,
            reason="",
        )

    return _make_result(
        framework="unknown",
        variant=None,
        confidence="low",
        deployable=False,
        estimator_class=None,
        task_type="ml",
        detail=f"Pickle scan inconclusive (modules={sorted(pkl_modules)[:5]!r}); load error: {load_err}",
        sklearn_version=sklearn_version,
        reason=(
            "Could not identify the framework: the file failed to load and its pickle did not "
            "reference any known ML framework (sklearn / xgboost / lightgbm / catboost)."
        ),
    )


def _inspect_keras(path: Path) -> dict[str, Any]:
    """.h5 (HDF5 magic) or .keras (zip archive)."""
    suffix = path.suffix.lower()
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except Exception as e:
        return _make_result(
            framework="tensorflow",
            confidence="low",
            deployable=False,
            task_type="dl",
            detail=f"Could not read file head: {e}",
            reason=f"File could not be read: {e}",
        )

    if suffix == ".h5":
        if head.startswith(HDF5_MAGIC):
            return _make_result(
                framework="tensorflow",
                variant=None,
                confidence="high",
                deployable=True,
                estimator_class="tf.keras.model",
                task_type="dl",
                detail="HDF5 magic bytes verified",
                reason="",
            )
        return _make_result(
            framework="tensorflow",
            confidence="low",
            deployable=False,
            task_type="dl",
            detail=f"Expected HDF5 magic, got {head!r}",
            reason="File has .h5 extension but is not a valid HDF5 archive.",
        )

    # .keras
    if head.startswith(ZIP_MAGIC):
        return _make_result(
            framework="tensorflow",
            variant=None,
            confidence="high",
            deployable=True,
            estimator_class="keras.Model (zip archive)",
            task_type="dl",
            detail="Keras v3 zip archive (PK magic verified)",
            reason="",
        )
    return _make_result(
        framework="tensorflow",
        confidence="low",
        deployable=False,
        task_type="dl",
        detail=f"Expected PK zip magic, got {head!r}",
        reason="File has .keras extension but is not a valid zip archive.",
    )


def _inspect_onnx(path: Path) -> dict[str, Any]:
    """ONNX has no fixed magic; trust the extension and let onnxruntime
    verify by trying a session at spec-extraction time."""
    return _make_result(
        framework="onnx",
        variant=None,
        confidence="high",
        deployable=True,
        estimator_class="onnx.ModelProto",
        task_type="dl",
        detail="ONNX file (protobuf format — verified at runtime by onnxruntime)",
        reason="",
    )


def _inspect_xgb_json(path: Path) -> dict[str, Any]:
    """XGBoost JSON dumps always have a top-level 'learner' key. UBJ is
    binary-JSON — we trust the extension since text JSON parsing won't work."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return _make_result(
                framework="xgboost",
                confidence="low",
                deployable=False,
                task_type="ml",
                detail=f"JSON parse failed: {e}",
                reason=f"File has .json extension but is not valid JSON: {e}",
            )
        if isinstance(data, dict) and "learner" in data:
            return _make_result(
                framework="xgboost",
                variant=None,
                confidence="high",
                deployable=True,
                estimator_class="xgboost.Booster",
                task_type="ml",
                detail="XGBoost JSON ('learner' key present)",
                reason="",
            )
        return _make_result(
            framework="unknown",
            confidence="low",
            deployable=False,
            task_type="ml",
            detail="JSON file but no 'learner' key — not an XGBoost dump.",
            reason="JSON does not look like an XGBoost model dump (missing 'learner' key).",
        )

    # .ubj — binary JSON, can't text-parse. Trust extension.
    return _make_result(
        framework="xgboost",
        variant=None,
        confidence="high",
        deployable=True,
        estimator_class="xgboost.Booster",
        task_type="ml",
        detail="UBJSON binary dump (extension trust — content not parsed)",
        reason="",
    )


# ── shape-extraction helpers (unchanged) ──────────────────────────────


def _coerce_dim(value: Any) -> int | None:
    try:
        if value is None:
            return None
        v = int(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _spec_from_shape(shape: tuple, channel_order: str) -> dict[str, Any]:
    if channel_order == "NHWC":
        if len(shape) != 4:
            return _dynamic_spec(reason=f"Unexpected NHWC rank: shape={shape!r}")
        _, h, w, c = shape
    else:
        if len(shape) != 4:
            return _dynamic_spec(reason=f"Unexpected NCHW rank: shape={shape!r}")
        _, c, h, w = shape

    height = _coerce_dim(h)
    width = _coerce_dim(w)
    channels = _coerce_dim(c)

    if height is None or width is None or channels is None:
        logger.warning(
            "Dynamic input shape detected (h=%s w=%s c=%s); falling back to defaults.",
            h, w, c,
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
    `n_features_in_` and no image spec is needed. YOLO also returns None
    because Ultralytics handles its own preprocessing.
    """
    fw = (framework or "").lower()
    if fw in {"sklearn", "xgboost", "lightgbm", "catboost", "joblib", "unknown", "yolo"}:
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


# ── public entry point ────────────────────────────────────────────────


def inspect_model_file(model_path: Path) -> dict[str, Any]:
    """
    Identify the deployment framework by inspecting file *content*, with
    extension as a secondary hint only.

    Returns a dict including the legacy keys (framework, estimator_class,
    has_predict_proba, n_features_in, task_type, detail, sklearn_version)
    plus four new keys:
      - variant: "torchscript" | "yolo" | "state_dict" | None
      - confidence: "high" | "low"
      - deployable: bool
      - reason: str (empty when deployable=True)
    """
    path = model_path.resolve()
    suffix = path.suffix.lower()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported model extension: {path.suffix}")

    if suffix in {".pt", ".pth"}:
        return _inspect_pytorch(path)
    if suffix == ".onnx":
        return _inspect_onnx(path)
    if suffix in {".h5", ".keras"}:
        return _inspect_keras(path)
    if suffix in XGB_EXTENSIONS:
        return _inspect_xgb_json(path)
    if suffix in ML_EXTENSIONS:
        return _inspect_pickle(path)

    return _make_result(
        framework="unknown",
        confidence="low",
        deployable=False,
        detail=f"No inspector for extension {suffix}",
        reason=f"Unsupported file extension: {suffix}",
    )


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
    if hasattr(obj, "steps"):
        return "sklearn", f"pipeline-like ({name}); steps detected"
    return "unknown", f"module={mod!r} name={name}"


def detect_with_llm(model_path: Path, inspection_error: str | None) -> dict[str, Any]:
    """LLM fallback when content inspection couldn't classify the file. Returns the
    same shape as ``inspect_model_file`` (legacy keys + new fields)."""
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
    task = data.get("task_type") if data.get("task_type") in {"ml", "dl"} else (
        "dl" if fw in {"pytorch", "tensorflow", "onnx"} else "ml"
    )
    return _make_result(
        framework=fw,
        variant=None,
        confidence="low",
        deployable=fw != "unknown",
        estimator_class=data.get("estimator_class"),
        has_predict_proba=bool(data.get("has_predict_proba", False)),
        n_features_in=data.get("n_features_in"),
        task_type=task,
        detail=data.get("detail", "llm"),
        reason="" if fw != "unknown" else "LLM could not classify the artifact.",
    )
