"""
HuggingFace Spaces deployer — uploads a model + Gradio app to HF Spaces.

Completely independent of the Docker pipeline.
"""
from __future__ import annotations

import random
import re
import string
import textwrap
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, SpaceStage, create_repo


IMAGE_MODEL_TYPES = {
    "Image Classification",
    "Image Segmentation",
    "Object Detection",
}

PIL_FILTER_MAP = {
    "BILINEAR": "Image.BILINEAR",
    "BICUBIC": "Image.BICUBIC",
    "NEAREST": "Image.NEAREST",
    "LANCZOS": "Image.LANCZOS",
}

VALID_NORMALIZATIONS = {"div255", "imagenet", "minus1to1", "raw"}
VALID_COLOR_ORDERS = {"RGB", "BGR"}
VALID_LAYOUTS = {"NCHW", "NHWC"}


def _normalize_spec(
    input_spec: dict | None,
    framework: str,
    is_segmentation: bool,
) -> dict:
    """Fill in defaults for any missing keys; clamp invalid values."""
    spec = dict(input_spec or {})
    fallback_size = 256 if is_segmentation else 224
    height = spec.get("height") or fallback_size
    width = spec.get("width") or fallback_size
    channels = spec.get("channels") or 3
    layout = spec.get("channel_order")
    if layout not in VALID_LAYOUTS:
        layout = "NCHW" if framework == "pytorch" else "NHWC"
    norm = spec.get("normalization")
    if norm not in VALID_NORMALIZATIONS:
        norm = "div255"
    color = spec.get("channel_color_order")
    if color not in VALID_COLOR_ORDERS:
        color = "RGB"
    interp = spec.get("interpolation")
    if interp not in PIL_FILTER_MAP:
        interp = "BILINEAR"
    return {
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "channel_order": layout,
        "normalization": norm,
        "channel_color_order": color,
        "interpolation": interp,
    }


def _build_preprocess_lines(spec: dict) -> str:
    """
    Lines (no leading indent) that take `img` (PIL) and produce `arr` (ndarray).
    Uses `height_used`, `width_used` for the resize target.
    """
    ch = spec["channels"]
    layout = spec["channel_order"]
    norm = spec["normalization"]
    color = spec["channel_color_order"]
    interp = PIL_FILTER_MAP[spec["interpolation"]]

    pil_mode = "L" if ch == 1 else "RGB"

    lines: list[str] = []
    lines.append(f'img = img.convert("{pil_mode}").resize((width_used, height_used), {interp})')
    lines.append("np_img = np.asarray(img, dtype=np.float32)")

    if norm == "div255":
        lines.append("np_img = np_img / 255.0")
    elif norm == "minus1to1":
        lines.append("np_img = (np_img / 127.5) - 1.0")
    elif norm == "imagenet":
        if ch == 3:
            lines.append("np_img = np_img / 255.0")
            lines.append("_imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)")
            lines.append("_imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)")
            lines.append("np_img = (np_img - _imagenet_mean) / _imagenet_std")
        else:
            lines.append("np_img = np_img / 255.0")
            lines.append("np_img = (np_img - 0.485) / 0.229")
    # "raw" → leave float32 0..255 untouched

    if ch == 1:
        lines.append("if np_img.ndim == 2:")
        lines.append("    np_img = np_img[..., None]")
    elif color == "BGR":
        lines.append("# RGB -> BGR")
        lines.append("np_img = np_img[..., ::-1]")

    if layout == "NCHW":
        lines.append("# NCHW (1, C, H, W)")
        lines.append("arr = np.transpose(np_img, (2, 0, 1))[None, ...]")
    else:
        lines.append("# NHWC (1, H, W, C)")
        lines.append("arr = np_img[None, ...]")

    return "\n".join(lines)


def generate_space_name(model_filename: str) -> str:
    name = Path(model_filename).stem
    name = re.sub(r"[^a-zA-Z0-9-]", "-", name).lower().strip("-")
    return f"{name}-deploy"


def check_space_available(api: HfApi, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        return False  # exists = not available
    except Exception:
        return True  # 404 = available


def resolve_space_name(api: HfApi, username: str, preferred_name: str) -> str:
    repo_id = f"{username}/{preferred_name}"
    if check_space_available(api, repo_id):
        return preferred_name
    suffix = "".join(random.choices(string.digits, k=4))
    return f"{preferred_name}-{suffix}"


def _framework_pieces(framework: str, model_filename: str) -> dict:
    """Return per-framework imports, load code, array-predict helper, and text invoke patterns."""
    if framework in ("sklearn", "joblib", "unknown"):
        return {
            "imports": "import joblib\nimport numpy as np",
            "load_code": f'model = joblib.load("{model_filename}")',
            "predict_array_helper": (
                "def _predict_array(X):\n"
                "    return model.predict(X)\n"
                "\n"
                "def _predict_proba(X):\n"
                "    if hasattr(model, 'predict_proba'):\n"
                "        try:\n"
                "            return model.predict_proba(X)\n"
                "        except Exception:\n"
                "            return None\n"
                "    return None\n"
            ),
            "text_invoke_api": "model.predict([req.text])",
            "text_invoke_ui": "model.predict([input_text])",
        }
    if framework == "pytorch":
        return {
            "imports": "import torch\nimport numpy as np",
            "load_code": (
                f'model = torch.jit.load("{model_filename}", map_location="cpu")\n'
                "model.eval()"
            ),
            "predict_array_helper": (
                "def _predict_array(X):\n"
                "    x = torch.tensor(X, dtype=torch.float32)\n"
                "    with torch.no_grad():\n"
                "        out = model(x)\n"
                "    return out.detach().cpu().numpy()\n"
                "\n"
                "def _predict_proba(X):\n"
                "    return None  # DL models: rely on raw output + softmax heuristic\n"
            ),
            "text_invoke_api": "model([req.text])",
            "text_invoke_ui": "model([input_text])",
        }
    if framework in ("tensorflow", "keras"):
        return {
            "imports": "import numpy as np\nimport keras",
            "load_code": f'model = keras.models.load_model("{model_filename}", compile=False)',
            "predict_array_helper": (
                "def _predict_array(X):\n"
                "    return model.predict(X.astype(np.float32), verbose=0)\n"
                "\n"
                "def _predict_proba(X):\n"
                "    return None  # DL models: rely on raw output + softmax heuristic\n"
            ),
            "text_invoke_api": "model.predict([req.text], verbose=0)",
            "text_invoke_ui": "model.predict([input_text], verbose=0)",
        }
    if framework == "onnx":
        return {
            "imports": "import numpy as np\nimport onnxruntime as ort",
            "load_code": (
                f'session = ort.InferenceSession("{model_filename}", providers=["CPUExecutionProvider"])\n'
                "input_name = session.get_inputs()[0].name"
            ),
            "predict_array_helper": (
                "def _predict_array(X):\n"
                "    return session.run(None, {input_name: X.astype(np.float32)})[0]\n"
                "\n"
                "def _predict_proba(X):\n"
                "    return None  # DL models: rely on raw output + softmax heuristic\n"
            ),
            "text_invoke_api": 'session.run(None, {input_name: np.array([req.text])})[0]',
            "text_invoke_ui": 'session.run(None, {input_name: np.array([input_text])})[0]',
        }
    if framework == "yolo":
        return {
            "imports": (
                "import sys\n"
                "import types\n"
                "import numpy as np\n"
                "\n"
                "# huggingface_hub compat shim. Both ultralytics (DetectMultiBackend)\n"
                "# and the yolov5 PyPI package do `from huggingface_hub.utils._errors\n"
                "# import RepositoryNotFoundError`. That private submodule was removed\n"
                "# in huggingface_hub 0.28+; errors live at huggingface_hub.errors and\n"
                "# are re-exported at huggingface_hub.utils. Recreate the missing\n"
                "# module so the legacy imports keep working without pinning hf_hub.\n"
                "try:\n"
                "    import huggingface_hub.utils._errors  # noqa: F401\n"
                "except ImportError:\n"
                "    import huggingface_hub.utils as _hf_utils\n"
                "    try:\n"
                "        import huggingface_hub.errors as _hf_errors\n"
                "    except ImportError:\n"
                "        _hf_errors = None\n"
                "    _shim = types.ModuleType('huggingface_hub.utils._errors')\n"
                "    for _name in (\n"
                "        'RepositoryNotFoundError', 'RevisionNotFoundError',\n"
                "        'EntryNotFoundError', 'LocalEntryNotFoundError',\n"
                "        'BadRequestError', 'HfHubHTTPError',\n"
                "        'OfflineModeIsEnabled', 'GatedRepoError',\n"
                "        'DisabledRepoError',\n"
                "    ):\n"
                "        _val = getattr(_hf_utils, _name, None)\n"
                "        if _val is None and _hf_errors is not None:\n"
                "            _val = getattr(_hf_errors, _name, None)\n"
                "        if _val is not None:\n"
                "            setattr(_shim, _name, _val)\n"
                "    sys.modules['huggingface_hub.utils._errors'] = _shim\n"
                "\n"
                "# torch.load weights_only compat. PyTorch 2.6 flipped the default of\n"
                "# weights_only from False to True. ultralytics's DetectMultiBackend\n"
                "# and yolov5's attempt_load both call torch.load(...) without that\n"
                "# kwarg, which now refuses to deserialize the model classes baked\n"
                "# into legacy YOLO checkpoints (e.g. models.yolo.DetectionModel).\n"
                "# Restore the old default — the user explicitly chose this file for\n"
                "# deployment, so trust is established at upload time.\n"
                "import torch as _torch\n"
                "_orig_torch_load = _torch.load\n"
                "def _torch_load_compat(*args, **kwargs):\n"
                "    kwargs.setdefault('weights_only', False)\n"
                "    return _orig_torch_load(*args, **kwargs)\n"
                "_torch.load = _torch_load_compat"
            ),
            "load_code": (
                'MODEL_BACKEND = "ultralytics"\n'
                "try:\n"
                "    from ultralytics import YOLO as _UltraYOLO\n"
                f'    model = _UltraYOLO("{model_filename}")\n'
                "except TypeError as _ultra_exc:\n"
                '    if "YOLOv5" in str(_ultra_exc):\n'
                "        import yolov5 as _yolov5\n"
                f'        model = _yolov5.load("{model_filename}")\n'
                '        MODEL_BACKEND = "yolov5"\n'
                "    else:\n"
                "        raise"
            ),
            "predict_array_helper": (
                "def _yolo_run(img):\n"
                "    if MODEL_BACKEND == 'ultralytics':\n"
                "        results = model(img, verbose=False)\n"
                "        result = results[0]\n"
                "        boxes = result.boxes\n"
                "        names = result.names\n"
                "        if boxes is None or len(boxes) == 0:\n"
                "            return [], names\n"
                "        xyxy = boxes.xyxy.cpu().numpy()\n"
                "        conf = boxes.conf.cpu().numpy()\n"
                "        cls = boxes.cls.cpu().numpy().astype(int)\n"
                "        return list(zip(xyxy, conf, cls)), names\n"
                "    # yolov5 backend: results.xyxy[0] is [N, 6] tensor\n"
                "    results = model(img)\n"
                "    names = results.names\n"
                "    preds = results.xyxy[0]\n"
                "    if hasattr(preds, 'cpu'):\n"
                "        preds = preds.cpu().numpy()\n"
                "    else:\n"
                "        preds = np.asarray(preds)\n"
                "    if preds.size == 0:\n"
                "        return [], names\n"
                "    xyxy = preds[:, :4]\n"
                "    conf = preds[:, 4]\n"
                "    cls = preds[:, 5].astype(int)\n"
                "    return list(zip(xyxy, conf, cls)), names\n"
            ),
            "text_invoke_api": "",
            "text_invoke_ui": "",
        }
    # fallback
    return {
        "imports": "import joblib\nimport numpy as np",
        "load_code": f'model = joblib.load("{model_filename}")',
        "predict_array_helper": (
            "def _predict_array(X):\n"
            "    return model.predict(X)\n"
            "\n"
            "def _predict_proba(X):\n"
            "    if hasattr(model, 'predict_proba'):\n"
            "        try:\n"
            "            return model.predict_proba(X)\n"
            "        except Exception:\n"
            "            return None\n"
            "    return None\n"
        ),
        "text_invoke_api": "model.predict([req.text])",
        "text_invoke_ui": "model.predict([input_text])",
    }


_CLASSIFICATION_HELPER = textwrap.dedent("""\
    def _shape_classification_output(arr_like):
        \"\"\"Apply a softmax-distribution heuristic to a model output and,
        if it looks like a probability vector, return the rich classification
        shape; otherwise return None so the caller can fall back to a raw
        {prediction: ...} response.

        Heuristic: after np.squeeze, the array must be 1-D with length >= 2,
        all values in [0, 1], and sum within 1e-3 of 1.0. This catches
        sklearn predict_proba output and DL outputs where softmax was applied
        in the network's final layer; it correctly skips raw logits, scalar
        regressor outputs, and class-label outputs from sklearn.predict.
        \"\"\"
        try:
            arr = np.squeeze(np.asarray(arr_like))
        except Exception:
            return None
        if arr.ndim != 1 or arr.shape[0] < 2:
            return None
        try:
            mn = float(arr.min())
            mx = float(arr.max())
            sm = float(arr.sum())
        except Exception:
            return None
        if mn < 0.0 or mx > 1.0 or abs(sm - 1.0) > 1e-3:
            return None
        pc = int(np.argmax(arr))
        return {
            "predicted_class": pc,
            "confidence": float(arr[pc]),
            "all_probabilities": arr.tolist(),
        }
    """)


def _segmentation_output_block() -> str:
    # All values returned to FastAPI are forced through plain Python ints /
    # str so the response serializer can never trip on numpy scalars (numpy
    # ints/floats inside a dict cause TypeError in jsonable_encoder on some
    # numpy versions, which would surface as a generic 500 with the body
    # {"detail": "Internal Server Error"} — the exact failure mode users
    # have been hitting on dental-resnet34-unet).
    return textwrap.dedent("""\
        arr_pred = np.asarray(pred)
        if arr_pred.ndim == 4 and arr_pred.shape[0] == 1:
            arr_pred = arr_pred[0]
        # Move channel-first (PyTorch) to channel-last for consistent argmax.
        if arr_pred.ndim == 3 and arr_pred.shape[0] < arr_pred.shape[-1] and arr_pred.shape[0] <= 64:
            arr_pred = np.transpose(arr_pred, (1, 2, 0))
        if arr_pred.ndim == 3 and arr_pred.shape[-1] > 1:
            mask = np.argmax(arr_pred, axis=-1)
        elif arr_pred.ndim == 3 and arr_pred.shape[-1] == 1:
            mask = arr_pred[..., 0]
        else:
            mask = arr_pred
        if mask.dtype != np.uint8:
            mx = float(mask.max()) if mask.size else 1.0
            if mx <= 1.0:
                mask = (mask > 0.5).astype(np.uint8) * 255
            else:
                scale = 255.0 / max(1.0, mx)
                mask = (mask.astype(np.float32) * scale).astype(np.uint8)
        pil_mask = Image.fromarray(mask.astype(np.uint8))
        buf = io.BytesIO()
        pil_mask.save(buf, format="PNG")
        mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return {
            "mask_png_base64": str(mask_b64),
            "shape": [int(d) for d in mask.shape],
            "input_size": [int(height_used), int(width_used)],
        }
        """)


def _build_yolo_bodies() -> tuple[str, str]:
    """Return (api_predict_body, ui_predict_body) for YOLO models.

    Uses _yolo_run() which transparently handles both Ultralytics (YOLOv8+)
    and yolov5 backends via a MODEL_BACKEND flag set at load time.
    """
    api_body = textwrap.dedent("""\
        try:
            img_bytes = base64.b64decode(req.image)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            rows, names = _yolo_run(img)
            detections = []
            for box, score, c in rows:
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                detections.append({
                    "class": label,
                    "confidence": float(score),
                    "bbox": [float(v) for v in box],
                })
            return {"detections": detections}
        except Exception as exc:
            tb_lines = traceback.format_exc().splitlines()[-10:]
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "traceback": tb_lines,
                },
            )
        """)

    ui_body = textwrap.dedent("""\
        try:
            if input_image is None:
                return "No image provided."
            img = input_image.convert("RGB") if input_image.mode != "RGB" else input_image
            rows, names = _yolo_run(img)
            if not rows:
                return "No objects detected."
            lines = []
            for box, score, c in rows:
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                x1, y1, x2, y2 = [float(v) for v in box]
                lines.append(f"{label}: {score:.3f}  bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            return "\\n".join(lines)
        except Exception as exc:
            return f"Error: {exc}"
        """)

    return api_body, ui_body


def _build_image_bodies(spec: dict, is_segmentation: bool) -> tuple[str, str]:
    """Return (api_predict_body, ui_predict_body) for image model types, using
    the confirmed input spec to bake exact preprocessing values."""
    preprocess = _build_preprocess_lines(spec)

    if is_segmentation:
        output_block = _segmentation_output_block()
    else:
        # Image classification: try the softmax heuristic on the raw output;
        # rich shape if it looks like a probability vector, else raw array.
        # tolist() yields Python types; we still cast input_size defensively
        # to dodge a numpy-int leak that would crash JSON serialization.
        output_block = textwrap.dedent("""\
            arr_pred = np.asarray(pred)
            _rich = _shape_classification_output(arr_pred)
            if _rich is not None:
                _rich["input_size"] = [int(height_used), int(width_used)]
                return _rich
            return {"prediction": arr_pred.tolist(), "input_size": [int(height_used), int(width_used)]}
            """)

    api_body = (
        "try:\n"
        f"    height_used = int(req.size) if req.size else {spec['height']}\n"
        f"    width_used = int(req.size) if req.size else {spec['width']}\n"
        "    img_bytes = base64.b64decode(req.image)\n"
        "    img = Image.open(io.BytesIO(img_bytes))\n"
        + textwrap.indent(preprocess, "    ") + "\n"
        "    pred = _predict_array(arr)\n"
        + textwrap.indent(output_block, "    ")
        + "except Exception as exc:\n"
        "    tb_lines = traceback.format_exc().splitlines()[-10:]\n"
        "    return JSONResponse(\n"
        "        status_code=500,\n"
        "        content={\n"
        "            \"error\": str(exc),\n"
        "            \"type\": type(exc).__name__,\n"
        "            \"hint\": \"If shape mismatch: retry with size=256, 320, 384, or 512\",\n"
        "            \"traceback\": tb_lines,\n"
        "        },\n"
        "    )\n"
    )

    ui_body = (
        "try:\n"
        "    if input_image is None:\n"
        "        return \"No image provided.\"\n"
        f"    height_used = {spec['height']}\n"
        f"    width_used = {spec['width']}\n"
        "    img = input_image\n"
        + textwrap.indent(preprocess, "    ") + "\n"
        "    pred = _predict_array(arr)\n"
        "    arr_pred = np.asarray(pred)\n"
        "    return f\"shape={arr_pred.shape}, sample={arr_pred.flatten()[:10].tolist()}\"\n"
        "except Exception as exc:\n"
        "    return f\"Error: {exc}\"\n"
    )

    return api_body, ui_body


def generate_gradio_app(
    framework: str,
    model_filename: str,
    model_type: str,
    input_spec: dict | None = None,
) -> str:
    """
    Build a single-file FastAPI + Gradio inference app.

    FastAPI is the parent app: owns /predict and /health (both
    include_in_schema=False) and mounts Gradio at "/" via
    gr.mount_gradio_app. The Gradio handler is named `predict_ui` so it does
    not collide with the FastAPI `predict_api` function.

    When `model_type` is an image type, `input_spec` (height/width/channels/
    channel_order/normalization/channel_color_order/interpolation) is baked
    directly into the preprocessing block — no runtime guessing.
    """
    is_yolo = framework == "yolo"
    if is_yolo:
        # YOLO is image-only; its own preprocessing handles size/layout.
        mt = "Object Detection"
    else:
        mt = (model_type or "Tabular/Regression").strip()
    is_image = mt in IMAGE_MODEL_TYPES
    is_segmentation = mt == "Image Segmentation"
    is_text = mt == "Text Classification"
    is_timeseries = mt == "Time Series"
    is_other = mt == "Other"

    pieces = _framework_pieces(framework, model_filename)
    framework_imports = pieces["imports"]
    load_code = pieces["load_code"]
    predict_array_helper = pieces["predict_array_helper"]
    text_invoke_api = pieces["text_invoke_api"]
    text_invoke_ui = pieces["text_invoke_ui"]

    # `traceback` is always required by the global exception handler we install
    # below so that uncaught errors (incl. response-serialization failures
    # that bypass the per-route try/except) come back with a proper Python
    # traceback instead of FastAPI's opaque {"detail": "Internal Server Error"}.
    extra_imports = "from typing import List, Optional\nimport traceback"
    if is_image or is_yolo:
        extra_imports += "\nimport base64\nimport io\nfrom PIL import Image"

    if is_yolo:
        api_predict_body, ui_predict_body = _build_yolo_bodies()
        ui_input_widget = 'gr.Image(type="pil", label="Upload image")'
        ui_input_param = "input_image"
        schema_class = (
            "class PredictRequest(BaseModel):\n"
            "    image: str  # base64-encoded image bytes\n"
        )
    elif is_image:
        spec = _normalize_spec(input_spec, framework=framework, is_segmentation=is_segmentation)
        api_predict_body, ui_predict_body = _build_image_bodies(
            spec=spec,
            is_segmentation=is_segmentation,
        )
        ui_input_widget = 'gr.Image(type="pil", label="Upload image")'
        ui_input_param = "input_image"
        schema_class = (
            "class PredictRequest(BaseModel):\n"
            "    image: str  # base64-encoded image bytes\n"
            "    size: Optional[int] = None\n"
        )
    elif is_text:
        schema_class = (
            "class PredictRequest(BaseModel):\n"
            "    text: str\n"
        )
        api_predict_body = (
            "_proba = _predict_proba([req.text])\n"
            "if _proba is not None:\n"
            "    _rich = _shape_classification_output(_proba)\n"
            "    if _rich is not None:\n"
            "        return _rich\n"
            f"pred = {text_invoke_api}\n"
            "_rich = _shape_classification_output(pred)\n"
            "if _rich is not None:\n"
            "    return _rich\n"
            "return {\"prediction\": pred.tolist() if hasattr(pred, \"tolist\") else (list(pred) if hasattr(pred, \"__iter__\") else pred)}\n"
        )
        ui_input_widget = 'gr.Textbox(placeholder="Enter text…", label="Text")'
        ui_input_param = "input_text"
        ui_predict_body = (
            f"pred = {text_invoke_ui}\n"
            "return str(pred.tolist() if hasattr(pred, \"tolist\") else (list(pred) if hasattr(pred, \"__iter__\") else pred))\n"
        )
    elif is_timeseries:
        schema_class = (
            "class PredictRequest(BaseModel):\n"
            "    sequence: List[float]\n"
        )
        api_predict_body = (
            "X = np.asarray([req.sequence])\n"
            "_proba = _predict_proba(X)\n"
            "if _proba is not None:\n"
            "    _rich = _shape_classification_output(_proba)\n"
            "    if _rich is not None:\n"
            "        return _rich\n"
            "pred = _predict_array(X)\n"
            "_rich = _shape_classification_output(pred)\n"
            "if _rich is not None:\n"
            "    return _rich\n"
            "return {\"prediction\": pred.tolist() if hasattr(pred, \"tolist\") else pred}\n"
        )
        ui_input_widget = 'gr.Textbox(placeholder="1.0, 2.1, 3.2, …", label="Sequence (comma-separated)")'
        ui_input_param = "input_text"
        ui_predict_body = (
            "values = [float(x.strip()) for x in (input_text or \"\").split(\",\") if x.strip()]\n"
            "X = np.asarray([values])\n"
            "pred = _predict_array(X)\n"
            "return str(pred.tolist() if hasattr(pred, \"tolist\") else pred)\n"
        )
    elif is_other:
        schema_class = (
            "class PredictRequest(BaseModel):\n"
            "    data: str\n"
        )
        api_predict_body = (
            "try:\n"
            "    values = [float(x.strip()) for x in (req.data or \"\").split(\",\") if x.strip()]\n"
            "    X = np.asarray([values])\n"
            "    pred = _predict_array(X)\n"
            "    return {\"prediction\": pred.tolist() if hasattr(pred, \"tolist\") else pred}\n"
            "except Exception as exc:\n"
            "    return JSONResponse(status_code=500, content={\"error\": str(exc), \"type\": type(exc).__name__})\n"
        )
        ui_input_widget = 'gr.Textbox(placeholder="Free-form input…", label="Data")'
        ui_input_param = "input_text"
        ui_predict_body = (
            "try:\n"
            "    values = [float(x.strip()) for x in (input_text or \"\").split(\",\") if x.strip()]\n"
            "    X = np.asarray([values])\n"
            "    pred = _predict_array(X)\n"
            "    return str(pred.tolist() if hasattr(pred, \"tolist\") else pred)\n"
            "except Exception as exc:\n"
            "    return f\"Error: {exc}\"\n"
        )
    else:
        # Tabular / Regression (default)
        schema_class = (
            "class PredictRequest(BaseModel):\n"
            "    features: List[float]\n"
        )
        api_predict_body = (
            "X = np.asarray([req.features])\n"
            "_proba = _predict_proba(X)\n"
            "if _proba is not None:\n"
            "    _rich = _shape_classification_output(_proba)\n"
            "    if _rich is not None:\n"
            "        return _rich\n"
            "pred = _predict_array(X)\n"
            "_rich = _shape_classification_output(pred)\n"
            "if _rich is not None:\n"
            "    return _rich\n"
            "return {\"prediction\": pred.tolist() if hasattr(pred, \"tolist\") else pred}\n"
        )
        ui_input_widget = 'gr.Textbox(placeholder="1.2, 3.4, 5.6", label="Features (comma-separated)")'
        ui_input_param = "input_text"
        ui_predict_body = (
            "values = [float(x.strip()) for x in (input_text or \"\").split(\",\") if x.strip()]\n"
            "X = np.asarray([values])\n"
            "pred = _predict_array(X)\n"
            "return str(pred.tolist() if hasattr(pred, \"tolist\") else pred)\n"
        )

    title = f"{framework} — {mt}"

    gradio_client_schema_patch = (
        "# --- gradio_client schema-walker compatibility patch -------------------\n"
        "# Pydantic 2 / FastAPI emit JSON Schemas containing boolean sub-schemas\n"
        "# (valid per JSON Schema spec; e.g. additionalProperties: false). All\n"
        "# released gradio_client versions assume every schema node is a dict and\n"
        "# call `if \"x\" in schema:` inside get_type / _json_schema_to_python_type.\n"
        "# That raises `TypeError: argument of type 'bool' is not iterable` when\n"
        "# Gradio's homepage calls /info, returning 500 on every request to /.\n"
        "# Wrap the walker so non-dict schemas degrade to 'Any' instead of crashing.\n"
        "import gradio_client.utils as _gc_utils\n"
        "\n"
        "def _bool_safe_schema(fn):\n"
        "    def wrapper(schema, *args, **kwargs):\n"
        "        if not isinstance(schema, dict):\n"
        "            return \"Any\"\n"
        "        return fn(schema, *args, **kwargs)\n"
        "    return wrapper\n"
        "\n"
        "if hasattr(_gc_utils, \"get_type\"):\n"
        "    _gc_utils.get_type = _bool_safe_schema(_gc_utils.get_type)\n"
        "if hasattr(_gc_utils, \"_json_schema_to_python_type\"):\n"
        "    _gc_utils._json_schema_to_python_type = _bool_safe_schema(\n"
        "        _gc_utils._json_schema_to_python_type\n"
        "    )\n"
        "if hasattr(_gc_utils, \"json_schema_to_python_type\"):\n"
        "    _gc_utils.json_schema_to_python_type = _bool_safe_schema(\n"
        "        _gc_utils.json_schema_to_python_type\n"
        "    )\n"
        "# ----------------------------------------------------------------------\n"
    )

    # The classification heuristic is needed for image classification, text,
    # tabular, and time series — every path that may emit the rich shape.
    # YOLO / segmentation / Other do not use it.
    needs_classify_helper = (
        (is_image and not is_segmentation and not is_yolo)
        or is_text
        or is_timeseries
        or (not is_image and not is_text and not is_timeseries and not is_other)
    )
    classification_helper = _CLASSIFICATION_HELPER if needs_classify_helper else ""

    return (
        gradio_client_schema_patch + "\n"
        + framework_imports + "\n"
        + extra_imports + "\n"
        + "import gradio as gr\n"
        + "import uvicorn\n"
        + "from fastapi import FastAPI\n"
        + "from fastapi.middleware.cors import CORSMiddleware\n"
        + "from fastapi.responses import JSONResponse\n"
        + "from pydantic import BaseModel\n\n"
        + load_code + "\n\n\n"
        + predict_array_helper + "\n"
        + classification_helper
        + schema_class + "\n\n"
        + "def predict_ui(" + ui_input_param + "):\n"
        + textwrap.indent(ui_predict_body, "    ") + "\n\n"
        + "demo = gr.Interface(\n"
        + "    fn=predict_ui,\n"
        + "    inputs=" + ui_input_widget + ",\n"
        + "    outputs=gr.Textbox(label=\"Prediction\"),\n"
        + f'    title="{title}",\n'
        + ")\n\n"
        + "app = FastAPI()\n"
        + "app.add_middleware(\n"
        + "    CORSMiddleware,\n"
        + "    allow_origins=[\"*\"],\n"
        + "    allow_methods=[\"*\"],\n"
        + "    allow_headers=[\"*\"],\n"
        + ")\n\n\n"
        # Catch-all exception handler. The per-route try/except inside
        # predict_api() cannot catch failures that happen AFTER the route
        # function returns — most importantly, FastAPI's response
        # serialization step. When a returned dict contains a value the JSON
        # encoder can't handle (numpy scalars, NaN/Inf floats, bytes, etc.)
        # the exception bubbles out of the route and reaches the ASGI layer,
        # which produces the unhelpful {"detail": "Internal Server Error"}
        # body that test users keep seeing. This handler intercepts every
        # unhandled exception app-wide and returns the same rich JSON shape
        # the per-route handler uses, so the test UI always shows what
        # actually went wrong.
        + "from fastapi import Request as _DA_Request\n"
        + "from starlette.exceptions import HTTPException as _DA_StarletteHTTPException\n\n"
        + "@app.exception_handler(Exception)\n"
        + "async def _da_global_exception_handler(request: _DA_Request, exc: Exception):\n"
        + "    if isinstance(exc, _DA_StarletteHTTPException):\n"
        + "        return JSONResponse(status_code=exc.status_code, content={\"detail\": exc.detail})\n"
        + "    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)\n"
        + "    tb_lines = [str(line).rstrip() for line in tb_lines][-12:]\n"
        + "    return JSONResponse(\n"
        + "        status_code=500,\n"
        + "        content={\n"
        + "            \"error\": str(exc) or repr(exc),\n"
        + "            \"type\": type(exc).__name__,\n"
        + "            \"path\": str(request.url.path),\n"
        + "            \"traceback\": tb_lines,\n"
        + "            \"hint\": (\n"
        + "                \"This error escaped the per-route try/except — usually a \"\n"
        + "                \"response-serialization failure (e.g. numpy scalar, NaN/Inf, \"\n"
        + "                \"or bytes in the returned dict). Check the Space's runtime logs.\"\n"
        + "            ),\n"
        + "        },\n"
        + "    )\n\n\n"
        + "@app.post(\"/predict\", include_in_schema=False)\n"
        + "def predict_api(req: PredictRequest):\n"
        + textwrap.indent(api_predict_body, "    ") + "\n\n"
        + "@app.get(\"/health\", include_in_schema=False)\n"
        + "def health():\n"
        + "    return {\"status\": \"ok\"}\n\n\n"
        + "app = gr.mount_gradio_app(app, demo, path=\"/\")\n\n\n"
        + "if __name__ == \"__main__\":\n"
        + "    uvicorn.run(app, host=\"0.0.0.0\", port=7860)\n"
    )


def generate_requirements(
    framework: str,
    model_type: str = "",
    sklearn_version: str | None = None,
) -> str:
    HF_PINNED = [
        "gradio==5.16.0",
        "huggingface_hub>=0.26.0",
        "requests",
    ]
    if framework in ("sklearn", "joblib"):
        sklearn_pin = f"scikit-learn=={sklearn_version}" if sklearn_version else "scikit-learn"
        pkgs = [sklearn_pin, "numpy", "scipy", "joblib"]
    elif framework == "pytorch":
        pkgs = ["torch", "numpy"]
    elif framework in ("tensorflow", "keras"):
        pkgs = ["tensorflow-cpu", "h5py", "numpy"]
    elif framework == "onnx":
        pkgs = ["onnxruntime", "numpy"]
    elif framework == "yolo":
        # ultralytics handles YOLOv8+; yolov5 (community PyPI package) is the
        # runtime fallback when the .pt is an original YOLOv5 checkpoint that
        # ultralytics rejects with a TypeError. Both packages share the torch
        # ecosystem so install cost is mostly torch (single copy).
        # opencv-python (pulled by both) imports libxcb at runtime and crashes
        # in headless containers — we install opencv-python-headless here and
        # force-reinstall it in the Dockerfile to overwrite cv2/.
        pkgs = [
            "ultralytics",
            "yolov5",
            "opencv-python-headless",
            "Pillow",
            "numpy",
        ]
    else:
        pkgs = ["numpy"]

    if (model_type or "") in IMAGE_MODEL_TYPES:
        pkgs.append("Pillow")

    # Dedupe while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for pkg in HF_PINNED + pkgs:
        if pkg not in seen:
            seen.add(pkg)
            deduped.append(pkg)
    return "\n".join(deduped)


def generate_dockerfile(framework: str = "") -> str:
    """Dockerfile for sdk="docker" HF Spaces — we control the full environment.

    For YOLO deployments, ultralytics hard-requires opencv-python (which needs
    libxcb / libGL — not present in slim containers). After the main install
    we force-reinstall opencv-python-headless with --no-deps so it overwrites
    cv2/ in site-packages. This is independent of pip resolver order.
    """
    base = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
"""
    if framework == "yolo":
        base += (
            "RUN pip install --no-cache-dir --force-reinstall --no-deps "
            "opencv-python-headless\n"
        )
    base += """COPY . .
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""
    return base


def generate_readme(space_name: str, framework: str) -> str:
    """
    HF Spaces config using sdk="docker" so we fully control the runtime.
    This avoids HF force-injecting an incompatible gradio version that
    breaks on `from huggingface_hub import HfFolder`.
    """
    return f"""---
title: {space_name}
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Auto-generated deployment for a {framework} model.
"""


def deploy_to_huggingface(
    model_path: str,
    framework: str,
    model_filename: str,
    hf_token: str,
    preferred_space_name: str | None = None,
    progress_callback=None,
    sklearn_version: str | None = None,
    model_type: str = "Tabular/Regression",
    input_spec: dict | None = None,
) -> dict:

    def update(step: str, msg: str):
        if progress_callback:
            progress_callback(step, msg)

    # YOLO is image-only Object Detection regardless of what the caller asked for.
    if framework == "yolo":
        model_type = "Object Detection"

    try:
        # 1. Validate token + get username
        update("validating", "Validating HuggingFace token...")
        api = HfApi(token=hf_token)
        username = api.whoami()["name"]

        # 2. Resolve space name
        update("naming", "Checking space name availability...")
        base_name = preferred_space_name or generate_space_name(model_filename)
        final_name = resolve_space_name(api, username, base_name)
        repo_id = f"{username}/{final_name}"

        # 3. Create Space
        update("creating_space", f"Creating Space: {repo_id}...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
            token=hf_token,
        )

        # 4. Generate files
        update("generating_files", "Generating inference server...")
        app_content = generate_gradio_app(
            framework,
            model_filename,
            model_type,
            input_spec=input_spec,
        )
        req_content = generate_requirements(
            framework,
            model_type=model_type,
            sklearn_version=sklearn_version,
        )
        readme_content = generate_readme(final_name, framework)
        dockerfile_content = generate_dockerfile(framework=framework)

        # 5. Upload all files in a single atomic commit. Separate upload_file
        #    calls each create their own commit and HF kicks off a build on
        #    every commit — that races the (large) model upload and makes the
        #    container start before the model file is actually on the Hub.
        update("uploading", "Uploading files to HuggingFace...")
        operations = [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_content.encode()),
            CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=dockerfile_content.encode()),
            CommitOperationAdd(path_in_repo="requirements.txt", path_or_fileobj=req_content.encode()),
            CommitOperationAdd(path_in_repo="app.py", path_or_fileobj=app_content.encode()),
            CommitOperationAdd(path_in_repo=model_filename, path_or_fileobj=model_path),
        ]
        api.create_commit(
            repo_id=repo_id,
            repo_type="space",
            operations=operations,
            commit_message="Initial deployment",
        )

        # 5b. Verify every file (especially the model) is confirmed present
        #     on the Hub before allowing the build poll to proceed.
        update("verifying", "Confirming all files uploaded...")
        expected = {"README.md", "Dockerfile", "requirements.txt", "app.py", model_filename}
        verify_deadline = time.time() + 60
        present: set[str] = set()
        while time.time() < verify_deadline:
            present = set(api.list_repo_files(repo_id=repo_id, repo_type="space"))
            if expected.issubset(present):
                break
            time.sleep(2)
        else:
            return {
                "status": "failed",
                "error": (
                    "Upload verification failed: missing files in repo: "
                    f"{sorted(expected - present)}"
                ),
            }

        # 6. Poll until RUNNING. YOLO pulls ultralytics + yolov5 + torch
        #    (~3GB) so it routinely needs 5-10 min on a cold HF builder; the
        #    default 5 min is too tight. Other frameworks keep the original
        #    budget. NB: the build keeps running on HF even after we stop
        #    polling — the timeout governs only how long *we* wait.
        timeout = 900 if framework == "yolo" else 300
        update("building", "HuggingFace is building your Space...")
        start = time.time()
        while time.time() - start < timeout:
            runtime = api.get_space_runtime(repo_id, token=hf_token)
            if runtime.stage == SpaceStage.RUNNING:
                break
            if runtime.stage in (SpaceStage.BUILD_ERROR, SpaceStage.CONFIG_ERROR):
                return {
                    "status": "failed",
                    "error": f"HuggingFace build failed: {runtime.stage}",
                }
            time.sleep(10)
        else:
            return {
                "status": "failed",
                "error": (
                    f"Timeout: Space took over {timeout // 60} minutes. "
                    f"Check manually: https://huggingface.co/spaces/{repo_id}"
                ),
            }

        # 7. Return success
        space_url = f"https://huggingface.co/spaces/{repo_id}"
        api_url = f"https://{username}-{final_name}.hf.space"
        update("live", "Deployment successful!")

        return {
            "status": "success",
            "space_url": space_url,
            "api_url": api_url,
            "space_name": final_name,
            "repo_id": repo_id,
            "framework": framework,
            "model_type": model_type,
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}
