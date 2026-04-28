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
            ),
            "text_invoke_api": 'session.run(None, {input_name: np.array([req.text])})[0]',
            "text_invoke_ui": 'session.run(None, {input_name: np.array([input_text])})[0]',
        }
    # fallback
    return {
        "imports": "import joblib\nimport numpy as np",
        "load_code": f'model = joblib.load("{model_filename}")',
        "predict_array_helper": (
            "def _predict_array(X):\n"
            "    return model.predict(X)\n"
        ),
        "text_invoke_api": "model.predict([req.text])",
        "text_invoke_ui": "model.predict([input_text])",
    }


def _segmentation_output_block() -> str:
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
            "mask_png_base64": mask_b64,
            "shape": list(mask.shape),
            "input_size": [height_used, width_used],
        }
        """)


def _build_image_bodies(spec: dict, is_segmentation: bool) -> tuple[str, str]:
    """Return (api_predict_body, ui_predict_body) for image model types, using
    the confirmed input spec to bake exact preprocessing values."""
    preprocess = _build_preprocess_lines(spec)

    if is_segmentation:
        output_block = _segmentation_output_block()
    else:
        output_block = textwrap.dedent("""\
            arr_pred = np.asarray(pred)
            return {"prediction": arr_pred.tolist(), "input_size": [height_used, width_used]}
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

    extra_imports = "from typing import List, Optional"
    if is_image:
        extra_imports += "\nimport base64\nimport io\nimport traceback\nfrom PIL import Image"

    if is_image:
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
            f"pred = {text_invoke_api}\n"
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
            "pred = _predict_array(X)\n"
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
            "pred = _predict_array(X)\n"
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

    return (
        framework_imports + "\n"
        + extra_imports + "\n"
        + "import gradio as gr\n"
        + "import uvicorn\n"
        + "from fastapi import FastAPI\n"
        + "from fastapi.middleware.cors import CORSMiddleware\n"
        + "from fastapi.responses import JSONResponse\n"
        + "from pydantic import BaseModel\n\n"
        + load_code + "\n\n\n"
        + predict_array_helper + "\n"
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
    else:
        pkgs = ["numpy"]

    if (model_type or "") in IMAGE_MODEL_TYPES:
        pkgs.append("Pillow")

    return "\n".join(HF_PINNED + pkgs)


def generate_dockerfile() -> str:
    """Dockerfile for sdk="docker" HF Spaces — we control the full environment."""
    return """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""


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
        dockerfile_content = generate_dockerfile()

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

        # 6. Poll until RUNNING (timeout 5 min)
        update("building", "HuggingFace is building your Space...")
        timeout = 300
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
                    "Timeout: Space took over 5 minutes. "
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
