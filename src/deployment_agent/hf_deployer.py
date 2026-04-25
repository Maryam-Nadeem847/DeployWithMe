"""
HuggingFace Spaces deployer — uploads a model + Gradio app to HF Spaces.

Completely independent of the Docker pipeline.
"""
from __future__ import annotations

import random
import re
import string
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, SpaceStage, create_repo


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


def generate_gradio_app(framework: str, model_filename: str) -> str:
    if framework in ("sklearn", "joblib"):
        load_code = f'model = joblib.load("{model_filename}")'
        imports = "import joblib\nimport numpy as np"
        predict_code = (
            "    values = [float(x.strip()) for x in input_text.split(',')]\n"
            "    X = np.array([values])\n"
            "    prediction = model.predict(X)\n"
            "    return str(prediction.tolist())"
        )
    elif framework == "pytorch":
        load_code = f'model = torch.jit.load("{model_filename}", map_location="cpu")\nmodel.eval()'
        imports = "import torch\nimport numpy as np"
        predict_code = (
            "    values = [float(x.strip()) for x in input_text.split(',')]\n"
            "    x = torch.tensor([values], dtype=torch.float32)\n"
            "    with torch.no_grad():\n"
            "        output = model(x)\n"
            "    return str(output.detach().cpu().numpy().tolist())"
        )
    elif framework in ("tensorflow", "keras"):
        load_code = f'model = keras.models.load_model("{model_filename}", compile=False)'
        imports = "import numpy as np\nimport keras"
        predict_code = (
            "    values = [float(x.strip()) for x in input_text.split(',')]\n"
            "    X = np.array([values], dtype=np.float32)\n"
            "    prediction = model.predict(X, verbose=0)\n"
            "    return str(prediction.tolist())"
        )
    elif framework == "onnx":
        load_code = (
            "import onnxruntime as ort\n"
            f'session = ort.InferenceSession("{model_filename}", providers=["CPUExecutionProvider"])\n'
            'input_name = session.get_inputs()[0].name'
        )
        imports = "import numpy as np"
        predict_code = (
            "    values = [float(x.strip()) for x in input_text.split(',')]\n"
            "    X = np.array([values], dtype=np.float32)\n"
            "    outputs = session.run(None, {input_name: X})\n"
            "    return str([o.tolist() for o in outputs])"
        )
    else:
        # Fallback: assume joblib-loadable
        load_code = f'model = joblib.load("{model_filename}")'
        imports = "import joblib\nimport numpy as np"
        predict_code = (
            "    values = [float(x.strip()) for x in input_text.split(',')]\n"
            "    X = np.array([values])\n"
            "    prediction = model.predict(X)\n"
            "    return str(prediction.tolist())"
        )

    return f"""{imports}
import gradio as gr

{load_code}

def predict(input_text):
{predict_code}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="1.2, 3.4, 5.6", label="Features (comma-separated)"),
    outputs=gr.Textbox(label="Prediction"),
    title="{framework} Model Inference",
)
demo.launch(server_name="0.0.0.0", server_port=7860)
"""


def generate_requirements(framework: str, sklearn_version: str | None = None) -> str:
    HF_PINNED = [
        "gradio==5.9.1",
        "huggingface_hub>=0.26.0",
    ]
    if framework in ("sklearn", "joblib"):
        sklearn_pin = f"scikit-learn=={sklearn_version}" if sklearn_version else "scikit-learn"
        pkgs = [sklearn_pin, "numpy", "scipy", "joblib"]
    elif framework == "pytorch":
        pkgs = ["torch", "numpy"]
    elif framework in ("tensorflow", "keras"):
        pkgs = ["tensorflow-cpu"]
    elif framework == "onnx":
        pkgs = ["onnxruntime", "numpy"]
    else:
        pkgs = ["numpy"]
    return "\n".join(HF_PINNED + pkgs)


def generate_dockerfile() -> str:
    """Dockerfile for sdk="docker" HF Spaces — we control the full environment."""
    return """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
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
        app_content = generate_gradio_app(framework, model_filename)
        req_content = generate_requirements(framework, sklearn_version=sklearn_version)
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
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}
