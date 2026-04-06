# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Autonomous ML/DL model deployment agent. Takes a serialized model file, detects its framework, generates a FastAPI inference server + Dockerfile, builds a Docker image (with LLM-powered self-healing on build failures), and runs it as a container with `/health` and `/predict` endpoints.

Supported model formats:
- **ML:** `.pkl`, `.pickle`, `.joblib`, `.sav` (sklearn, xgboost, lightgbm, catboost)
- **DL:** `.pt`, `.pth` (PyTorch), `.onnx`, `.h5`/`.keras` (TensorFlow/Keras)

## Commands

```bash
# Install (editable, from project root)
pip install -e .

# Set PYTHONPATH (required for scripts, not needed if installed via pip)
export PYTHONPATH="$(pwd)/src"

# Deploy a model (CLI)
python -m deployment_agent path/to/model.joblib --json

# Run the API control server (frontend backend)
uvicorn deployment_agent.api_server:app --host 0.0.0.0 --port 8080 --reload

# Frontend dev server (React + Vite, port 5173)
cd frontend && npm install && npm run dev

# Train test model zoos
python scripts/train_classical_ml_zoo.py
python scripts/train_dl_test_models.py
```

## Architecture

### LangGraph Deployment Pipeline (`src/deployment_agent/graph/`)

The core is a LangGraph `StateGraph` with this node flow:

```
validate → detect → prepare → write → build ⇄ heal (up to 2 retries) → run → health
```

- **State** (`graph/state.py`): `DeployState` TypedDict flows through all nodes
- **Nodes** (`graph/nodes.py`): Each node is a pure function `(DeployState) -> dict` that returns partial state updates
- **Workflow** (`graph/workflow.py`): Wires nodes with conditional routing. Exposes `run_deploy()` for CLI and `run_deploy_until_build()` + `run_deploy_run_and_health()` for the API server's human-in-the-loop confirmation flow

### Key Modules

- **`detection.py`**: Inspects model files — extension-based detection for DL, `joblib.load()` + module introspection for ML. Falls back to LLM classification on failure.
- **`generators/bundle.py`**: Generates per-framework `main.py` (FastAPI app), `Dockerfile`, and `requirements.txt`. Uses framework-specific code templates (`build_main_pytorch`, `build_main_onnx`, `build_main_tensorflow`, `build_main_py` for classical ML).
- **`docker_ops.py`**: Docker build/run/cleanup commands via subprocess. Contains `heal_with_llm()` which asks an LLM to fix broken Dockerfiles/requirements from build logs.
- **`llm_client.py`**: Dual-provider LLM client — tries Gemini first, falls back to Groq on rate limits. All LLM calls go through `complete_json()` or `complete_text()`.
- **`config.py`**: Env vars via dotenv. Key settings: `GEMINI_API_KEY`, `GROQ_API_KEY`, `DEPLOY_AGENT_BUILD_ROOT`.
- **`api_server.py`**: FastAPI control server ("DeployWithMe") for the React frontend. Manages deploy jobs with threading, confirmation checkpoint, and container lifecycle. CORS allows `localhost:5173`.

### Frontend (`frontend/`)

React 18 + Vite + Tailwind CSS 4. Communicates with the control API at port 8080. Key flow: upload model → poll status → confirm deployment → view result / test predictions.

### Build Artifacts

Generated deployment bundles go to `.deploy_builds/<build_id>/` (configurable via `DEPLOY_AGENT_BUILD_ROOT`). Each contains `main.py`, `Dockerfile`, `requirements.txt`, and the model file. Docker images are tagged `deploy_agent:<build_id>`, containers named `deploy_agent_<build_id>`.

## Environment Variables

- `GEMINI_API_KEY` / `GROQ_API_KEY` — At least one required for LLM-powered healing and ambiguous detection
- `GEMINI_MODEL` — defaults to `models/gemini-2.5-flash`
- `GROQ_MODEL` — defaults to `llama3-8b-8192`
- `DEPLOY_AGENT_BUILD_ROOT` — where build artifacts are staged (default: `.deploy_builds/`)
- `DEPLOY_AGENT_CONTAINER_PYTHON` — Python version for container base image (default: `3.11`)

## Known Issues & Constraints

- **numpy + TensorFlow conflict**: numpy 2.x breaks TF. Always pin numpy 
  separately in its own RUN step before installing tensorflow in the Dockerfile.

- **PyTorch saving**: Models MUST be saved as TorchScript (`torch.jit.save`), 
  not `torch.save`. Regular saves embed Python class references that fail 
  inside Docker.

- **TensorFlow 2.16+**: Use standalone `keras` (`import keras`), never 
  `tf.keras` — it was removed in 2.16.

- **Windows subprocess encoding**: All subprocess calls must use 
  `encoding="utf-8", errors="replace"` and guard against None stdout.

- **Orphaned containers**: Always force-remove existing containers before 
  `docker run` to prevent "name already in use" errors.

- **Do not change**: API endpoint paths in api_server.py, port 8080 
  (control API), ports 8001-8999 (model APIs), or LangGraph node names.