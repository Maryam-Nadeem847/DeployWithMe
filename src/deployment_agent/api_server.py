"""
DeployWithMe — FastAPI control server for the deployment agent.

Exposes HTTP APIs so a React frontend can trigger deploys, poll status,
confirm before going live, list/stop containers, and proxy test predictions.

Run: uvicorn deployment_agent.api_server:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.formparsers import MultiPartParser

from deployment_agent.docker_ops import docker_rmi_force, docker_rm_force, run_cmd
from deployment_agent.hf_deployer import deploy_to_huggingface, generate_space_name

# Starlette caps each multipart part at 1 MB by default. Model files routinely
# run into hundreds of MB, so raise the cap before any routes parse uploads.
MultiPartParser.max_part_size = 1024 * 1024 * 1024  # 1 GB

logger = logging.getLogger(__name__)

app = FastAPI(title="DeployWithMe Control API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        # Sandboxed iframes used by the cloud test UI (srcDoc + sandbox without
        # allow-same-origin) report Origin: null. Whitelist that explicitly so
        # the test UI's fetch() to /api/test-cloud-predict succeeds.
        "null",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict[str, dict[str, Any]] = {}
cloud_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _safe_filename(name: str | None) -> str:
    if not name:
        return "model.bin"
    base = Path(name).name
    base = re.sub(r"[^\w.\-]", "_", base)
    if not base or base in {".", ".."}:
        return "model.bin"
    return base


def _strip_internal(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not str(k).startswith("_")}


def _docker_daemon_ok() -> bool:
    code, _, _ = run_cmd(["docker", "info"])
    return code == 0


def _infer_host_port_from_inspect(data: dict[str, Any]) -> int | None:
    ports = (data.get("NetworkSettings") or {}).get("Ports") or {}
    for key, bindings in ports.items():
        if "8000/tcp" in key and bindings:
            try:
                return int(bindings[0].get("HostPort", "0")) or None
            except (ValueError, TypeError, IndexError):
                pass
    # fallback: parse 8000/tcp
    b = ports.get("8000/tcp")
    if b and isinstance(b, list) and b:
        try:
            return int(b[0].get("HostPort", "0")) or None
        except (ValueError, TypeError):
            return None
    return None


def _env_list_to_dict(env_list: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in env_list or []:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok", "docker": "running" if _docker_daemon_ok() else "not_running"}


class ConfirmBody(BaseModel):
    confirmed: bool = Field(..., description="True to run container, False to cancel")


class TestPredictBody(BaseModel):
    features: list[float]


class TestCloudPredictBody(BaseModel):
    """Body for the cloud-predict proxy used by the auto-generated test UI."""

    api_url: str = Field(..., description="Base URL of the deployed HF Space, e.g. https://user-name.hf.space")
    payload: dict[str, Any] = Field(..., description="JSON body to forward verbatim to <api_url>/predict")


# HF Spaces endpoints follow https://<username>-<spacename>.hf.space; restrict
# the proxy to that host so this endpoint cannot be abused as an SSRF gadget
# against arbitrary URLs on the local network.
_HF_SPACE_API_URL_RE = re.compile(r"^https://[A-Za-z0-9._-]+\.hf\.space$")


# In-memory cache for generated test-UI HTML. Loading the iframe via this
# server's origin (instead of via srcDoc on the Vite dev server) makes the
# proxy at /api/test-cloud-predict a same-origin fetch from inside the
# iframe — that bypasses every browser's CORS rules entirely (no preflight,
# no null-origin handling), which is the only way to make this 100%
# reliable across browsers.
_TEST_UI_TTL_SECONDS = 3600
_test_ui_store: dict[str, dict[str, Any]] = {}
_test_ui_lock = threading.Lock()


def _purge_expired_test_ui_locked() -> None:
    now = time.time()
    expired = [k for k, v in _test_ui_store.items() if v["expires_at"] < now]
    for k in expired:
        _test_ui_store.pop(k, None)


class StoreTestUIBody(BaseModel):
    """Body for the test-UI cache endpoint."""

    html: str = Field(..., min_length=1, max_length=4_000_000)


def _deployment_worker(job_id: str, model_path: str, requirements_path: str | None) -> None:
    from deployment_agent.graph.workflow import run_deploy_run_and_health, run_deploy_until_build

    with _jobs_lock:
        job = jobs[job_id]

    try:
        with _jobs_lock:
            job["status"] = "running"
            job["progress"] = 10

        build_t0 = time.time()
        state = run_deploy_until_build(model_path, requirements_path)

        with _jobs_lock:
            job["decision_log"] = list(state.get("decision_log", []))
            det = state.get("detection") or {}
            job["framework"] = det.get("framework")
            job["model_name"] = Path(model_path).name

        if state.get("error"):
            with _jobs_lock:
                job["status"] = "failed"
                job["error"] = str(state.get("error"))
                job["progress"] = 100
                job["last_build_log"] = state.get("last_build_log")
            return

        build_sec = int(time.time() - build_t0)
        dockerfile_path = Path(state["build_dir"]) / "Dockerfile"
        preview = ""
        if dockerfile_path.is_file():
            lines = dockerfile_path.read_text(encoding="utf-8").splitlines()[:10]
            preview = "\n".join(lines)

        with _jobs_lock:
            job["confirmation_data"] = {
                "image_tag": state.get("image_tag"),
                "port": None,
                "framework": job["framework"],
                "model_name": job["model_name"],
                "build_duration_seconds": build_sec,
                "dockerfile_preview": preview,
                "memory_limit": "container default (Docker)",
            }
            job["status"] = "awaiting_confirmation"
            job["progress"] = 75
            job["_deploy_state"] = state

        ev: threading.Event = job["_confirm_event"]
        ev.wait(timeout=3600)

        with _jobs_lock:
            confirmed = job.get("_user_confirmed")

        if not confirmed:
            with _jobs_lock:
                job["status"] = "failed"
                job["error"] = "Deployment cancelled by user."
                job["progress"] = 100
            if state.get("image_tag"):
                docker_rmi_force(str(state["image_tag"]))
            return

        with _jobs_lock:
            job["status"] = "running"
            job["progress"] = 85

        final = run_deploy_run_and_health(state)

        with _jobs_lock:
            job["decision_log"] = list(final.get("decision_log", job["decision_log"]))
            if final.get("error"):
                job["status"] = "failed"
                job["error"] = str(final.get("error"))
                job["api_url"] = None
            else:
                job["status"] = "success"
                job["error"] = None
                job["api_url"] = final.get("api_url")
                if job.get("confirmation_data"):
                    port = final.get("host_port")
                    job["confirmation_data"] = dict(job["confirmation_data"])
                    job["confirmation_data"]["port"] = port
            job["progress"] = 100

    except Exception as e:
        logger.exception("Deployment worker failed: %s", e)
        with _jobs_lock:
            job = jobs.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = str(e)
                job["progress"] = 100


@app.post("/api/deploy")
async def deploy(
    model_file: UploadFile = File(...),
    requirements_file: UploadFile | None = File(None),
) -> dict[str, str]:
    if not _docker_daemon_ok():
        raise HTTPException(status_code=503, detail="Docker daemon is not reachable")

    job_id = uuid.uuid4().hex[:8]
    tmp = Path(tempfile.gettempdir()) / f"deploywithme_{job_id}"
    tmp.mkdir(parents=True, exist_ok=True)

    model_name = _safe_filename(model_file.filename)
    model_path = tmp / model_name
    content = await model_file.read()
    model_path.write_bytes(content)

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    if file_size_mb > 50:
        return JSONResponse({
            "route": "cloud",
            "reason": (
                f"Model is {file_size_mb:.1f}MB. "
                "Routing to HuggingFace Spaces for better performance."
            ),
            "suggested_space_name": generate_space_name(model_name),
            "file_size_mb": round(file_size_mb, 1),
        })

    req_path: str | None = None
    if requirements_file and requirements_file.filename:
        rq_name = _safe_filename(requirements_file.filename)
        rq = tmp / rq_name
        rq.write_bytes(await requirements_file.read())
        req_path = str(rq.resolve())

    with _jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "started",
            "progress": 0,
            "framework": None,
            "model_name": model_name,
            "decision_log": [],
            "confirmation_data": None,
            "api_url": None,
            "error": None,
            "_confirm_event": threading.Event(),
            "_user_confirmed": None,
            "_deploy_state": None,
            "_temp_dir": str(tmp),
            "last_build_log": None,
        }

    # Run worker in thread so FastAPI returns immediately
    t = threading.Thread(
        target=_deployment_worker,
        args=(job_id, str(model_path.resolve()), req_path),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return _strip_internal(job)


@app.post("/api/confirm/{job_id}")
def confirm(job_id: str, body: ConfirmBody) -> dict[str, str]:
    with _jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        if job.get("status") != "awaiting_confirmation":
            raise HTTPException(status_code=400, detail="Job is not awaiting confirmation")
        job["_user_confirmed"] = bool(body.confirmed)
        job["_confirm_event"].set()

    if body.confirmed:
        return {"status": "proceeding"}
    return {"status": "cancelled"}


@app.get("/api/deployments")
def list_deployments() -> list[dict[str, Any]]:
    code, so, _ = run_cmd(
        ["docker", "ps", "--filter", "name=deploy_agent", "--format", "{{.Names}}"]
    )
    if code != 0:
        return []

    names = [n.strip() for n in (so or "").splitlines() if n.strip()]
    result: list[dict[str, Any]] = []

    for name in names:
        c2, insp_raw, _ = run_cmd(["docker", "inspect", name])
        if c2 != 0 or not insp_raw:
            continue
        try:
            data = json.loads(insp_raw)[0]
        except (json.JSONDecodeError, IndexError):
            continue

        port = _infer_host_port_from_inspect(data)
        if not port:
            continue

        env = _env_list_to_dict((data.get("Config") or {}).get("Env") or [])
        framework = env.get("DEPLOY_FRAMEWORK", "unknown")
        model_name = env.get("DEPLOY_MODEL_NAME", name)

        api_url = f"http://127.0.0.1:{port}"
        running = (data.get("State") or {}).get("Running", False)

        result.append(
            {
                "container_name": name,
                "api_url": api_url,
                "docs_url": f"{api_url}/docs",
                "framework": framework,
                "model_name": model_name,
                "status": "running" if running else "stopped",
                "port": port,
            }
        )

    return result


@app.delete("/api/deployments/{container_name:path}")
def stop_deployment(container_name: str) -> dict[str, Any]:
    if not container_name.startswith("deploy_agent_"):
        raise HTTPException(status_code=400, detail="Invalid container name")
    docker_rm_force(container_name)
    return {"stopped": True, "container": container_name}


@app.post("/api/test-predict/{port}")
def test_predict(port: int, body: TestPredictBody) -> Any:
    if port < 1 or port > 65535:
        raise HTTPException(status_code=400, detail="Invalid port")
    url = f"http://127.0.0.1:{port}/predict"
    try:
        r = requests.post(url, json={"features": body.features}, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    try:
        return r.json()
    except Exception:
        return {"raw": r.text, "status_code": r.status_code}


@app.post("/api/test-cloud-predict")
def test_cloud_predict(body: TestCloudPredictBody) -> JSONResponse:
    """Same-origin proxy for the auto-generated cloud test UI.

    The Gemini-generated test interface lives inside a sandboxed iframe
    (srcDoc + ``sandbox="allow-scripts allow-forms"``) which gives it a
    *null* origin. Cross-origin fetches from a null origin to a deployed
    HF Space hit two practical brick walls: HF's edge proxy occasionally
    drops CORS preflight responses, and several browsers refuse to honour
    ``Access-Control-Allow-Origin: *`` for null-origin requests when the
    response also carries credentials-like cookies. Routing through this
    same-origin endpoint sidesteps both: the iframe talks to localhost
    (whose CORS we control), and we make a server-to-server call to HF
    where CORS does not apply.
    """
    api_url = body.api_url.strip().rstrip("/")
    if not _HF_SPACE_API_URL_RE.match(api_url):
        raise HTTPException(
            status_code=400,
            detail="api_url must be a Hugging Face Space URL (https://<owner>-<space>.hf.space).",
        )

    target = f"{api_url}/predict"
    try:
        # HF Spaces can stall the first request after a build for ~30s while
        # the container warms up; budget generously.
        r = requests.post(target, json=body.payload, timeout=180)
    except requests.RequestException as e:
        return JSONResponse(
            status_code=502,
            content={
                "error": f"Could not reach the deployed Space: {e}",
                "type": type(e).__name__,
            },
        )

    try:
        data = r.json()
    except ValueError:
        return JSONResponse(
            status_code=502 if r.ok else r.status_code,
            content={
                "error": "Space returned a non-JSON response.",
                "status_code": r.status_code,
                "body": r.text[:2000],
                "logs_url": f"{api_url}/logs",
            },
        )

    # When an exception bypasses the deployed app's per-route try/except —
    # e.g. response-serialization failures inside FastAPI itself — the
    # response body collapses to FastAPI's default {"detail":"Internal Server
    # Error"}, which gives the user nothing to act on. Detect that exact
    # shape and replace it with an actionable envelope pointing at the
    # Space's runtime logs.
    if (
        r.status_code >= 500
        and isinstance(data, dict)
        and set(data.keys()) == {"detail"}
        and str(data.get("detail", "")).strip().lower() == "internal server error"
    ):
        return JSONResponse(
            status_code=r.status_code,
            content={
                "error": "The deployed Space crashed without returning a traceback.",
                "type": "InternalServerError",
                "status_code": r.status_code,
                "hint": (
                    "This usually means the Space was deployed with an older "
                    "version of the agent that lacks the global exception "
                    "handler. Re-deploy the same model to pick up the latest "
                    "agent code, then test again. The new deployment will "
                    "return the actual Python traceback in this body."
                ),
                "logs_url": f"{api_url}/logs",
                "space_url": api_url,
            },
        )
    return JSONResponse(status_code=r.status_code, content=data)


@app.post("/api/test-ui/store")
def store_test_ui(body: StoreTestUIBody) -> dict[str, str]:
    """Cache an auto-generated test-UI HTML doc and return a render URL.

    The frontend points the iframe at the returned URL on this same server
    (port 8080) instead of inlining the HTML via ``srcDoc`` on the Vite dev
    server (port 5173). That makes the iframe and the prediction proxy
    same-origin, so the iframe's fetch() to /api/test-cloud-predict needs
    no CORS preflight at all and works in every browser. This is the
    permanent fix for the recurring "Failed to fetch" error users hit when
    testing cloud deployments.
    """
    test_id = uuid.uuid4().hex[:16]
    with _test_ui_lock:
        _purge_expired_test_ui_locked()
        _test_ui_store[test_id] = {
            "html": body.html,
            "expires_at": time.time() + _TEST_UI_TTL_SECONDS,
        }
    return {"test_id": test_id, "render_url": f"/api/test-ui/render/{test_id}"}


@app.get("/api/test-ui/render/{test_id}")
def render_test_ui(test_id: str) -> HTMLResponse:
    """Serve cached test-UI HTML so the iframe loads from this server's origin."""
    with _test_ui_lock:
        _purge_expired_test_ui_locked()
        entry = _test_ui_store.get(test_id)
    if not entry:
        return HTMLResponse(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Expired</title></head>"
            "<body style='font-family:system-ui,sans-serif;padding:32px;color:#334155'>"
            "<h2>Test UI not found or expired.</h2>"
            "<p>Click <strong>Test Deployment</strong> again to regenerate.</p>"
            "</body></html>",
            status_code=404,
        )
    # Serve as text/html with no caching so successive Test Deployment clicks
    # always show the latest generated UI.
    return HTMLResponse(
        entry["html"],
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


# ── HuggingFace Spaces cloud deployment endpoints ─────────────────────


class CloudConfirmBody(BaseModel):
    confirmed_space_name: str
    hf_token: str
    model_type: str | None = None
    model_type_description: str | None = None
    input_spec: dict[str, Any] | None = None


def _cloud_deploy_worker(job_id: str) -> None:
    with _jobs_lock:
        job = cloud_jobs[job_id]
        model_path = job["_model_path"]
        model_filename = job["model_filename"]
        framework = job["framework"]
        sklearn_version = job.get("sklearn_version")
        hf_token = job["_hf_token"]
        space_name = job["_confirmed_space_name"]
        model_type = job.get("model_type") or "Tabular/Regression"
        input_spec = job.get("input_spec")

    def on_progress(step: str, msg: str):
        with _jobs_lock:
            job["step"] = step
            job["message"] = msg

    result = deploy_to_huggingface(
        model_path=model_path,
        framework=framework,
        model_filename=model_filename,
        hf_token=hf_token,
        preferred_space_name=space_name,
        progress_callback=on_progress,
        sklearn_version=sklearn_version,
        model_type=model_type,
        input_spec=input_spec,
    )

    with _jobs_lock:
        job["result"] = result
        job["status"] = result.get("status", "failed")
        job["step"] = "done"
        job["message"] = result.get("error") or "Deployment complete."


@app.post("/api/deploy/cloud")
async def deploy_cloud(
    model_file: UploadFile = File(...),
    hf_token: str = Form(...),
    preferred_space_name: str | None = Form(None),
) -> dict[str, Any]:
    from deployment_agent.detection import extract_input_spec, inspect_model_file

    job_id = uuid.uuid4().hex[:8]
    tmp = Path(tempfile.gettempdir()) / f"deploywithme_cloud_{job_id}"
    tmp.mkdir(parents=True, exist_ok=True)

    model_name = _safe_filename(model_file.filename)
    model_path = tmp / model_name
    content = await model_file.read()
    model_path.write_bytes(content)

    detection = inspect_model_file(model_path)
    if detection.get("deployable", True) is False:
        # Clean up the uploaded file: no job will be created so nothing else owns the temp dir.
        shutil.rmtree(tmp, ignore_errors=True)
        reason = detection.get("reason") or detection.get("detail") or "Model is not deployable."
        raise HTTPException(status_code=400, detail=reason)
    framework = detection.get("framework", "unknown")
    sklearn_version = detection.get("sklearn_version")
    input_spec_auto = extract_input_spec(model_path, framework)
    detection["input_spec"] = input_spec_auto
    suggested = preferred_space_name or generate_space_name(model_name)

    with _jobs_lock:
        cloud_jobs[job_id] = {
            "job_id": job_id,
            "status": "awaiting_confirmation",
            "step": "created",
            "message": "Waiting for user confirmation.",
            "framework": framework,
            "model_filename": model_name,
            "suggested_space_name": suggested,
            "sklearn_version": sklearn_version,
            "input_spec_auto": input_spec_auto,
            "input_spec": None,
            "model_type": None,
            "model_type_description": None,
            "result": None,
            "_model_path": str(model_path.resolve()),
            "_hf_token": hf_token,
            "_confirmed_space_name": None,
            "_temp_dir": str(tmp),
        }

    return {
        "job_id": job_id,
        "suggested_space_name": suggested,
        "framework": framework,
        "input_spec_auto": input_spec_auto,
    }


@app.post("/api/deploy/cloud/confirm/{job_id}")
def confirm_cloud(job_id: str, body: CloudConfirmBody) -> dict[str, str]:
    with _jobs_lock:
        job = cloud_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        if job["status"] != "awaiting_confirmation":
            raise HTTPException(status_code=400, detail="Job is not awaiting confirmation")
        job["_confirmed_space_name"] = body.confirmed_space_name
        job["_hf_token"] = body.hf_token
        job["model_type"] = body.model_type
        job["model_type_description"] = body.model_type_description
        job["input_spec"] = body.input_spec or job.get("input_spec_auto")
        job["status"] = "in_progress"
        job["step"] = "starting"
        job["message"] = "Starting cloud deployment..."

    t = threading.Thread(target=_cloud_deploy_worker, args=(job_id,), daemon=True)
    t.start()

    return {"status": "started"}


@app.get("/api/status/cloud/{job_id}")
def get_cloud_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = cloud_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return {
            "step": job["step"],
            "message": job["message"],
            "status": job["status"],
            "result": job["result"],
        }
