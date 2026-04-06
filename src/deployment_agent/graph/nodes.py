from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from deployment_agent import config
from deployment_agent.detection import detect_with_llm, inspect_model_file
from deployment_agent.docker_ops import (
    docker_build,
    docker_rm_force,
    docker_run_detached,
    heal_with_llm,
    infer_python_tag,
    pick_free_port,
    short_id,
)
from deployment_agent.generators.bundle import write_deployment_bundle
from deployment_agent.graph.state import DeployState

logger = logging.getLogger(__name__)


def node_validate(state: DeployState) -> dict[str, Any]:
    log = list(state.get("decision_log", []))
    mp = Path(state["model_path"]).resolve()
    if not mp.is_file():
        return {"error": f"Model file not found: {mp}", "decision_log": log}
    rp = state.get("requirements_path")
    user_req: str | None = None
    if rp:
        p = Path(rp).resolve()
        if not p.is_file():
            return {"error": f"requirements.txt not found: {p}", "decision_log": log}
        user_req = p.read_text(encoding="utf-8")
        log.append(f"Loaded optional requirements from {p.name}.")
    else:
        log.append("No requirements path provided; inference dependencies will be auto-selected.")
    log.append(f"Validated model path: {mp.name}")
    return {"error": None, "user_requirements": user_req, "decision_log": log}


def node_detect(state: DeployState) -> dict[str, Any]:
    log = list(state.get("decision_log", []))
    mp = Path(state["model_path"]).resolve()
    try:
        meta = inspect_model_file(mp)
        log.append(
            f"Detected framework={meta['framework']} ({meta.get('detail', '')}). "
            f"Estimator: {meta.get('estimator_class')}."
        )
    except Exception as e:
        log.append(f"Local inspection failed ({e}); consulting LLM for classification.")
        try:
            meta = detect_with_llm(mp, str(e))
            log.append(f"LLM classification: framework={meta.get('framework')}. {meta.get('detail', '')}")
        except Exception as llm_e:
            return {"error": f"Detection failed: {llm_e}", "decision_log": log}

    if meta.get("framework") == "unknown":
        return {
            "error": "Could not determine a supported framework for this artifact.",
            "detection": meta,
            "decision_log": log,
        }
    return {"detection": meta, "error": None, "decision_log": log}


def node_prepare_build(state: DeployState) -> dict[str, Any]:
    log = list(state.get("decision_log", []))
    bid = short_id()
    build_dir = (config.BUILD_ROOT / bid).resolve()
    user_req = state.get("user_requirements")
    framework = (state.get("detection") or {}).get("framework", "unknown")
    task_type = (state.get("detection") or {}).get("task_type", "ml")
    tag = infer_python_tag(user_req) or ("3.11" if task_type == "dl" else config.DEFAULT_CONTAINER_PYTHON)
    log.append(f"Build id {bid}; staging directory {build_dir}.")
    log.append(f"Selected CPython {tag} for the container (user hint or default).")
    if task_type == "dl":
        log.append(
            "DL build notice: first build may take 8-10 minutes (~500MB downloads); subsequent builds are much faster due to Docker layer cache."
        )
    return {
        "build_id": bid,
        "build_dir": str(build_dir),
        "python_tag": tag,
        "task_type": task_type,
        "framework": framework,
        "heal_attempts": int(state.get("heal_attempts", 0)),
        "decision_log": log,
    }


def node_write_bundle(state: DeployState) -> dict[str, Any]:
    log = list(state.get("decision_log", []))
    build_dir = Path(state["build_dir"])
    meta = state["detection"]
    _, decisions = write_deployment_bundle(
        build_dir,
        Path(state["model_path"]).resolve(),
        meta,
        state.get("user_requirements"),
        state.get("python_tag"),
    )
    log.extend(decisions)
    return {"decision_log": log, "error": None}


def node_docker_build(state: DeployState) -> dict[str, Any]:
    log = list(state.get("decision_log", []))
    build_dir = Path(state["build_dir"])
    image_tag = f"deploy_agent:{state['build_id']}"
    timeout_sec = 900 if state.get("task_type") == "dl" else 300
    ok, blog = docker_build(build_dir, image_tag, timeout_sec=timeout_sec)
    if ok:
        log.append(f"Docker image built: {image_tag}")
        return {"image_tag": image_tag, "last_build_log": blog, "decision_log": log, "error": None}
    log.append("Docker build failed; captured logs for healing or abort.")
    return {"image_tag": image_tag, "last_build_log": blog, "decision_log": log, "error": "docker_build_failed"}


def node_heal(state: DeployState) -> dict[str, Any]:
    log = list(state.get("decision_log", []))
    build_dir = Path(state["build_dir"])
    attempts = int(state.get("heal_attempts", 0)) + 1
    df = (build_dir / "Dockerfile").read_text(encoding="utf-8")
    req = (build_dir / "requirements.txt").read_text(encoding="utf-8")
    try:
        new_req, new_df, expl = heal_with_llm(df, req, state.get("last_build_log", ""))
    except Exception as e:
        log.append(f"Heal attempt {attempts} failed (LLM): {e}")
        return {
            "heal_attempts": attempts,
            "decision_log": log,
            "error": "heal_failed",
            "last_build_log": state.get("last_build_log", ""),
        }
    (build_dir / "requirements.txt").write_text(new_req, encoding="utf-8")
    (build_dir / "Dockerfile").write_text(new_df, encoding="utf-8")
    log.append(f"Heal attempt {attempts}: {expl}")
    return {"heal_attempts": attempts, "decision_log": log, "error": None}


def node_run(state: DeployState) -> dict[str, Any]:
    log = list(state.get("decision_log", []))
    port = pick_free_port()
    name = f"deploy_agent_{state['build_id']}"
    docker_rm_force(name)
    meta = state.get("detection") or {}
    model_file = Path(state["model_path"]).name
    run_env = {
        "DEPLOY_FRAMEWORK": str(meta.get("framework", "unknown")),
        "DEPLOY_MODEL_NAME": model_file,
    }
    shm = "512m" if state.get("task_type") == "dl" else None
    ok, rlog = docker_run_detached(state["image_tag"], port, name, env=run_env, shm_size=shm)
    if not ok:
        log.append(f"docker run failed:\n{rlog}")
        return {"error": "docker_run_failed", "decision_log": log}
    log.append(f"Started container {name} on host port {port}.")
    return {
        "container_name": name,
        "host_port": port,
        "api_url": f"http://127.0.0.1:{port}",
        "decision_log": log,
        "error": None,
    }


def node_health(state: DeployState) -> dict[str, Any]:
    import time

    import requests

    log = list(state.get("decision_log", []))
    url = state.get("api_url")
    if not url:
        return {"error": "missing_api_url", "decision_log": log}
    is_dl = state.get("task_type") == "dl"
    timeout = 120 if is_dl else 60
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                payload = {}
                try:
                    payload = r.json()
                except Exception:
                    payload = {}
                status = payload.get("status")
                if status == "ok":
                    log.append(f"Health check OK: GET {url}/health -> 200")
                    return {"error": None, "decision_log": log}
                if status in {"load_failed", "state_dict_only"}:
                    last_err = payload.get("error") or payload.get("message") or status
                    break
                if status == "loading":
                    last_err = "model still loading"
                    time.sleep(1.5)
                    continue
                last_err = f"health payload status={status!r}"
            last_err = f"status {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(1.5)
    name = state.get("container_name")
    if name:
        docker_rm_force(name)
        log.append(f"Health check failed ({last_err}); removed container {name} (rollback).")
    return {"error": f"health_failed: {last_err}", "decision_log": log, "api_url": ""}
