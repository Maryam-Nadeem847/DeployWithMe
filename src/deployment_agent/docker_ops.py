from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def run_cmd(args: list[str], cwd: Path | None = None, timeout_sec: int | None = None) -> tuple[int, str, str]:
    logger.info("Running: %s", " ".join(args))
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as e:
        so = e.stdout if isinstance(e.stdout, str) else ""
        se = e.stderr if isinstance(e.stderr, str) else ""
        return 124, so or "", (se or "") + f"\nCommand timed out after {timeout_sec}s."


def docker_build(context_dir: Path, image_tag: str, timeout_sec: int = 300) -> tuple[bool, str]:
    code, so, se = run_cmd(
        ["docker", "build", "-t", image_tag, "."],
        cwd=context_dir,
        timeout_sec=timeout_sec,
    )
    log = f"STDOUT:\n{so}\nSTDERR:\n{se}"
    return code == 0, log


def docker_run_detached(
    image_tag: str,
    host_port: int,
    name: str,
    env: dict[str, str] | None = None,
    shm_size: str | None = None,
) -> tuple[bool, str]:
    args = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"{host_port}:8000",
    ]
    if shm_size:
        args.extend(["--shm-size", shm_size])
    if env:
        for k, v in env.items():
            args.extend(["-e", f"{k}={v}"])
    args.append(image_tag)
    code, so, se = run_cmd(args)
    log = f"STDOUT:\n{so}\nSTDERR:\n{se}"
    return code == 0, log


def docker_rm_force(name: str) -> None:
    run_cmd(["docker", "stop", name])
    run_cmd(["docker", "rm", "-f", name])


def docker_rmi_force(image_tag: str) -> None:
    run_cmd(["docker", "rmi", "-f", image_tag])


def infer_python_tag(user_requirements: str | None) -> str | None:
    if not user_requirements:
        return None
    m = re.search(r"(?i)#\s*python\s*[:=]\s*(\d+\.\d+)", user_requirements)
    if m:
        return m.group(1)
    m = re.search(r"(?i)cpython\s*(\d+\.\d+)", user_requirements)
    if m:
        return m.group(1)
    return detect_python_version_needed(user_requirements)


def detect_python_version_needed(requirements_content: str) -> str:
    import re

    for line in requirements_content.split("\n"):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        match = re.search(r"numpy==(\d+)\.", line)
        if match and int(match.group(1)) >= 2:
            return "3.11"
    return "3.10"


def heal_with_llm(
    dockerfile: str,
    requirements: str,
    build_log: str,
) -> tuple[str, str, str]:
    """Returns (new_requirements, new_dockerfile, explanation)."""
    from deployment_agent.llm_client import complete_json

    system = (
        "You fix Docker build failures for a minimal FastAPI ML inference image. "
        "Return JSON only with keys: requirements_txt (string), dockerfile (string), "
        "explanation (string, plain English, no markdown)."
    )
    user = json.dumps(
        {
            "dockerfile": dockerfile,
            "requirements_txt": requirements,
            "build_log_excerpt": build_log[-12000:],
        }
    )
    data = complete_json(system, user)
    req = data.get("requirements_txt") or requirements
    df = data.get("dockerfile") or dockerfile
    expl = data.get("explanation") or "Patched requirements/Dockerfile from build log."
    return req, df, expl


def short_id() -> str:
    return uuid.uuid4().hex[:10]
