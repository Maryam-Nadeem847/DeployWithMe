from __future__ import annotations

from typing import Any, TypedDict


class DeployState(TypedDict, total=False):
    model_path: str
    requirements_path: str | None
    user_requirements: str | None

    build_id: str
    build_dir: str
    python_tag: str
    task_type: str
    framework: str

    detection: dict[str, Any]
    image_tag: str
    container_name: str
    host_port: int
    api_url: str

    heal_attempts: int
    last_build_log: str

    error: str | None
    decision_log: list[str]
