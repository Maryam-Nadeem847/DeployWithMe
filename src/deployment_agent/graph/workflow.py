from __future__ import annotations

from langgraph.graph import END, StateGraph

from deployment_agent.graph.nodes import (
    node_detect,
    node_docker_build,
    node_health,
    node_heal,
    node_prepare_build,
    node_run,
    node_validate,
    node_write_bundle,
)
from deployment_agent.graph.state import DeployState


def route_after_validate(state: DeployState) -> str:
    return "end" if state.get("error") else "detect"


def route_after_detect(state: DeployState) -> str:
    return "end" if state.get("error") else "prepare"


def route_after_build(state: DeployState) -> str:
    if not state.get("error"):
        return "run"
    if state.get("error") == "docker_build_failed":
        if int(state.get("heal_attempts", 0)) < 2:
            return "heal"
    return "end"


def route_after_heal(state: DeployState) -> str:
    return "end" if state.get("error") == "heal_failed" else "build"


def route_after_run(state: DeployState) -> str:
    return "end" if state.get("error") else "health"


def route_after_health(state: DeployState) -> str:
    return "end"


def build_graph():
    g = StateGraph(DeployState)
    g.add_node("validate", node_validate)
    g.add_node("detect", node_detect)
    g.add_node("prepare", node_prepare_build)
    g.add_node("write", node_write_bundle)
    g.add_node("build", node_docker_build)
    g.add_node("heal", node_heal)
    g.add_node("run", node_run)
    g.add_node("health", node_health)

    g.set_entry_point("validate")
    g.add_conditional_edges("validate", route_after_validate, {"detect": "detect", "end": END})
    g.add_conditional_edges("detect", route_after_detect, {"prepare": "prepare", "end": END})
    g.add_edge("prepare", "write")
    g.add_edge("write", "build")
    g.add_conditional_edges(
        "build",
        route_after_build,
        {"run": "run", "heal": "heal", "end": END},
    )
    g.add_conditional_edges("heal", route_after_heal, {"build": "build", "end": END})
    g.add_conditional_edges("run", route_after_run, {"health": "health", "end": END})
    g.add_conditional_edges("health", route_after_health, {"end": END})
    return g.compile()


def run_deploy(model_path: str, requirements_path: str | None = None) -> DeployState:
    app = build_graph()
    init: DeployState = {
        "model_path": model_path,
        "requirements_path": requirements_path,
        "decision_log": [],
    }
    return app.invoke(init)


def run_deploy_until_build(model_path: str, requirements_path: str | None = None) -> DeployState:
    """
    Run validate → detect → prepare → write → docker build (with heal loop).
    Stops before container run (for human-in-the-loop API server).
    """
    state: DeployState = {
        "model_path": model_path,
        "requirements_path": requirements_path,
        "decision_log": [],
    }
    for node_fn in (node_validate, node_detect, node_prepare_build, node_write_bundle):
        state.update(node_fn(state))
        if state.get("error"):
            return state

    while True:
        state.update(node_docker_build(state))
        if not state.get("error"):
            break
        if state.get("error") == "docker_build_failed":
            if int(state.get("heal_attempts", 0)) < 2:
                state.update(node_heal(state))
                if state.get("error") == "heal_failed":
                    return state
            else:
                return state
        else:
            return state
    return state


def run_deploy_run_and_health(state: DeployState) -> DeployState:
    """Continue after image is built: run container + health check."""
    state.update(node_run(state))
    if state.get("error"):
        return state
    state.update(node_health(state))
    return state
