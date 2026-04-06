from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from deployment_agent.graph.workflow import run_deploy


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Autonomous ML model deployment (Phase 1: classical ML).")
    p.add_argument("model", type=str, help="Path to .pkl / .joblib model file")
    p.add_argument(
        "--requirements",
        "-r",
        type=str,
        default=None,
        help="Optional requirements.txt from training environment",
    )
    p.add_argument("--json", action="store_true", help="Print machine-readable result JSON on stdout")
    args = p.parse_args(argv)

    model_path = str(Path(args.model).expanduser().resolve())
    req_path = str(Path(args.requirements).expanduser().resolve()) if args.requirements else None

    result = run_deploy(model_path, req_path)

    if args.json:
        out = {
            "api_url": result.get("api_url"),
            "error": result.get("error"),
            "image_tag": result.get("image_tag"),
            "container_name": result.get("container_name"),
            "host_port": result.get("host_port"),
            "detection": result.get("detection"),
            "decision_log": result.get("decision_log", []),
        }
        print(json.dumps(out, indent=2))
        return 0 if not result.get("error") else 1

    for line in result.get("decision_log", []):
        print(line)
    if result.get("error"):
        print(f"\nFAILED: {result['error']}", file=sys.stderr)
        if result.get("last_build_log"):
            print("\n--- Last build log ---\n", result["last_build_log"][:8000], file=sys.stderr)
        return 1
    print(f"\nLive API: {result.get('api_url')}")
    print(f"Health:   {result.get('api_url')}/health")
    print(f"Predict:  POST {result.get('api_url')}/predict  JSON {{\"features\": [...]}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
