from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "trained_models" / "manifest.json"


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}. Run train_classical_ml_zoo.py first.")

    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print("PowerShell deploy commands:\n")
    for i, row in enumerate(rows, start=1):
        model = row["file_path"]
        print(f"# {i}. {row['task']} | {row['model_name']} | {row['metric_name']}={row['metric_value']:.4f}")
        print(f"python -m deployment_agent \"{model}\" --json")
        print()


if __name__ == "__main__":
    main()
