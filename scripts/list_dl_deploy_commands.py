from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DL_DIR = ROOT / "trained_models" / "dl"


def main() -> None:
    if not DL_DIR.exists():
        raise SystemExit(f"{DL_DIR} not found. Run train_dl_test_models.py first.")
    files = sorted([p for p in DL_DIR.iterdir() if p.suffix.lower() in {".pt", ".pth", ".onnx", ".h5"}])
    if not files:
        raise SystemExit("No DL model files found.")
    print("PowerShell DL deploy commands:\n")
    for f in files:
        print(f"# {f.name}")
        print(f"python -m deployment_agent \"{f}\" --json")
        print()


if __name__ == "__main__":
    main()
