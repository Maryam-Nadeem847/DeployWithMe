from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
try:
    import torch
except Exception:
    torch = None

import keras
from keras import layers

try:
    import onnx  # noqa: F401
except Exception:
    onnx = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "trained_models" / "dl"
OUT.mkdir(parents=True, exist_ok=True)


def prepare_data():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)


if torch is not None:
    # Keep class at module level to avoid local-object pickle issues.
    class IrisClassifierTorch(torch.nn.Module):
        def __init__(self, in_dim: int = 4, out_dim: int = 3):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, out_dim),
            )

        def forward(self, x):
            return self.net(x)
else:
    class IrisClassifierTorch:  # type: ignore[no-redef]
        pass


def train_torch(X_train, y_train):
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed in this Python environment. "
            "Install with: python -m pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu"
        )

    model = IrisClassifierTorch()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(200):
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
    model.eval()
    return model


def export_onnx(model, sample_input):
    if torch is None:
        raise RuntimeError("PyTorch is required for ONNX export.")
    if onnx is None:
        raise RuntimeError(
            "ONNX export requires the `onnx` package. Install with: python -m pip install onnx"
        )

    out_path = OUT / "iris_torch.onnx"
    torch.onnx.export(
        model,
        sample_input,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return out_path


def train_tf(X_train, y_train):
    model = keras.Sequential(
        [
            layers.Input(shape=(4,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    return model


def main() -> None:
    if torch is None:
        raise SystemExit(
            "Missing required DL package: torch. "
            "Install with: python -m pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu"
        )

    X_train, _, y_train, _ = prepare_data()

    # PyTorch full model
    torch_model = train_torch(X_train, y_train)
    pt_path = OUT / "iris_torch_full.pt"
    # Save as TorchScript so the container can load it without needing the original Python class.
    sample = torch.tensor(X_train[:1], dtype=torch.float32)
    scripted = torch.jit.trace(torch_model.eval(), sample)
    torch.jit.save(scripted, str(pt_path))
    print(f"[OK] saved {pt_path}")

    # PyTorch state dict sample (to test informative error path)
    pth_path = OUT / "iris_torch_state_dict.pth"
    torch.save(torch_model.state_dict(), pth_path)
    print(f"[OK] saved {pth_path}")

    # ONNX
    sample = torch.tensor(X_train[:1], dtype=torch.float32)
    try:
        onnx_path = export_onnx(torch_model, sample)
        print(f"[OK] saved {onnx_path}")
    except Exception as e:
        print(f"[SKIP] ONNX export: {e}")

    # TensorFlow H5
    tf_model = train_tf(X_train, y_train)
    h5_path = OUT / "iris_tf.h5"
    tf_model.save(h5_path)
    print(f"[OK] saved {h5_path}")


if __name__ == "__main__":
    main()
