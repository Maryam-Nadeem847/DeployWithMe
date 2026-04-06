from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras import layers

try:
    import onnx  # noqa: F401
except Exception:
    onnx = None

try:
    import torch
except Exception:
    torch = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "trained_models" / "dl_new"
OUT.mkdir(parents=True, exist_ok=True)


def prepare_digits():
    X, y = load_digits(return_X_y=True)  # X: (n, 64)
    X = X.astype(np.float32) / 16.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)


if torch is not None:
    class DigitsMLPTorch(torch.nn.Module):
        """Slightly heavier than previous: 64 -> 128 -> 64 -> 10."""

        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)


def train_torch_digits(X_train, y_train):
    if torch is None:
        raise RuntimeError(
            "PyTorch not installed. Install: python -m pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu"
        )

    model = DigitsMLPTorch()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    model.train()
    for _ in range(300):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    model.eval()
    return model


def export_torchscript(model, sample_x) -> Path:
    out = OUT / "digits_torchscript.pt"
    scripted = torch.jit.trace(model.eval(), sample_x)
    torch.jit.save(scripted, str(out))
    return out


def export_onnx(model, sample_x) -> Path:
    if onnx is None:
        raise RuntimeError("ONNX export requires `onnx`. Install: python -m pip install onnx")
    out = OUT / "digits_torch.onnx"
    torch.onnx.export(
        model,
        sample_x,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    return out


def train_keras_digits(X_train, y_train) -> "keras.Model":
    # Slightly heavier than previous: more layers + dropout.
    model = keras.Sequential(
        [
            layers.Input(shape=(64,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=80, batch_size=32, verbose=0)
    return model


def main() -> None:
    X_train, _, y_train, _ = prepare_digits()

    # PyTorch TorchScript + ONNX from same trained model
    if torch is not None:
        tmodel = train_torch_digits(X_train, y_train)
        sample = torch.tensor(X_train[:2], dtype=torch.float32)

        pt_path = export_torchscript(tmodel, sample)
        print(f"[OK] saved {pt_path}")

        try:
            onnx_path = export_onnx(tmodel, sample)
            print(f"[OK] saved {onnx_path}")
        except Exception as e:
            print(f"[SKIP] ONNX export: {e}")
    else:
        print("[SKIP] torch not installed; skipping TorchScript/ONNX artifacts.")

    # Keras H5 model (digits)
    kmodel = train_keras_digits(X_train, y_train)
    h5_path = OUT / "digits_keras.h5"
    kmodel.save(h5_path)
    print(f"[OK] saved {h5_path}")


if __name__ == "__main__":
    main()

