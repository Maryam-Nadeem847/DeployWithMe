from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "trained_models" / "ml_new"
OUT.mkdir(parents=True, exist_ok=True)


def train_digits_classifier() -> Path:
    X, y = load_digits(return_X_y=True)  # 8x8 images flattened -> 64 features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Different from your earlier RF: multinomial logistic regression on digits
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    path = OUT / "digits_logreg.joblib"
    joblib.dump(model, path)
    print(f"[OK] saved {path}  accuracy={acc:.4f}  n_features={X.shape[1]}")
    return path


def train_digits_hgb_classifier() -> Path:
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modern tree boosting in sklearn (different from earlier GB/RF config)
    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    path = OUT / "digits_histgbc.joblib"
    joblib.dump(model, path)
    print(f"[OK] saved {path}  accuracy={acc:.4f}  n_features={X.shape[1]}")
    return path


def train_california_regressors() -> tuple[Path, Path]:
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # (1) HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.06, random_state=42)
    hgb.fit(X_train, y_train)
    pred1 = hgb.predict(X_test)
    rmse1 = np.sqrt(mean_squared_error(y_test, pred1))
    p1 = OUT / "california_histgbr.joblib"
    joblib.dump(hgb, p1)
    print(f"[OK] saved {p1}  rmse={rmse1:.4f}  n_features={X.shape[1]}")

    # (2) Ridge baseline (different from earlier synthetic regression)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    ridge.fit(X_train, y_train)
    pred2 = ridge.predict(X_test)
    rmse2 = np.sqrt(mean_squared_error(y_test, pred2))
    p2 = OUT / "california_ridge.joblib"
    joblib.dump(ridge, p2)
    print(f"[OK] saved {p2}  rmse={rmse2:.4f}  n_features={X.shape[1]}")

    return p1, p2


def main() -> None:
    train_digits_classifier()
    train_digits_hgb_classifier()
    train_california_regressors()


if __name__ == "__main__":
    main()