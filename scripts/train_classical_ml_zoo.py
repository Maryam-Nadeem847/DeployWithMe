from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_regression,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "trained_models"


@dataclass
class TrainedArtifact:
    task: str
    framework: str
    model_name: str
    file_path: str
    metric_name: str
    metric_value: float
    extra: dict


def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def build_classification_models() -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = [
        ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("SGDClassifier", make_pipeline(StandardScaler(), SGDClassifier(max_iter=2000, tol=1e-3))),
        ("SVC_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=250, random_state=42)),
        ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=250, random_state=42)),
        ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=42)),
        ("AdaBoostClassifier", AdaBoostClassifier(random_state=42)),
        ("GaussianNB", GaussianNB()),
        ("GaussianProcessClassifier", GaussianProcessClassifier()),
        ("MLPClassifier", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))),
    ]

    xgb = _safe_import("xgboost")
    if xgb is not None:
        models.append(
            (
                "XGBClassifier",
                xgb.XGBClassifier(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    random_state=42,
                ),
            )
        )

    lgb = _safe_import("lightgbm")
    if lgb is not None:
        models.append(
            (
                "LGBMClassifier",
                lgb.LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.08,
                    random_state=42,
                    verbosity=-1,
                ),
            )
        )

    cb = _safe_import("catboost")
    if cb is not None:
        models.append(
            (
                "CatBoostClassifier",
                cb.CatBoostClassifier(
                    iterations=250,
                    learning_rate=0.08,
                    depth=6,
                    verbose=False,
                    random_seed=42,
                ),
            )
        )

    return models


def build_regression_models() -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.01)),
        ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5)),
        ("SGDRegressor", make_pipeline(StandardScaler(), SGDRegressor(max_iter=2000, tol=1e-3, random_state=42))),
        ("SVR_rbf", make_pipeline(StandardScaler(), SVR(kernel="rbf"))),
        ("KNeighborsRegressor", KNeighborsRegressor(n_neighbors=5)),
        ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=42)),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=250, random_state=42)),
        ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=250, random_state=42)),
        ("GradientBoostingRegressor", GradientBoostingRegressor(random_state=42)),
        ("AdaBoostRegressor", AdaBoostRegressor(random_state=42)),
        ("GaussianProcessRegressor", GaussianProcessRegressor()),
        ("MLPRegressor", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42))),
    ]

    xgb = _safe_import("xgboost")
    if xgb is not None:
        models.append(
            (
                "XGBRegressor",
                xgb.XGBRegressor(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                ),
            )
        )

    lgb = _safe_import("lightgbm")
    if lgb is not None:
        models.append(
            (
                "LGBMRegressor",
                lgb.LGBMRegressor(
                    n_estimators=250,
                    learning_rate=0.08,
                    random_state=42,
                    verbosity=-1,
                ),
            )
        )

    cb = _safe_import("catboost")
    if cb is not None:
        models.append(
            (
                "CatBoostRegressor",
                cb.CatBoostRegressor(
                    iterations=250,
                    learning_rate=0.08,
                    depth=6,
                    verbose=False,
                    random_seed=42,
                ),
            )
        )

    return models


def _train_and_save(
    task: str,
    dataset_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    models: list[tuple[str, object]],
    score_fn: Callable,
    metric_name: str,
) -> list[TrainedArtifact]:
    rows: list[TrainedArtifact] = []
    target_dir = OUT_DIR / task / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models:
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            score = float(score_fn(y_test, pred))
            file_name = f"{model_name}.joblib"
            artifact_path = target_dir / file_name
            joblib.dump(model, artifact_path)
            framework = type(model).__module__.split(".")[0]
            rows.append(
                TrainedArtifact(
                    task=task,
                    framework=framework,
                    model_name=model_name,
                    file_path=str(artifact_path),
                    metric_name=metric_name,
                    metric_value=score,
                    extra={"dataset": dataset_name},
                )
            )
            print(f"[OK] {task}/{dataset_name}/{model_name} -> {metric_name}={score:.4f}")
        except Exception as e:
            print(f"[SKIP] {task}/{dataset_name}/{model_name}: {e}")
    return rows


def main() -> None:
    from sklearn.model_selection import train_test_split

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.json"

    artifacts: list[TrainedArtifact] = []

    cls_sets = [
        ("iris", *load_iris(return_X_y=True)),
        ("wine", *load_wine(return_X_y=True)),
        ("breast_cancer", *load_breast_cancer(return_X_y=True)),
    ]
    cls_models = build_classification_models()
    for ds_name, X, y in cls_sets:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        artifacts.extend(
            _train_and_save(
                task="classification",
                dataset_name=ds_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=cls_models,
                score_fn=accuracy_score,
                metric_name="accuracy",
            )
        )

    # One synthetic regression set keeps runtime acceptable and feature count stable
    Xr, yr = make_regression(
        n_samples=3000,
        n_features=20,
        n_informative=15,
        noise=12.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg_models = build_regression_models()
    artifacts.extend(
        _train_and_save(
            task="regression",
            dataset_name="synthetic_regression",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=reg_models,
            score_fn=lambda yt, yp: mean_squared_error(yt, yp, squared=False),
            metric_name="rmse",
        )
    )

    payload = [a.__dict__ for a in artifacts]
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {len(artifacts)} model artifacts.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
