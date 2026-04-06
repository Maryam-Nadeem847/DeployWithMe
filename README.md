# Deployment Agent (ML + DL)

This project auto-deploys serialized ML/DL models to Docker + FastAPI.

Supported artifacts:
- ML: `.pkl`, `.pickle`, `.joblib`, `.sav`
- DL: `.pt`, `.pth`, `.onnx`, `.h5`, `.keras`

## 1) Train a model zoo

From project root:

```powershell
pip install -r requirements.txt
pip install xgboost lightgbm catboost
$env:PYTHONPATH = "e:\Cursor projects\deployment_agent\src"
python .\scripts\train_classical_ml_zoo.py
```

Notes:
- Optional packages (`xgboost`, `lightgbm`, `catboost`) are included above to create more models.
- If you skip those installs, script still runs and trains scikit-learn models.

## 2) Print deploy commands

```powershell
$env:PYTHONPATH = "e:\Cursor projects\deployment_agent\src"
python .\scripts\list_deploy_commands.py
```

## 3) Deploy any trained model

Example:

```powershell
$env:PYTHONPATH = "e:\Cursor projects\deployment_agent\src"
python -m deployment_agent "e:\Cursor projects\deployment_agent\trained_models\classification\iris\RandomForestClassifier.joblib" --json
```

## 4) Train DL test models

```powershell
$env:PYTHONPATH = "e:\Cursor projects\deployment_agent\src"
python .\scripts\train_dl_test_models.py
```

This creates:
- `trained_models\dl\iris_torch_full.pt`
- `trained_models\dl\iris_torch_state_dict.pth`
- `trained_models\dl\iris_torch.onnx`
- `trained_models\dl\iris_tf.h5`

## 5) Print DL deploy commands

```powershell
$env:PYTHONPATH = "e:\Cursor projects\deployment_agent\src"
python .\scripts\list_dl_deploy_commands.py
```

On success, output includes:
- `api_url`
- `container_name`
- `image_tag`
- `decision_log`

Health endpoint:

```text
GET {api_url}/health
```

Predict endpoint:

```text
POST {api_url}/predict
{
  "features": [ ... ]
}
```

Notes:
- If `/health` returns `{"status":"load_failed"...}` the model failed to load; check `error` field.
- If `/health` returns `{"status":"state_dict_only"...}` for `.pth`, deployment is healthy but inference is blocked because state dicts need model architecture.
