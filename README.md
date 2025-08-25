# Brake Failure — End-to-End MLOps (MLflow + DVC + Docker + GitHub Actions)

This repo is a production-ready template for a **brake failure prediction** project inspired by typical MLflow experiment setups.  
It includes:
- **Synthetic data generator**
- **DVC pipeline**: data → split → features → train → evaluate
- **MLflow tracking**, model registry-ready
- **Flask API** for inference + **Dockerfile**
- **GitHub Actions** CI workflow

---

## Quickstart (Local)

```bash
# 1) Create & activate venv (Windows PowerShell)
py -3.11 -m venv venv
venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure MLflow (optional - defaults to ./mlruns local)
# Set your own tracking URI if you have an MLflow server:
# $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"  # PowerShell
# export MLFLOW_TRACKING_URI="http://127.0.0.1:5000" # bash/zsh
# You can also set:
# export MLFLOW_EXPERIMENT_NAME="brake-failure-exp"

# 4) Generate data, run full pipeline with DVC
dvc repro

# 5) Launch MLflow UI (local tracking)
mlflow ui --port 5001

# 6) Serve the model via Flask (loads last MLflow run from artifacts/last_run.txt)
python app.py
# or production server
gunicorn -b 0.0.0.0:8080 app:app

# 7) Docker (build & run)
docker build -t brake-failure-api:latest .
docker run -p 8080:8080 --env-file .env.example brake-failure-api:latest
```

### Example request

```bash
curl -X POST http://127.0.0.1:8080/predict   -H "Content-Type: application/json"   -d '{
        "speed": 72.0,
        "pressure": 4.3,
        "temperature": 95.0,
        "brake_pad_thickness": 7.8,
        "vibration": 0.14,
        "fluid_level": 0.62,
        "wheel_speed_diff": 1.2
      }'
```

---

## DVC Pipeline

Stages and artifacts are defined in `dvc.yaml`. Parameters are in `params.yaml`.

```text
generate_data -> split -> features -> train -> evaluate
```

- **MLflow** logs params/metrics/model. Latest run id is also stored in `artifacts/last_run.txt` for the API.
- To change model or hyperparameters, edit `params.yaml`.

---

## Repo Structure

```
brake_fail-mlops/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ stages/
│  │  ├─ 00_generate_data.py
│  │  ├─ 01_split.py
│  │  ├─ 02_features.py
│  │  ├─ 03_train.py
│  │  └─ 04_evaluate.py
│  ├─ __init__.py
│  └─ utils.py
├─ artifacts/
├─ notebooks/
├─ app.py
├─ dvc.yaml
├─ params.yaml
├─ requirements.txt
├─ Dockerfile
├─ .gitignore
└─ .github/workflows/ci.yml
```

---

## MLflow Model Registry (optional)

If you run a centralized MLflow Tracking Server with a backend store + model registry, set:

```bash
export MLFLOW_TRACKING_URI="http://<server>:<port>"
export MLFLOW_EXPERIMENT_NAME="brake-failure-exp"
```

Then you can register a model version by promoting the latest run in the UI or via API.

---

## Notes

- The dataset is **synthetic** by default (see `params.yaml`). If you already have real sensor data, drop it at `data/raw/brake_sensor_data.csv` and set `data.generate=false` in `params.yaml`.
- For Windows: set execution policy to allow venv activation if needed.
- This template favors **clarity & reproducibility**.
