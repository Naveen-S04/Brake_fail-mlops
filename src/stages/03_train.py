import argparse, os, json
import numpy as np
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from src.utils import read_params, ensure_dir

def main(params_path):
    P = read_params(params_path)
    tracking_uri = P["mlflow"].get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI", "")
    exp_name = P["mlflow"].get("experiment_name") or os.environ.get("MLFLOW_EXPERIMENT_NAME", "default")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    with mlflow.start_run() as run:
        params = P["train"]
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)

        # quick train metrics
        train_proba = model.predict_proba(X_train)[:,1]
        auc = roc_auc_score(y_train, train_proba)
        f1 = f1_score(y_train, (train_proba >= 0.5).astype(int))
        mlflow.log_metric("train_auc", float(auc))
        mlflow.log_metric("train_f1", float(f1))

        # log model
        sample = X_train[:5]
        mlflow.sklearn.log_model(model, "model", input_example=sample)

        # save run id for serving
        ensure_dir("artifacts")
        with open("artifacts/last_run.txt","w") as f:
            f.write(run.info.run_id)
        print("Run ID:", run.info.run_id)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
