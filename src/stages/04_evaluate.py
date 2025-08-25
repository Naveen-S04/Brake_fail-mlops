import argparse, os, json
import numpy as np
import mlflow, mlflow.sklearn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

def load_model(run_id):
    uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(uri)

def main(params_path):
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    run_id = open("artifacts/last_run.txt").read().strip()
    model = load_model(run_id)

    proba = model.predict_proba(X_test)[:,1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "test_auc": float(roc_auc_score(y_test, proba)),
        "test_f1": float(f1_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds)),
        "test_recall": float(recall_score(y_test, preds)),
        "test_accuracy": float(accuracy_score(y_test, preds))
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/test_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("Test metrics:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
