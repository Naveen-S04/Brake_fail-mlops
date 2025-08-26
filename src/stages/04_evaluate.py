import argparse, os, joblib, json, numpy as np
from sklearn.metrics import accuracy_score, f1_score
from src.utils import ensure_dir

def main(params_path):
    # load model & test data
    model_path = "artifacts/model.joblib"
    clf = joblib.load(model_path)

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Evaluation done | Accuracy={acc:.3f} F1={f1:.3f}")

    # save metrics
    ensure_dir("artifacts")
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
