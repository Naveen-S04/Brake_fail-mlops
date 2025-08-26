import argparse, os, joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.utils import read_params, ensure_dir

def main(params_path):
    P = read_params(params_path)

    # load preprocessed data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # hyperparameters
    n_estimators = P["train"]["n_estimators"]
    max_depth = P["train"]["max_depth"]

    # artifact dir
    artifacts_dir = "artifacts"
    ensure_dir(artifacts_dir)
    model_path = os.path.join(artifacts_dir, "model.joblib")

    # MLflow local tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("brake-failure-exp")

    with mlflow.start_run(run_name="rf_train"):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # metrics
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # log params + metrics + model
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(clf, "model")

        # save local copy for DVC
        joblib.dump(clf, model_path)

        print(f"✅ Model trained | Accuracy={acc:.3f} F1={f1:.3f}")
        print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
