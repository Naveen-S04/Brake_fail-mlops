import argparse, os, joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.utils import read_params, ensure_dir

def main(params_path):
    # read params
    P = read_params(params_path)

    # load preprocessed features
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # ensure artifact dir
    model_dir = "models"
    ensure_dir(model_dir)
    model_path = os.path.join(model_dir, "model.pkl")

    # --- MLflow setup ---
    mlflow.set_tracking_uri("file:./mlruns")   # local logging
    mlflow.set_experiment("brake-failure-exp")

    with mlflow.start_run(run_name="rf_train"):
        # model hyperparameters from params.yaml
        n_estimators = P["train"]["n_estimators"]
        max_depth = P["train"]["max_depth"]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        clf.fit(X_train, y_train)

        # predictions
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # log metrics + params
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # save model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(clf, model_path)
        mlflow.sklearn.log_model(clf, "model")

        print(f"✅ Model trained | Accuracy={acc:.3f} F1={f1:.3f}")
        print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
