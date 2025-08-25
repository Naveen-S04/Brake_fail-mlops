import argparse, os, joblib
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.utils import read_params, ensure_dir

FEATURES = ["speed","pressure","temperature","brake_pad_thickness","vibration","fluid_level","wheel_speed_diff"]
TARGET = "brake_failed"

def build_preprocessor(impute_strategy="median", scale=True):
    steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)

def main(params_path):
    P = read_params(params_path)
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    pre = build_preprocessor(P["features"]["impute_strategy"], P["features"]["scale"])
    X_train_proc = pre.fit_transform(X_train)
    X_test_proc = pre.transform(X_test)

    out_dir = "data/processed"
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train_proc)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test_proc)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    art_dir = "artifacts"
    ensure_dir(art_dir)
    joblib.dump(pre, os.path.join(art_dir, "preprocess.joblib"))
    print("Features prepared:", X_train_proc.shape, X_test_proc.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
