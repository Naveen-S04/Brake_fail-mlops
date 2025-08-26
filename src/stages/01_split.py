import argparse, os
import numpy as np
import pandas as pd
from src.utils import ensure_dir, read_params

FEATURES = ["speed","pressure","temperature","brake_pad_thickness","vibration","fluid_level","wheel_speed_diff"]
TARGET = "brake_failed"

def main(params_path):
    P = read_params(params_path)

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values

    out_dir = "data/processed"
    ensure_dir(out_dir)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    print("Split:", X_train.shape, X_test.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
