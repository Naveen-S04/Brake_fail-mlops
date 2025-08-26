import argparse, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import read_params, ensure_dir

def main(params_path):
    P = read_params(params_path)
    n_samples = P.get("generate", {}).get("n_samples", 20000)

    # Example synthetic data
    df = pd.DataFrame({
        "speed": np.random.randint(0, 120, n_samples),
        "pressure": np.random.uniform(10, 100, n_samples),
        "temperature": np.random.uniform(20, 100, n_samples),
        "brake_pad_thickness": np.random.uniform(1, 20, n_samples),
        "vibration": np.random.uniform(0, 5, n_samples),
        "fluid_level": np.random.uniform(0, 100, n_samples),
        "wheel_speed_diff": np.random.uniform(0, 50, n_samples),
        "brake_failed": np.random.randint(0, 2, n_samples)
    })

    out_dir = "data/processed"
    ensure_dir(out_dir)

    # Split train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Generated synthetic data: {n_samples}")
    print(f"Saved train.csv -> {train_path}")
    print(f"Saved test.csv -> {test_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
