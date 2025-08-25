import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import read_params, ensure_dir

def main(params_path):
    P = read_params(params_path)
    df = pd.read_csv("data/raw/brake_sensor_data.csv")
    train_df, test_df = train_test_split(
        df, test_size=P["data"]["test_size"], random_state=P["data"]["random_state"], stratify=df["brake_failed"]
    )
    out_dir = "data/processed"
    ensure_dir(out_dir)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print("Split:", train_df.shape, test_df.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
