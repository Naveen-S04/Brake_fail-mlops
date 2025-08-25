import argparse, os
import numpy as np, pandas as pd
from src.utils import read_params, ensure_dir

def generate_synthetic(n_samples=20000, random_state=42):
    rng = np.random.default_rng(random_state)
    speed = rng.normal(60, 20, n_samples).clip(0, 200)
    pressure = rng.normal(5, 2, n_samples).clip(0, 10)
    temperature = rng.normal(80, 30, n_samples).clip(-10, 200)
    brake_pad_thickness = rng.normal(8, 2.5, n_samples).clip(0, 15)
    vibration = rng.normal(0.1, 0.05, n_samples).clip(0, 1)
    fluid_level = rng.uniform(0.2, 1.0, n_samples)
    wheel_speed_diff = rng.normal(0.8, 0.6, n_samples).clip(0, 5)

    # Failure risk function (heuristic)
    score = (
        0.015*speed +
        0.25*pressure +
        0.01*temperature +
        -0.4*brake_pad_thickness +
        3.0*vibration +
        -1.5*fluid_level +
        0.6*wheel_speed_diff +
        rng.normal(0, 0.5, n_samples)
    )
    prob = 1.0/(1.0+np.exp(-score))
    y = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "speed": speed,
        "pressure": pressure,
        "temperature": temperature,
        "brake_pad_thickness": brake_pad_thickness,
        "vibration": vibration,
        "fluid_level": fluid_level,
        "wheel_speed_diff": wheel_speed_diff,
        "brake_failed": y
    })
    return df

def main(params_path):
    P = read_params(params_path)
    raw_dir = "data/raw"
    ensure_dir(raw_dir)
    if P["data"]["generate"]:
        df = generate_synthetic(P["data"]["n_samples"], P["data"]["random_state"])
        df.to_csv(os.path.join(raw_dir, "brake_sensor_data.csv"), index=False)
        print("Generated synthetic data:", len(df))
    else:
        path = os.path.join(raw_dir, "brake_sensor_data.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found and data.generate=false")
        print("Using existing data:", path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params.yaml")
    args = ap.parse_args()
    main(args.params)
