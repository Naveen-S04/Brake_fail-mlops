import os, yaml, json, numpy as np, pandas as pd

def read_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
