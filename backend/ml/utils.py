"""ML utilities."""
from pathlib import Path

import joblib


def load_joblib(path: Path):
    return joblib.load(path)


def save_joblib(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
