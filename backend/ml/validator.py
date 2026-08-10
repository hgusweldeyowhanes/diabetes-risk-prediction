"""Biomarker input validation utilities."""
from __future__ import annotations

FEATURE_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 400),
    "BloodPressure": (0, 202),
    "SkinThickness": (0, 200),
    "Insulin": (0, 900),
    "BMI": (0.0, 70.0),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (1, 120),
}


def validate_biomarkers(data: dict) -> list:
    """Return list of human-readable errors; empty if OK."""
    errors = []
    for key, (lo, hi) in FEATURE_RANGES.items():
        if key not in data:
            errors.append(f"Missing field: {key}")
            continue
        try:
            val = float(data[key])
        except (TypeError, ValueError):
            errors.append(f"Invalid number for {key}")
            continue
        if val < lo or val > hi:
            errors.append(f"{key} out of range [{lo}, {hi}]")
    if not errors:
        if (
            float(data.get("Glucose", 0)) == 0
            and float(data.get("BloodPressure", 0)) == 0
            and float(data.get("BMI", 0)) == 0
            and float(data.get("Age", 0)) == 0
        ):
            errors.append("Core vitals cannot all be zero")
    return errors
