import pytest
from ml.validator import validate_biomarkers


def test_validator_ok():
    errors = validate_biomarkers(
        {
            "Pregnancies": 1,
            "Glucose": 120,
            "BloodPressure": 70,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": 28,
            "DiabetesPedigreeFunction": 0.5,
            "Age": 40,
        }
    )
    assert errors == []


def test_validator_missing():
    assert validate_biomarkers({})  # non-empty list
