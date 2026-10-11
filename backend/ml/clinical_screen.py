"""FINDRISC-style screening score (not the ML model).

Uses age, BMI, optional sex/waist/activity/family history.
Independent of the Pima 8-marker stack — shown beside it so men and
people without OGTT labs still get a transparent checklist score.
"""
from __future__ import annotations


def _age_points(age: int) -> int:
    if age < 45:
        return 0
    if age < 55:
        return 2
    if age < 65:
        return 3
    return 4


def _bmi_points(bmi: float) -> int:
    if bmi < 25:
        return 0
    if bmi < 30:
        return 1
    return 3


def _waist_points(waist_cm, sex: str) -> int:
    if waist_cm is None:
        return 0
    try:
        waist = float(waist_cm)
    except (TypeError, ValueError):
        return 0
    if sex == "M":
        if waist < 94:
            return 0
        if waist <= 102:
            return 3
        return 4
    # Female / unknown use female cutoffs (FINDRISC)
    if waist < 80:
        return 0
    if waist <= 88:
        return 3
    return 4


def findrisc_screen(
    *,
    age: int,
    bmi: float,
    sex: str = "U",
    waist_cm=None,
    physically_active: bool = True,
    family_history: bool = False,
) -> dict:
    """Return points, tier, and a short note. Max 20 with the fields we collect."""
    sex = (sex or "U").upper()[:1]
    parts = {
        "age": _age_points(int(age)),
        "bmi": _bmi_points(float(bmi)),
        "waist": _waist_points(waist_cm, sex),
        "activity": 0 if physically_active else 2,
        "family": 5 if family_history else 0,
    }
    score = int(sum(parts.values()))
    if score < 7:
        tier, label = "low", "Low"
    elif score < 12:
        tier, label = "slightly_elevated", "Slightly elevated"
    elif score < 15:
        tier, label = "moderate", "Moderate"
    elif score < 21:
        tier, label = "high", "High"
    else:
        tier, label = "very_high", "Very high"
    return {
        "name": "FINDRISC-style",
        "score": score,
        "max": 20,
        "tier": tier,
        "label": label,
        "parts": parts,
        "note": (
            "Simple lifestyle checklist (age, BMI, waist, activity, family). "
            "It is not a diagnosis and is not the same as the lab-marker model."
        ),
    }
