"""Singleton access to the trained diabetes predictor."""
from __future__ import annotations

from typing import Optional

from .pipeline import DiabetesPredictor

_predictor: Optional[DiabetesPredictor] = None


def get_predictor() -> DiabetesPredictor:
    global _predictor
    if _predictor is None:
        _predictor = DiabetesPredictor()
    return _predictor


def reset_predictor() -> None:
    global _predictor
    _predictor = None
